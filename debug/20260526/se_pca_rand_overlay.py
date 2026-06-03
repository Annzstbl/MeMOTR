"""
每条视频序列随机抽 N 帧，保存 stem（最浅层）SE 的 PCA-RGB 图，以透明度叠加在原图 RGB 上。
各帧独立前向（无序贯跟踪 / GMC），stem SE 仅依赖当前帧图像。

示例：
conda activate hsmot
cd /data1/users/litianhao01/hsmot/MeMOTR
CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/se_pca_rand_overlay.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq all \
  --frames-per-seq 5 \
  --seed 42

输出：debug/20260526/<config>/se_overlay/<seq>/
  - <seq>__frame000123__rgb.jpg
  - <seq>__frame000123__se_pca.jpg
  - <seq>__frame000123__overlay.jpg
  - se_meta.json

叠加时按有效区域裁剪 SE（去掉 pad 对应区域），再缩放到原图尺寸，避免 padding 拉伸伪影。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DEBUG_DIR not in sys.path:
    sys.path.insert(0, DEBUG_DIR)

from models import build_model
from models.utils import load_checkpoint, get_model
from utils.utils import load_yaml_with_inheritance
from utils.nested_tensor import tensor_list_to_nested_tensor
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset

# 复用 test4 的 stem SE 捕获与 PCA 可视化
_test4_path = os.path.join(CURRENT_DIR, "test4.py")
_spec = importlib.util.spec_from_file_location("debug_test4", _test4_path)
_test4 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_test4)

SECaptureHook = _test4.SECaptureHook
se_pca_to_bgr = _test4.se_pca_to_bgr
_prepare_se_tensor = _test4._prepare_se_tensor
_downsample_valid_mask = _test4._downsample_valid_mask
_resolve_train_config_path = _test4._resolve_train_config_path
_resolve_checkpoint_path = _test4._resolve_checkpoint_path
_resolve_img_format = _test4._resolve_img_format
_resolve_img_root = _test4._resolve_img_root
_resolve_config_root = _test4._resolve_config_root
_init_track_instances_static = _test4._init_track_instances_static
SPECTRAL_BAND_CENTERS_NM = _test4.SPECTRAL_BAND_CENTERS_NM


def _ori_to_vis_rgb(ori_image: np.ndarray) -> np.ndarray:
    vis = ori_image.copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)
    c = vis.shape[2]
    if c == 8:
        vis = vis[:, :, [4, 2, 1]]
    elif c >= 3:
        vis = vis[:, :, :3]
    else:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    return vis


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _align_se_to_effective_rgb(
    se_bgr: np.ndarray,
    valid_mask: Optional[np.ndarray],
    eff_h: int,
    eff_w: int,
    pad_h: int,
    pad_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 SE PCA 图裁剪到与有效原图 [0:eff_h, 0:eff_w] 对应的区域，再缩放到 (eff_w, eff_h)。
    避免把 pad 区域一并拉伸到原图尺寸。
    """
    h_se, w_se = se_bgr.shape[:2]
    h_crop = min(h_se, max(1, int(round(eff_h * h_se / pad_h))))
    w_crop = min(w_se, max(1, int(round(eff_w * w_se / pad_w))))
    se_crop = se_bgr[:h_crop, :w_crop]
    se_up = cv2.resize(se_crop, (eff_w, eff_h), interpolation=cv2.INTER_LINEAR)

    if valid_mask is not None:
        vm = np.asarray(valid_mask[:h_crop, :w_crop], dtype=np.uint8)
        valid_up = cv2.resize(vm, (eff_w, eff_h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        valid_up = np.ones((eff_h, eff_w), dtype=bool)
    return se_up, valid_up


def save_se_pca_triplet(
    se: torch.Tensor,
    ori_image: np.ndarray,
    out_dir: str,
    frame_stem: str,
    pad_h: int,
    pad_w: int,
    valid_mask: Optional[torch.Tensor] = None,
    overlay_alpha: float = 0.55,
    pca_components: int = 3,
    vis_max_size: int = 1200,
) -> Dict[str, str]:
    """保存 rgb / se_pca / overlay 三张图（均已对齐到无 pad 原图尺寸）。"""
    x = se.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    mask_np = None
    if valid_mask is not None:
        mask_np = valid_mask.detach().cpu().numpy()

    se_bgr = se_pca_to_bgr(x.numpy(), valid_mask=mask_np, n_components=pca_components)
    rgb = np.ascontiguousarray(_ori_to_vis_rgb(ori_image))
    eff_h, eff_w = rgb.shape[:2]

    se_bgr_up, valid_up = _align_se_to_effective_rgb(
        se_bgr, mask_np, eff_h, eff_w, pad_h, pad_w
    )
    se_rgb = cv2.cvtColor(se_bgr_up, cv2.COLOR_BGR2RGB)

    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    blend = rgb.astype(np.float32)
    valid_3c = valid_up[..., None]
    blend = np.where(
        valid_3c,
        alpha * se_rgb.astype(np.float32) + (1.0 - alpha) * blend,
        blend,
    )
    blend = np.clip(blend, 0, 255).astype(np.uint8)

    rgb_out = _resize_max_side(rgb, vis_max_size)
    se_out = _resize_max_side(se_rgb, vis_max_size)
    blend_out = _resize_max_side(blend, vis_max_size)

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "rgb": os.path.join(out_dir, f"{frame_stem}__rgb.jpg"),
        "se_pca": os.path.join(out_dir, f"{frame_stem}__se_pca.jpg"),
        "overlay": os.path.join(out_dir, f"{frame_stem}__overlay.jpg"),
    }
    cv2.imwrite(paths["rgb"], cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["se_pca"], cv2.cvtColor(se_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["overlay"], cv2.cvtColor(blend_out, cv2.COLOR_RGB2BGR))
    return paths


def _list_sequences(img_root: str, seq_arg: str) -> List[str]:
    if seq_arg.strip().lower() == "all":
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"image root not found: {img_root}")
        return sorted(
            d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))
        )
    return [s.strip() for s in seq_arg.split(",") if s.strip()]


def _sample_frame_numbers(n_total: int, n_sample: int, rng: random.Random) -> List[int]:
    """1-indexed frame numbers."""
    if n_total <= 0:
        return []
    k = min(n_sample, n_total)
    return sorted(rng.sample(range(1, n_total + 1), k))


def _empty_tracks(model: nn.Module, device: torch.device) -> List[TrackInstances]:
    inner = get_model(model)
    tracks = TrackInstances(
        hidden_dim=inner.hidden_dim,
        num_classes=inner.num_classes,
    )
    return [tracks.to(device)]


def run_seq_random_se_overlay(
    model: nn.Module,
    seq: str,
    seq_dir: str,
    out_dir: str,
    se_hook: SECaptureHook,
    frame_numbers: List[int],
    npy2rgb: bool = False,
    dataset_type: Optional[str] = None,
    se_apply_sigmoid: bool = False,
    overlay_alpha: float = 0.55,
    pca_components: int = 3,
    vis_max_size: int = 1200,
) -> List[Dict[str, str]]:
    if not frame_numbers:
        return []

    device = next(model.parameters()).device
    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb, dataset_type=dataset_type)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {seq_dir}")

    inner_model = get_model(model)
    use_prior_map = (
        hasattr(inner_model, "scem_module")
        and inner_model.scem_module is not None
        and inner_model.scem_module.prior_mode is not None
    )

    print(
        f"[Info] seq={seq}: independent forward on frames {frame_numbers} "
        f"(dataset={len(dataset)})"
    )

    saved_paths: List[str] = []
    model.eval()
    with torch.no_grad():
        for frame_num in frame_numbers:
            if frame_num < 1 or frame_num > len(dataset):
                print(f"[Warn] frame {frame_num} out of range [1, {len(dataset)}], skip")
                continue

            dataset_idx = frame_num - 1
            print(f"[Info] forward frame {frame_num} (index {dataset_idx}, non-sequential)")

            image, ori_image = dataset[dataset_idx][0]
            frame = tensor_list_to_nested_tensor([image]).to(device)
            tracks = _empty_tracks(model, device)

            forward_kwargs = dict(frame=frame, tracks=tracks, debug=False)
            if use_prior_map:
                gmc = np.eye(2, 3, dtype=np.float32)
                forward_kwargs["gmc"] = torch.tensor(gmc, dtype=torch.float32).unsqueeze(0).to(device)

            model(**forward_kwargs)

            se_raw = se_hook.pop()
            if se_raw is None:
                raise RuntimeError(f"SE not captured at frame {frame_num}.")
            se = _prepare_se_tensor(
                se_raw,
                stem_returns_gate=se_hook.stem_returns_gate,
                apply_sigmoid=se_apply_sigmoid,
            )
            valid_mask = _downsample_valid_mask(frame.masks[0], se.shape[-2:])
            pad_h, pad_w = int(frame.tensors.shape[-2]), int(frame.tensors.shape[-1])
            frame_stem = f"{seq}__frame{frame_num:06d}"
            paths = save_se_pca_triplet(
                se,
                ori_image=ori_image,
                out_dir=out_dir,
                frame_stem=frame_stem,
                pad_h=pad_h,
                pad_w=pad_w,
                valid_mask=valid_mask,
                overlay_alpha=overlay_alpha,
                pca_components=pca_components,
                vis_max_size=vis_max_size,
            )
            saved_paths.append(paths)
            print(
                f"[Info] saved frame {frame_num}: rgb={paths['rgb']}, "
                f"se={paths['se_pca']}, overlay={paths['overlay']} "
                f"(se={tuple(se.shape)}, pad=({pad_h},{pad_w}), eff=({ori_image.shape[0]},{ori_image.shape[1]}))"
            )
            del frame, image

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-sequence random frames: stem SE PCA overlay on RGB (per-frame, non-sequential)."
    )
    parser.add_argument("--train-config", type=str, default="20260511-2.yaml")
    parser.add_argument("--checkpoint", type=str, default="last.pth")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="hsmot_8ch")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--seq",
        type=str,
        default="all",
        help="Comma-separated sequence names, or 'all' for every folder under img root.",
    )
    parser.add_argument(
        "--frames-per-seq",
        type=int,
        default=5,
        help="每条序列随机抽取的帧数（默认 5）。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机抽帧种子。")
    parser.add_argument("--img-format", type=str, default="npy2jpg", choices=["npy2jpg", "npy"])
    parser.add_argument("--npy2rgb", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--se-apply-sigmoid", action="store_true")
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="SE PCA 叠加权重，越大 SE 越明显（0~1）。",
    )
    parser.add_argument("--vis-max-size", type=int, default=1200)
    args = parser.parse_args()

    if args.frames_per_seq < 1:
        raise ValueError(f"--frames-per-seq must be >= 1, got {args.frames_per_seq}")
    if args.pca_components < 1:
        raise ValueError(f"--pca-components must be >= 1, got {args.pca_components}")

    train_cfg_path = _resolve_train_config_path(args.train_config)
    train_config = load_yaml_with_inheritance(path=train_cfg_path)

    if args.data_root is None:
        args.data_root = train_config.get("DATA_ROOT", "") or ""
    if not args.data_root:
        raise ValueError("DATA_ROOT is empty.")
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(
            os.path.join(os.path.dirname(train_cfg_path), args.data_root)
        )

    config_root = _resolve_config_root(train_cfg_path, args.output_dir)
    overlay_root = os.path.join(config_root, "se_overlay")
    os.makedirs(overlay_root, exist_ok=True)

    base_name = args.dataset_name.replace("_8ch", "") if "hsmot" in args.dataset_name else args.dataset_name
    img_subdir, dataset_type = _resolve_img_format(args.img_format, train_config)
    img_root = _resolve_img_root(args.data_root, base_name, args.split, img_subdir)
    print(f"[Info] img format={args.img_format} -> {img_root}")

    seq_list = _list_sequences(img_root, args.seq)
    if not seq_list:
        raise ValueError("No sequences to process.")
    print(f"[Info] {len(seq_list)} sequence(s), {args.frames_per_seq} random frame(s) each, seed={args.seed}")

    train_config["MEMOTR_VERSION"] = "20260511_figure"
    model = build_model(config=train_config)
    model.to(torch.device(args.device))
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    print(f"[Info] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    _init_track_instances_static(train_config, model)

    se_hook = SECaptureHook(model)
    rng = random.Random(args.seed)
    all_meta: Dict[str, Any] = {
        "config": train_cfg_path,
        "checkpoint": checkpoint_path,
        "vis_method": "stem_se_pca_triplet_rgb_se_overlay",
        "inference_mode": "per_frame_independent",
        "align_to_effective_crop": True,
        "pca_components": args.pca_components,
        "overlay_alpha": args.overlay_alpha,
        "frames_per_seq": args.frames_per_seq,
        "seed": args.seed,
        "stem_returns_gate": se_hook.stem_returns_gate,
        "se_apply_sigmoid": args.se_apply_sigmoid,
        "band_centers_nm": SPECTRAL_BAND_CENTERS_NM,
        "per_seq": [],
    }

    try:
        for seq in seq_list:
            seq_dir = os.path.join(img_root, seq)
            if not os.path.isdir(seq_dir):
                print(f"[Warn] Skip missing seq dir: {seq_dir}")
                continue

            n_total = len(SeqDataset(seq_dir=seq_dir, npy2rgb=args.npy2rgb, dataset_type=dataset_type))
            frame_numbers = _sample_frame_numbers(n_total, args.frames_per_seq, rng)
            seq_out = os.path.join(overlay_root, seq)
            os.makedirs(seq_out, exist_ok=True)

            saved = run_seq_random_se_overlay(
                model=model,
                seq=seq,
                seq_dir=seq_dir,
                out_dir=seq_out,
                se_hook=se_hook,
                frame_numbers=frame_numbers,
                npy2rgb=args.npy2rgb,
                dataset_type=dataset_type,
                se_apply_sigmoid=args.se_apply_sigmoid,
                overlay_alpha=args.overlay_alpha,
                pca_components=args.pca_components,
                vis_max_size=args.vis_max_size,
            )

            seq_meta = {
                "seq": seq,
                "dataset_frames": n_total,
                "sampled_frames": frame_numbers,
                "files": saved,
            }
            meta_path = os.path.join(seq_out, "se_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(seq_meta, f, indent=2, ensure_ascii=False)
            all_meta["per_seq"].append(seq_meta)
            print(f"[Info] seq={seq} done, {len(saved)} frame(s) x3 images -> {seq_out}")
    finally:
        se_hook.close()

    summary_path = os.path.join(overlay_root, "se_overlay_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] SE PCA overlays under: {overlay_root}")
    print(f"[Done] Summary: {summary_path}")


if __name__ == "__main__":
    main()
