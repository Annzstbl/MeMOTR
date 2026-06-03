"""
加载 checkpoint，对一条视频序列做序贯推理，逐帧保存 stem SE 的 3 通道可视化图，不保存其它内容。

示例：
conda activate hsmot
cd /data1/users/litianhao01/hsmot/MeMOTR
CUDA_VISIBLE_DEVICES=1 \
python debug/20260526/test4.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data33-1 \
  --end-frames all \
  --pca-components 3 \
  --se-apply-sigmoid  

输出目录：debug/20260526/<config>/se_frames/<seq>/
  - se_frame_000001.jpg  PCA 降维到 3 维后映射为 RGB（逐帧、仅有效像素拟合）
  - se_frame_000002.jpg
  - ...
  - se_meta.json

加 --se-apply-sigmoid 可对 sig_raw（conv3d_se_v4）做 sigmoid 后再做 PCA 可视化。
"""

import os
import sys
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DEBUG_DIR not in sys.path:
    sys.path.insert(0, DEBUG_DIR)

from models import build_model
from models.matcher import is_rect_memotr_version
from models.runtime_tracker import RuntimeTracker
from models.utils import load_checkpoint, get_model
from utils.utils import load_yaml_with_inheritance
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.GMC import compute_gmc_sequence
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset

SPECTRAL_BAND_CENTERS_NM = [422.5, 487.5, 550.0, 602.5, 660.0, 725.0, 785.0, 887.2]


class SECaptureHook:
    """从 backbone.stem_conv 捕获 SE 输出（sig_raw 或 gate）。"""

    def __init__(self, model: nn.Module):
        inner = get_model(model)
        backbone = getattr(inner, "backbone", None)
        if backbone is None or not hasattr(backbone, "backbone"):
            raise RuntimeError("Model backbone does not expose stem_conv for SE capture.")
        stem_conv = getattr(backbone.backbone, "stem_conv", None)
        if stem_conv is None:
            raise RuntimeError("stem_conv not found; STEM may not be conv3d_se*.")
        self.stem_returns_gate = (
            hasattr(stem_conv, "return_before_sigmoid") and not stem_conv.return_before_sigmoid
        )
        self._se_tensor: Optional[torch.Tensor] = None
        self._handle = stem_conv.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _inputs, output) -> None:
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("stem_conv forward did not return (feature, se_weights).")
        self._se_tensor = output[1]

    def pop(self) -> Optional[torch.Tensor]:
        se = self._se_tensor
        self._se_tensor = None
        return se

    def close(self) -> None:
        self._handle.remove()


def _downsample_valid_mask(pad_mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if pad_mask.dim() == 2:
        pad_mask = pad_mask.unsqueeze(0).unsqueeze(0)
    elif pad_mask.dim() == 3:
        pad_mask = pad_mask.unsqueeze(1)
    down = F.interpolate(pad_mask.float(), size=target_hw, mode="nearest")
    return ~down[0, 0].to(dtype=torch.bool)


def _prepare_se_tensor(se: torch.Tensor, stem_returns_gate: bool, apply_sigmoid: bool) -> torch.Tensor:
    if stem_returns_gate or not apply_sigmoid:
        return se
    return torch.sigmoid(se)


def se_pca_to_bgr(
    se: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    n_components: int = 3,
) -> np.ndarray:
    """[C,H,W] SE -> BGR uint8 [H,W,3]：对有效像素做 PCA，PC1/2/3 映射 R/G/B。"""
    se = np.asarray(se, dtype=np.float64)
    if se.ndim != 3:
        raise ValueError(f"Expected SE [C,H,W], got {se.shape}")
    c, h, w = se.shape
    n_components = min(n_components, c)

    valid = np.ones((h, w), dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (h, w):
            raise ValueError(f"valid_mask shape {valid.shape} != SE spatial {(h, w)}")

    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    if not valid.any() or n_components < 1:
        return bgr

    flat_valid = valid.reshape(-1)
    x = se.reshape(c, -1).T[flat_valid]
    n = x.shape[0]
    if n < 2:
        return bgr

    x_mean = x.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    cov = (x_centered.T @ x_centered) / max(n - 1, 1)
    u, _, _ = np.linalg.svd(cov, full_matrices=False)
    w_pca = u[:, :n_components]
    x_proj = x_centered @ w_pca

    rgb = np.zeros((n, 3), dtype=np.float32)
    for i in range(n_components):
        col = x_proj[:, i]
        vmin, vmax = float(col.min()), float(col.max())
        if vmax > vmin:
            rgb[:, i] = np.clip((col - vmin) / (vmax - vmin), 0.0, 1.0)

    bgr_flat = bgr.reshape(-1, 3)
    for i in range(min(3, n_components)):
        bgr_flat[flat_valid, 2 - i] = (rgb[:, i] * 255.0).astype(np.uint8)
    return bgr


def _upscale_bgr(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    if max(h, w) >= max_side:
        return img
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def save_frame_se_pca(
    se: torch.Tensor,
    out_dir: str,
    frame_num: int,
    valid_mask: Optional[torch.Tensor] = None,
    vis_max_size: int = 1200,
    n_components: int = 3,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    x = se.detach().cpu().numpy()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected SE [C,H,W] or [1,C,H,W], got {tuple(se.shape)}")

    mask_np = None
    if valid_mask is not None:
        mask_np = valid_mask.detach().cpu().numpy()

    bgr = se_pca_to_bgr(x, valid_mask=mask_np, n_components=n_components)
    bgr = _upscale_bgr(bgr, vis_max_size)
    path = os.path.join(out_dir, f"se_frame_{frame_num:06d}.jpg")
    cv2.imwrite(path, bgr)
    return path


def _resolve_train_config_path(train_config_arg: str) -> str:
    if os.path.isabs(train_config_arg):
        return train_config_arg
    return os.path.abspath(os.path.join(CURRENT_DIR, train_config_arg))


def _resolve_checkpoint_path(checkpoint_arg: str, train_config: dict, train_cfg_path: str) -> str:
    if os.path.isabs(checkpoint_arg):
        return checkpoint_arg
    outputs_dir = train_config.get("OUTPUTS_DIR", "")
    if not outputs_dir:
        raise ValueError("Relative checkpoint path requires OUTPUTS_DIR in train config.")
    if not os.path.isabs(outputs_dir):
        outputs_dir = os.path.abspath(os.path.join(os.path.dirname(train_cfg_path), outputs_dir))
    return os.path.abspath(os.path.join(outputs_dir, checkpoint_arg))


def _normalize_img_format(fmt: str) -> Tuple[str, str]:
    key = fmt.strip().lower()
    if key in ("npy2jpg", "3jpg", "jpg"):
        return "npy2jpg", "3JPG"
    if key in ("npy",):
        return "npy", "NPY"
    raise ValueError(f"Unsupported img format: {fmt!r}. Use npy2jpg or npy.")


def _resolve_img_format(img_format_arg: Optional[str], train_config: dict) -> Tuple[str, str]:
    if img_format_arg:
        return _normalize_img_format(img_format_arg)
    cfg_type = str(train_config.get("DATASET_TYPE", "3JPG")).upper()
    if cfg_type == "3JPG":
        return "npy2jpg", "3JPG"
    return "npy", "NPY"


def _resolve_img_root(data_root: str, base_name: str, split: str, img_subdir: str) -> str:
    return os.path.join(data_root, base_name, split, img_subdir)


def _resolve_config_root(train_cfg_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        return os.path.abspath(output_dir)
    config_name = os.path.splitext(os.path.basename(train_cfg_path))[0]
    return os.path.join(CURRENT_DIR, config_name)


def _init_track_instances_static(train_config: dict, model: nn.Module) -> None:
    rect_bbox = is_rect_memotr_version(train_config.get("MEMOTR_VERSION", ""))
    TrackInstances.set_static_properties(
        use_spectral_decoder=train_config.get("DECODER_SPECTRAL", True),
        use_dab=train_config["USE_DAB"],
        use_q_spec=bool(getattr(get_model(model), "use_q_spec", False)),
        bbox_dim=4 if rect_bbox else 5,
    )


def run_forward_save_se(
    model: nn.Module,
    train_config: dict,
    seq_dir: str,
    npy2rgb: bool,
    dataset_type: Optional[str],
    start_frame: int,
    end_frame: Optional[int],
    se_out_dir: str,
    se_hook: SECaptureHook,
    se_apply_sigmoid: bool = False,
    vis_max_size: int = 1200,
    pca_components: int = 3,
) -> List[str]:
    device = next(model.parameters()).device
    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb, dataset_type=dataset_type)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {seq_dir}")

    start_idx = max(0, start_frame - 1)
    if start_idx >= len(dataset):
        raise RuntimeError(
            f"Start frame {start_frame} exceeds dataset length {len(dataset)} at {seq_dir}"
        )

    if end_frame is None:
        end_idx = len(dataset) - 1
    else:
        end_idx = min(len(dataset) - 1, end_frame - 1)
    if end_idx < start_idx:
        raise RuntimeError(
            f"Invalid frame range [{start_frame}, {end_frame}] for dataset length {len(dataset)}"
        )

    num_frames = end_idx - start_idx + 1
    actual_end_frame = start_frame + num_frames - 1
    print(
        f"[Info] Processing frames {start_frame}-{actual_end_frame} "
        f"({num_frames} frames, dataset total={len(dataset)})"
    )

    inner_model = get_model(model)
    tracker = RuntimeTracker(
        det_score_thresh=0.5,
        track_score_thresh=0.5,
        miss_tolerance=30,
        use_motion=False,
        motion_min_length=3,
        motion_max_length=5,
        visualize=False,
        use_dab=train_config["USE_DAB"],
        decoder_spectral=train_config.get("DECODER_SPECTRAL", True),
    )

    tracks = [
        TrackInstances(
            hidden_dim=inner_model.hidden_dim,
            num_classes=inner_model.num_classes,
        ).to(device)
    ]

    use_prior_map = (
        hasattr(inner_model, "scem_module")
        and inner_model.scem_module is not None
        and inner_model.scem_module.prior_mode is not None
    )

    saved_paths: List[str] = []
    prev_frame = None

    model.eval()
    with torch.no_grad():
        for i in range(num_frames):
            dataset_idx = start_idx + i
            frame_num = start_frame + i
            print(f"[Info] forward frame {frame_num} (index {dataset_idx})")

            image, _ = dataset[dataset_idx][0]
            frame = tensor_list_to_nested_tensor([image]).to(device)

            gmc = None
            if use_prior_map:
                if prev_frame is not None:
                    gmc = compute_gmc_sequence(
                        images=[prev_frame[0], frame.tensors[0]],
                        method="sparseOptFlow",
                        downscale=1,
                    )[-1]
                else:
                    gmc = np.eye(2, 3, dtype=np.float32)
                prev_frame = frame.tensors.detach().clone()
                gmc = torch.tensor(gmc, dtype=torch.float32).unsqueeze(0).to(device)

            forward_kwargs = dict(frame=frame, tracks=tracks, debug=False)
            if gmc is not None:
                forward_kwargs["gmc"] = gmc
            res = model(**forward_kwargs)

            se_raw = se_hook.pop()
            if se_raw is None:
                raise RuntimeError(f"SE not captured at frame {frame_num}.")
            se = _prepare_se_tensor(
                se_raw,
                stem_returns_gate=se_hook.stem_returns_gate,
                apply_sigmoid=se_apply_sigmoid,
            )
            valid_mask = _downsample_valid_mask(frame.masks[0], se.shape[-2:])
            path = save_frame_se_pca(
                se,
                se_out_dir,
                frame_num,
                valid_mask=valid_mask,
                vis_max_size=vis_max_size,
                n_components=pca_components,
            )
            saved_paths.append(path)
            print(f"[Info] saved {path} se_shape={tuple(se.shape)} pca_components={pca_components}")

            previous_tracks, new_tracks = tracker.update(model_outputs=res, tracks=tracks)
            tracks = inner_model.postprocess_single_frame(previous_tracks, new_tracks, None)
            del res, frame, image

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Load model, run one video sequence, save per-frame stem SE as PCA RGB JPG."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="20260511-2.yaml",
        help="Train config yaml (absolute, or relative to this script dir).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.pth",
        help="Checkpoint path (absolute, or relative to OUTPUTS_DIR in train config).",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override DATA_ROOT from train config.")
    parser.add_argument("--dataset-name", type=str, default="hsmot_8ch")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seq", type=str, default="data39-1", help="Single sequence name.")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument(
        "--end-frames",
        type=str,
        default="all",
        help="Last frame index (inclusive), or 'all' for full sequence.",
    )
    parser.add_argument(
        "--img-format",
        type=str,
        default="npy2jpg",
        choices=["npy2jpg", "npy"],
    )
    parser.add_argument("--npy2rgb", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--se-apply-sigmoid",
        action="store_true",
        help="对 sig_raw（conv3d_se_v4）做 sigmoid 后再可视化；默认使用 raw SE。",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="PCA 主成分数（映射到 RGB 时最多取前 3 个）。",
    )
    parser.add_argument(
        "--vis-max-size",
        type=int,
        default=1200,
        help="输出 JPG 短边最大像素（上采样，便于查看）。0 表示不放大。",
    )

    args = parser.parse_args()

    seq = args.seq.strip()
    if not seq:
        raise ValueError("--seq must be a non-empty sequence name.")
    if "," in seq:
        raise ValueError("test4.py only supports one sequence; remove comma in --seq.")

    train_cfg_path = _resolve_train_config_path(args.train_config)
    if not os.path.isfile(train_cfg_path):
        raise FileNotFoundError(f"Config not found: {train_cfg_path}")
    train_config = load_yaml_with_inheritance(path=train_cfg_path)

    data_root_arg = args.data_root
    if data_root_arg is None:
        args.data_root = train_config.get("DATA_ROOT", "") or ""
    if not args.data_root:
        raise ValueError("DATA_ROOT is empty: set in train config or pass --data-root.")
    if not os.path.isabs(args.data_root):
        if data_root_arg is None:
            args.data_root = os.path.abspath(
                os.path.join(os.path.dirname(train_cfg_path), args.data_root)
            )
        else:
            args.data_root = os.path.abspath(args.data_root)

    end_frame: Optional[int] = None
    end_raw = args.end_frames
    if end_raw is not None and end_raw.strip().lower() not in ("none", "all"):
        end_frame = int(end_raw.strip())

    if args.start_frame < 1:
        raise ValueError(f"--start-frame must be >= 1, got {args.start_frame}")
    if end_frame is not None and end_frame < args.start_frame:
        raise ValueError(
            f"--end-frames ({end_frame}) must be >= --start-frame ({args.start_frame})"
        )

    config_root = _resolve_config_root(train_cfg_path, args.output_dir)
    base_name = args.dataset_name.replace("_8ch", "") if "hsmot" in args.dataset_name else args.dataset_name
    img_subdir, dataset_type = _resolve_img_format(args.img_format, train_config)
    img_root = _resolve_img_root(args.data_root, base_name, args.split, img_subdir)
    seq_dir = os.path.join(img_root, seq)
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence dir not found: {seq_dir}")

    se_out_dir = os.path.join(config_root, "se_frames", seq)
    os.makedirs(se_out_dir, exist_ok=True)
    print(f"[Info] img format={args.img_format} -> {seq_dir}")
    print(f"[Info] SE output -> {se_out_dir}")

    print(f"[Info] Building model from: {train_cfg_path}")
    train_config["MEMOTR_VERSION"] = "20260511_figure"
    model = build_model(config=train_config)
    model.to(torch.device(args.device))

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[Info] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    _init_track_instances_static(train_config, model)

    if args.pca_components < 1:
        raise ValueError(f"--pca-components must be >= 1, got {args.pca_components}")

    se_hook = SECaptureHook(model)
    print(
        f"[Info] SE capture: stem_returns_gate={se_hook.stem_returns_gate}, "
        f"apply_sigmoid={args.se_apply_sigmoid}, pca_components={args.pca_components}"
    )

    try:
        saved_paths = run_forward_save_se(
            model=model,
            train_config=train_config,
            seq_dir=seq_dir,
            npy2rgb=args.npy2rgb,
            dataset_type=dataset_type,
            start_frame=args.start_frame,
            end_frame=end_frame,
            se_out_dir=se_out_dir,
            se_hook=se_hook,
            se_apply_sigmoid=args.se_apply_sigmoid,
            vis_max_size=args.vis_max_size,
            pca_components=args.pca_components,
        )
    finally:
        se_hook.close()

    image_shape = None
    if saved_paths:
        sample = cv2.imread(saved_paths[0])
        if sample is not None:
            image_shape = list(sample.shape)

    meta = {
        "seq": seq,
        "start_frame": args.start_frame,
        "end_frame": end_frame,
        "num_frames_saved": len(saved_paths),
        "output_format": "jpg_bgr",
        "vis_method": "pca_per_frame",
        "pca_components": args.pca_components,
        "image_shape_hwc": image_shape,
        "vis_max_size": args.vis_max_size,
        "stem_returns_gate": se_hook.stem_returns_gate,
        "se_apply_sigmoid": args.se_apply_sigmoid,
        "band_centers_nm": SPECTRAL_BAND_CENTERS_NM,
        "config": train_cfg_path,
        "checkpoint": checkpoint_path,
        "files": saved_paths,
    }
    meta_path = os.path.join(se_out_dir, "se_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Saved {len(saved_paths)} SE frames under: {se_out_dir}")
    print(f"[Done] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
