"""
data30-10（或其它序列）前 N 帧：可视化 SCEM 中每个 prototype 在图像上的空间响应。

响应来源：SCEM.forward(return_debug=True) 中的 spectral_evidence_multilevel[lvl][B,K,H,W]，
即 posterior 路由后、与 prototype 字典匹配得到的 signed spatial evidence（见 SCEM.py）。

各帧独立前向，无序贯跟踪。默认保存：原图 RGB、每个 prototype 的热力图、叠加图、以及总览 grid。

示例：
conda activate hsmot
cd /data1/users/litianhao01/hsmot/MeMOTR
CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/scem_proto_vis.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data30-10 \
  --end-frames 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.cm as mpl_cm
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
from models.utils import load_checkpoint, get_model
from utils.utils import load_yaml_with_inheritance
from utils.nested_tensor import tensor_list_to_nested_tensor
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset

_test4_path = os.path.join(CURRENT_DIR, "test4.py")
_spec = importlib.util.spec_from_file_location("debug_test4", _test4_path)
_test4 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_test4)

_resolve_train_config_path = _test4._resolve_train_config_path
_resolve_checkpoint_path = _test4._resolve_checkpoint_path
_resolve_img_format = _test4._resolve_img_format
_resolve_img_root = _test4._resolve_img_root
_resolve_config_root = _test4._resolve_config_root
_init_track_instances_static = _test4._init_track_instances_static

# BGR LUT，256 级
_CMAP_LUT_BGR: Dict[str, np.ndarray] = {}


def _get_cmap_lut_bgr(name: str) -> np.ndarray:
    key = name.strip().lower()
    if key not in _CMAP_LUT_BGR:
        if key in ("coolwarm", "bwr", "rdbu"):
            mpl_name = "coolwarm" if key != "rdbu" else "RdBu_r"
        elif key in ("positive", "magma", "activation"):
            mpl_name = "magma"
        elif key in ("viridis",):
            mpl_name = "viridis"
        else:
            raise ValueError(f"Unknown cmap: {name!r}")
        rgba = mpl_cm.get_cmap(mpl_name)(np.linspace(0.0, 1.0, 256))
        rgb = (rgba[:, :3] * 255.0).astype(np.uint8)
        _CMAP_LUT_BGR[key] = rgb[:, ::-1]  # RGB -> BGR
    return _CMAP_LUT_BGR[key]


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
    return np.ascontiguousarray(vis)


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    return cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _downsample_pad_mask(pad_mask: torch.Tensor, target_hw: Tuple[int, int]) -> np.ndarray:
    """frame.masks: True=padding -> 返回 bool [H,W] True=padding。"""
    if pad_mask.dim() == 2:
        m = pad_mask.unsqueeze(0).unsqueeze(0)
    elif pad_mask.dim() == 3:
        m = pad_mask.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected pad_mask shape {tuple(pad_mask.shape)}")
    down = F.interpolate(m.float(), size=target_hw, mode="nearest")
    return down[0, 0].detach().cpu().numpy().astype(bool)


def _align_map_to_effective(
    arr2d: np.ndarray,
    eff_h: int,
    eff_w: int,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    h_map, w_map = arr2d.shape
    h_crop = min(h_map, max(1, int(round(eff_h * h_map / pad_h))))
    w_crop = min(w_map, max(1, int(round(eff_w * w_map / pad_w))))
    cropped = arr2d[:h_crop, :w_crop].astype(np.float32)
    return cv2.resize(cropped, (eff_w, eff_h), interpolation=cv2.INTER_LINEAR)


def _valid_mask_from_pad(pad_mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if pad_mask is not None and pad_mask.shape == shape:
        valid = ~pad_mask.astype(bool)
    return valid


def _prototype_value_range(
    arr: np.ndarray,
    valid: np.ndarray,
    cmap_mode: str,
    percentile: float,
) -> Tuple[float, float]:
    """单个 prototype 在有效像素上独立归一化（分位数定范围）。"""
    if not valid.any():
        return -1.0, 1.0

    vals = arr[valid].reshape(-1).astype(np.float64)
    if cmap_mode in ("positive", "magma", "activation", "viridis"):
        vmax = float(np.percentile(vals, percentile))
        return 0.0, max(vmax, 1e-6)

    abs_p = float(np.percentile(np.abs(vals), percentile))
    lim = max(abs_p, 1e-6)
    return -lim, lim


def _map_to_color_bgr(
    arr2d: np.ndarray,
    valid: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_mode: str,
) -> np.ndarray:
    """标量 evidence -> BGR；pad 区域保持深灰底。"""
    arr = np.asarray(arr2d, dtype=np.float32)
    h, w = arr.shape
    color = np.full((h, w, 3), 32, dtype=np.uint8)

    if not valid.any() or vmax <= vmin:
        return color

    if cmap_mode in ("positive", "magma", "activation", "viridis"):
        arr_vis = np.maximum(arr, 0.0)
        norm = np.clip((arr_vis - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        mid = 0.5 * (vmin + vmax)
        half = max((vmax - vmin) * 0.5, 1e-6)
        norm = np.clip((arr - mid) / half, -1.0, 1.0)
        norm = (norm + 1.0) * 0.5

    lut = _get_cmap_lut_bgr(cmap_mode)
    idx = (norm * 255.0).astype(np.uint8)
    color = lut[idx]
    color[~valid] = 32
    return color


def _overlay_on_rgb(
    rgb: np.ndarray,
    heat_bgr: np.ndarray,
    valid: np.ndarray,
    alpha: float,
    min_strength: float = 0.08,
) -> np.ndarray:
    """仅在响应足够强的有效像素上叠加，减弱“整图发脏”。"""
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = rgb.astype(np.float32)
    gray = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    strength = np.abs(gray - (32.0 / 255.0))
    mask = valid & (strength >= min_strength)
    m = mask[..., None]
    a = float(np.clip(alpha, 0.0, 1.0))
    out = np.where(m, a * heat_rgb + (1.0 - a) * base, base)
    return np.clip(out, 0, 255).astype(np.uint8)


def _label_tile(tile: np.ndarray, text: str) -> np.ndarray:
    out = tile.copy()
    cv2.putText(
        out,
        text,
        (4, out.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (4, out.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return out


def _make_proto_grid(
    tiles: List[np.ndarray],
    cols: int,
    tile_size: int = 160,
    border: int = 2,
) -> np.ndarray:
    if not tiles:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    n = len(tiles)
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    th, tw = tile_size, tile_size
    grid = np.full((rows * th, cols * tw, 3), 40, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        if tile.shape[:2] != (th, tw):
            tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
        y0, x0 = r * th + border, c * tw + border
        h_inner, w_inner = th - 2 * border, tw - 2 * border
        if h_inner > 0 and w_inner > 0:
            inner = cv2.resize(tile, (w_inner, h_inner), interpolation=cv2.INTER_AREA)
            grid[y0 : y0 + h_inner, x0 : x0 + w_inner] = inner
    return grid


def _empty_tracks(model: nn.Module, device: torch.device) -> List[TrackInstances]:
    inner = get_model(model)
    return [
        TrackInstances(hidden_dim=inner.hidden_dim, num_classes=inner.num_classes).to(device)
    ]


def _list_sequences(img_root: str, seq_arg: str) -> List[str]:
    if seq_arg.strip().lower() == "all":
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"image root not found: {img_root}")
        return sorted(
            d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))
        )
    return [s.strip() for s in seq_arg.split(",") if s.strip()]


def save_scem_prototype_maps(
    token_debug: Dict[str, Any],
    ori_image: np.ndarray,
    out_dir: str,
    frame_stem: str,
    frame_pad_mask: torch.Tensor,
    pad_h: int,
    pad_w: int,
    levels: Optional[List[int]] = None,
    overlay_alpha: float = 0.38,
    vis_max_size: int = 1200,
    grid_cols: int = 8,
    cmap_mode: str = "coolwarm",
    percentile: float = 99.0,
) -> Dict[str, Any]:
    spec_levels = token_debug.get("spectral_evidence_multilevel", [])
    if not spec_levels:
        raise RuntimeError("scem_token_debug missing spectral_evidence_multilevel.")

    rgb = _ori_to_vis_rgb(ori_image)
    eff_h, eff_w = rgb.shape[:2]
    os.makedirs(out_dir, exist_ok=True)

    rgb_path = os.path.join(out_dir, f"{frame_stem}__rgb.jpg")
    rgb_out = _resize_max_side(rgb, vis_max_size)
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))

    level_indices = levels if levels is not None else list(range(len(spec_levels)))
    meta: Dict[str, Any] = {
        "rgb": rgb_path,
        "levels": {},
        "num_prototypes": None,
    }

    for lvl in level_indices:
        if lvl < 0 or lvl >= len(spec_levels):
            continue
        spec_evi = spec_levels[lvl]
        if not torch.is_tensor(spec_evi) or spec_evi.dim() != 4:
            print(f"[Warn] skip level {lvl}: bad spectral_evidence shape")
            continue

        k_num = int(spec_evi.shape[1])
        meta["num_prototypes"] = k_num
        h_map, w_map = int(spec_evi.shape[-2]), int(spec_evi.shape[-1])
        pad_mask_lvl = _downsample_pad_mask(frame_pad_mask, (h_map, w_map))

        lvl_dir = os.path.join(out_dir, f"{frame_stem}__L{lvl}")
        os.makedirs(lvl_dir, exist_ok=True)

        pad_eff = cv2.resize(
            pad_mask_lvl.astype(np.uint8),
            (eff_w, eff_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        arrs_eff: List[np.ndarray] = []
        valids: List[np.ndarray] = []
        for k in range(k_num):
            arr = spec_evi[0, k].detach().cpu().numpy()
            arr_eff = _align_map_to_effective(arr, eff_h, eff_w, pad_h, pad_w)
            arrs_eff.append(arr_eff)
            valids.append(_valid_mask_from_pad(pad_eff, arr_eff.shape))

        tiles: List[np.ndarray] = []
        lvl_files: Dict[str, List[str]] = {"heatmap": [], "overlay": []}
        proto_ranges: List[List[float]] = []

        for k, arr_eff in enumerate(arrs_eff):
            valid = valids[k]
            vmin, vmax = _prototype_value_range(arr_eff, valid, cmap_mode, percentile)
            proto_ranges.append([vmin, vmax])
            heat_bgr = _map_to_color_bgr(arr_eff, valid, vmin, vmax, cmap_mode)
            overlay_rgb = _overlay_on_rgb(rgb, heat_bgr, valid, overlay_alpha)

            heat_path = os.path.join(lvl_dir, f"proto{k:03d}__heatmap.jpg")
            ov_path = os.path.join(lvl_dir, f"proto{k:03d}__overlay.jpg")
            heat_out = _resize_max_side(_label_tile(heat_bgr, str(k)), vis_max_size)
            ov_out = _resize_max_side(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), vis_max_size)
            cv2.imwrite(heat_path, heat_out)
            cv2.imwrite(ov_path, ov_out)

            tiles.append(heat_out)
            lvl_files["heatmap"].append(heat_path)
            lvl_files["overlay"].append(ov_path)

        cols = min(grid_cols, max(1, k_num))
        grid = _make_proto_grid(tiles, cols=cols)
        grid_path = os.path.join(out_dir, f"{frame_stem}__L{lvl}__proto_grid.jpg")
        cv2.imwrite(grid_path, grid)

        meta["levels"][str(lvl)] = {
            "spatial_hw": [h_map, w_map],
            "num_prototypes": k_num,
            "proto_grid": grid_path,
            "norm_mode": "per_prototype",
            "color_ranges": proto_ranges,
            "cmap": cmap_mode,
            "percentile": percentile,
            "files": lvl_files,
        }
        print(
            f"[Info] level {lvl}: K={k_num}, cmap={cmap_mode}, "
            f"per-proto norm (p{percentile:g}), grid={grid_path}"
        )

    return meta


def run_seq_proto_vis(
    model: nn.Module,
    seq: str,
    seq_dir: str,
    out_dir: str,
    start_frame: int,
    end_frame: int,
    npy2rgb: bool = False,
    dataset_type: Optional[str] = None,
    levels: Optional[List[int]] = None,
    overlay_alpha: float = 0.38,
    vis_max_size: int = 1200,
    grid_cols: int = 8,
    cmap_mode: str = "coolwarm",
    percentile: float = 99.0,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb, dataset_type=dataset_type)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {seq_dir}")

    end_frame = min(end_frame, len(dataset))
    if start_frame < 1 or start_frame > end_frame:
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}]")

    inner_model = get_model(model)
    use_prior_map = (
        hasattr(inner_model, "scem_module")
        and inner_model.scem_module is not None
        and inner_model.scem_module.prior_mode is not None
    )
    if inner_model.scem_module is None:
        raise RuntimeError("Model has no scem_module; enable SCEM in config.")

    frame_list = list(range(start_frame, end_frame + 1))
    print(
        f"[Info] seq={seq}: non-sequential SCEM prototype vis, "
        f"frames {frame_list} (dataset={len(dataset)})"
    )

    results: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for frame_num in frame_list:
            dataset_idx = frame_num - 1
            print(f"[Info] forward frame {frame_num} (index {dataset_idx})")

            image, ori_image = dataset[dataset_idx][0]
            frame = tensor_list_to_nested_tensor([image]).to(device)
            tracks = _empty_tracks(model, device)

            forward_kwargs = dict(frame=frame, tracks=tracks, debug=True)
            if use_prior_map:
                gmc = np.eye(2, 3, dtype=np.float32)
                forward_kwargs["gmc"] = torch.tensor(gmc, dtype=torch.float32).unsqueeze(0).to(device)

            res = model(**forward_kwargs)
            token_debug = res.get("scem_token_debug") if isinstance(res, dict) else None
            if not isinstance(token_debug, dict):
                raise RuntimeError(f"No scem_token_debug at frame {frame_num} (debug=True?).")

            pad_h, pad_w = int(frame.tensors.shape[-2]), int(frame.tensors.shape[-1])
            frame_stem = f"{seq}__frame{frame_num:06d}"
            frame_dir = os.path.join(out_dir, frame_stem)
            os.makedirs(frame_dir, exist_ok=True)

            meta = save_scem_prototype_maps(
                token_debug=token_debug,
                ori_image=ori_image,
                out_dir=frame_dir,
                frame_stem=frame_stem,
                frame_pad_mask=frame.masks[0],
                pad_h=pad_h,
                pad_w=pad_w,
                levels=levels,
                overlay_alpha=overlay_alpha,
                vis_max_size=vis_max_size,
                grid_cols=grid_cols,
                cmap_mode=cmap_mode,
                percentile=percentile,
            )
            meta["frame"] = frame_num
            results.append(meta)
            del res, frame, image

    return results


def _parse_levels(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not raw.strip():
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize per-prototype SCEM spectral_evidence on image (non-sequential)."
    )
    parser.add_argument("--train-config", type=str, default="20260511-2.yaml")
    parser.add_argument("--checkpoint", type=str, default="last.pth")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="hsmot_8ch")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--seq",
        type=str,
        default="data30-10",
        help="序列名、逗号列表，或 all（test 目录下全部子文件夹）。",
    )
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frames", type=int, default=5, help="末帧序号（含），默认前 5 帧。")
    parser.add_argument("--img-format", type=str, default="npy2jpg", choices=["npy2jpg", "npy"])
    parser.add_argument("--npy2rgb", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="要可视化的 SCEM 特征层索引，逗号分隔；默认全部层。",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.38)
    parser.add_argument("--vis-max-size", type=int, default=1200)
    parser.add_argument("--grid-cols", type=int, default=8, help="prototype grid 列数。")
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        choices=["coolwarm", "rdbu", "positive", "magma", "viridis"],
        help="coolwarm/rdbu=正负对称；positive/magma/viridis=仅正响应。",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="每个 prototype 在自身有效像素上，按绝对值（或正值）的分位数定 vmax。",
    )
    args = parser.parse_args()

    if args.start_frame < 1:
        raise ValueError("--start-frame must be >= 1")
    if args.end_frames < args.start_frame:
        raise ValueError("--end-frames must be >= --start-frame")

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
    proto_root = os.path.join(config_root, "scem_proto")
    os.makedirs(proto_root, exist_ok=True)

    base_name = args.dataset_name.replace("_8ch", "") if "hsmot" in args.dataset_name else args.dataset_name
    img_subdir, dataset_type = _resolve_img_format(args.img_format, train_config)
    img_root = _resolve_img_root(args.data_root, base_name, args.split, img_subdir)

    seq_list = _list_sequences(img_root, args.seq)
    if not seq_list:
        raise ValueError("No sequences to process.")
    print(f"[Info] img_root={img_root}")
    print(f"[Info] {len(seq_list)} sequence(s), frames {args.start_frame}-{args.end_frames}")

    levels = _parse_levels(args.levels)

    train_config["MEMOTR_VERSION"] = "20260511_figure"
    model = build_model(config=train_config)
    model.to(torch.device(args.device))
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    print(f"[Info] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    _init_track_instances_static(train_config, model)

    all_summary: Dict[str, Any] = {
        "seq_list": seq_list,
        "start_frame": args.start_frame,
        "end_frame": args.end_frames,
        "inference_mode": "per_frame_independent",
        "norm_mode": "per_prototype",
        "response_tensor": "spectral_evidence_multilevel[B,K,H,W]",
        "config": train_cfg_path,
        "checkpoint": checkpoint_path,
        "cmap": args.cmap,
        "percentile": args.percentile,
        "levels": levels,
        "per_seq": [],
    }

    for seq in seq_list:
        seq_dir = os.path.join(img_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[Warn] Skip missing seq dir: {seq_dir}")
            continue

        out_root = os.path.join(proto_root, seq)
        os.makedirs(out_root, exist_ok=True)
        print(f"\n[Info] ===== seq={seq} -> {out_root} =====")

        results = run_seq_proto_vis(
            model=model,
            seq=seq,
            seq_dir=seq_dir,
            out_dir=out_root,
            start_frame=args.start_frame,
            end_frame=args.end_frames,
            npy2rgb=args.npy2rgb,
            dataset_type=dataset_type,
            levels=levels,
            overlay_alpha=args.overlay_alpha,
            vis_max_size=args.vis_max_size,
            grid_cols=args.grid_cols,
            cmap_mode=args.cmap,
            percentile=args.percentile,
        )

        seq_summary = {
            "seq": seq,
            "frames": results,
        }
        seq_meta_path = os.path.join(out_root, "scem_proto_meta.json")
        with open(seq_meta_path, "w", encoding="utf-8") as f:
            json.dump(seq_summary, f, indent=2, ensure_ascii=False)
        all_summary["per_seq"].append(seq_summary)
        print(f"[Info] seq={seq} done, meta={seq_meta_path}")

    summary_path = os.path.join(proto_root, "scem_proto_all_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] SCEM prototype maps under: {proto_root}")
    print(f"[Done] All-summary: {summary_path}")


if __name__ == "__main__":
    main()
