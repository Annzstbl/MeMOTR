"""
基于 GT 大小目标分组，调制 SCEM prototype 空间响应后可视化。

规则（K=32 时默认前 16 / 后 16）：
  - prototype 0..15-1：大目标区域保持；小目标区域按 --wrong-region-mode 处理
  - prototype 16..K-1：小目标区域保持；大目标区域按 --wrong-region-mode 处理
  - 系数≠1 的像素（错尺度抑制/反向、对应尺度增强等）：仅对正值 ×w，负值与零不变
  - wrong-region-mode：invert / attenuate / invert_attenuate 决定错尺度 w 的符号与大小
  - 非 GT 覆盖区域系数为 1

大小目标：按当前帧 GT 框面积中位数划分（≥ 中位数为大，< 为小）。
抑制范围：GT 热力图默认 k=6.5，evidence 上模糊 σ=1.2 再膨胀 4px（`--mask-k` / `--mask-blur` / `--mask-dilate`）。

输出（每帧、每层 lvl，`--levels 0,1`）：
  - `<stem>__L{lvl}_mod/`、`<stem>__L{lvl}_raw/`（raw 可用 `--no-save-raw` 关闭）
  - `<stem>__L{lvl}__gt_masks/`：该层 evidence 分辨率上的大/小 mask
  - 每层独立按 (H,W) 生成 GT mask 并调制 spectral_evidence

示例：
python debug/20260526/scem_proto_vis_gt_size.py --seq data --end-frames 5 --levels 0,1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.utils import load_checkpoint, get_model
from utils.utils import load_yaml_with_inheritance
from utils.nested_tensor import tensor_list_to_nested_tensor
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset
from hsmot.datasets.pipelines.channel import HeatmapFromRotateGt

_spv_path = os.path.join(CURRENT_DIR, "scem_proto_vis.py")
_spv_spec = importlib.util.spec_from_file_location("scem_proto_vis", _spv_path)
spv = importlib.util.module_from_spec(_spv_spec)
assert _spv_spec.loader is not None
_spv_spec.loader.exec_module(spv)

_test4_path = os.path.join(CURRENT_DIR, "test4.py")
_t4_spec = importlib.util.spec_from_file_location("debug_test4", _test4_path)
_test4 = importlib.util.module_from_spec(_t4_spec)
assert _t4_spec.loader is not None
_t4_spec.loader.exec_module(_test4)

_resolve_train_config_path = _test4._resolve_train_config_path
_resolve_checkpoint_path = _test4._resolve_checkpoint_path
_resolve_img_format = _test4._resolve_img_format
_resolve_img_root = _test4._resolve_img_root
_resolve_config_root = _test4._resolve_config_root
_init_track_instances_static = _test4._init_track_instances_static


def _resolve_label_path(data_root: str, split: str, seq: str) -> str:
    candidates = [
        os.path.join(data_root, "hsmot", split, "mot", f"{seq}.txt"),
        os.path.join(data_root, "hsmot", "mot", f"{seq}.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"GT label not found for {seq}, tried: {candidates}")


def load_gt_by_frame(label_path: str) -> Dict[int, np.ndarray]:
    """frame_idx(0-based) -> (N, 8) xyxyxyxy float32。"""
    label_full: Dict[int, List[np.ndarray]] = defaultdict(list)
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            frame_id = int(parts[0]) - 1
            coords = list(map(float, parts[2:10]))
            label_full[frame_id].append(np.array(coords, dtype=np.float32))
    return {k: np.stack(v, axis=0) for k, v in label_full.items() if v}


def _polygon_area_xyxyxyxy(box: np.ndarray) -> float:
    pts = box.reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def split_boxes_by_area(
    boxes_xyxyxyxy: np.ndarray,
    area_percentile: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """返回 large_boxes, small_boxes, area_threshold。"""
    if boxes_xyxyxyxy.size == 0:
        return boxes_xyxyxyxy[:0], boxes_xyxyxyxy[:0], 0.0

    areas = np.array([_polygon_area_xyxyxyxy(b) for b in boxes_xyxyxyxy], dtype=np.float64)
    thresh = float(np.percentile(areas, area_percentile))
    large_mask = areas >= thresh
    small_mask = ~large_mask
    if not large_mask.any():
        large_mask[np.argmax(areas)] = True
        small_mask = ~large_mask
    if not small_mask.any():
        small_mask[np.argmin(areas)] = True
        large_mask = ~small_mask
    return boxes_xyxyxyxy[large_mask], boxes_xyxyxyxy[small_mask], thresh


def _dilate_soft_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """在 evidence 分辨率上膨胀软 mask，扩大抑制/保留区域的有效范围。"""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.float32), kernel)


def _blur_soft_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma)


def _normalize_mask_max_one(mask: np.ndarray) -> np.ndarray:
    """非负 mask 按峰值缩放到 max=1；空图保持全 0。"""
    out = np.clip(mask.astype(np.float32), 0.0, None)
    peak = float(out.max())
    if peak > 0.0:
        out /= peak
    return out


def _heatmap_from_boxes(
    boxes_xyxyxyxy: np.ndarray,
    img_hw: Tuple[int, int],
    device: torch.device,
    heatmap_k: float = 6.5,
) -> np.ndarray:
    h, w = img_hw
    if boxes_xyxyxyxy.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    xy = torch.tensor(boxes_xyxyxyxy, dtype=torch.float32, device=device)
    hm = HeatmapFromRotateGt.heatmap_from_rotate_gt_xyxyxyxy(
        xy,
        img_hw,
        version="le135",
        mode="fixed_peak",
        peak=1.0,
        reduce="sum",
        k=heatmap_k,
    )
    return hm.detach().cpu().numpy().astype(np.float32)


def build_large_small_masks(
    pad_h: int,
    pad_w: int,
    large_boxes: np.ndarray,
    small_boxes: np.ndarray,
    target_hw: Tuple[int, int],
    device: torch.device,
    heatmap_k: float = 6.5,
    mask_dilate: int = 4,
    mask_blur: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """在 target_hw 上的软 mask [0,1]。"""
    m_large = _heatmap_from_boxes(large_boxes, (pad_h, pad_w), device, heatmap_k=heatmap_k)
    m_small = _heatmap_from_boxes(small_boxes, (pad_h, pad_w), device, heatmap_k=heatmap_k)
    th, tw = target_hw
    m_large = cv2.resize(m_large, (tw, th), interpolation=cv2.INTER_LINEAR)
    m_small = cv2.resize(m_small, (tw, th), interpolation=cv2.INTER_LINEAR)
    m_large = _blur_soft_mask(m_large, mask_blur)
    m_small = _blur_soft_mask(m_small, mask_blur)
    m_large = _dilate_soft_mask(m_large, mask_dilate)
    m_small = _dilate_soft_mask(m_small, mask_dilate)
    m_large = _normalize_mask_max_one(m_large)
    m_small = _normalize_mask_max_one(m_small)
    return m_large, m_small


def prototype_size_modulation_map(
    m_large: np.ndarray,
    m_small: np.ndarray,
    k: int,
    k_split: int,
    wrong_region_mode: str = "invert",
    small_coef: float = 0.05,
    large_coef: float = 0.05,
) -> np.ndarray:
    """
    返回 [H,W] 逐像素调制系数（与 evidence 相乘）。

    wrong_region_mode:
      - invert / attenuate / invert_attenuate：错尺度 w 见下；evidence 侧统一为 w≠1 时仅正值 ×w
      - 对应尺度可用 w>1 增强（如 m_small×3）；增强同样只作用于正值
    """
    mode = wrong_region_mode.strip().lower()
    bg = (1.0 - np.clip(m_large + m_small, 0.0, 1.0)).astype(np.float32)

    if mode == "invert":
        wrong_small, wrong_large = -1.0, -1.0
    elif mode == "attenuate":
        wrong_small, wrong_large = small_coef, large_coef
    elif mode == "invert_attenuate":
        wrong_small, wrong_large = -abs(small_coef), -abs(large_coef)
    else:
        raise ValueError(
            f"wrong_region_mode must be invert|attenuate|invert_attenuate, got {wrong_region_mode!r}"
        )


    #临时修改
    if k > k_split:
        return (m_large * 2.0 + m_small * wrong_small + bg * 1.0).astype(np.float32)
    return (m_small * 3.0 + m_large * wrong_large + bg * 1.0).astype(np.float32)
    # if k < k_split:
    #     return (m_large * 2.0 + m_small * wrong_small + bg * 1.0).astype(np.float32)
    # return (m_small * 1.0 + m_large * wrong_large + bg * 1.0).astype(np.float32)


prototype_size_weights = prototype_size_modulation_map


def _apply_evidence_size_modulation(
    evi: torch.Tensor,
    w_map: torch.Tensor,
    _wrong_region_mode: str,
) -> torch.Tensor:
    """w≠1 的像素仅调制正值（增强/削弱/反向）；w=1 与负值、零保持原 evidence。"""
    non_unit = torch.abs(w_map - 1.0) > 1e-6
    pos = evi > 0
    return torch.where(
        non_unit & pos,
        evi * w_map,
        torch.where(non_unit, evi, evi * w_map),
    )


def modulate_spectral_evidence(
    spectral_evi: torch.Tensor,
    m_large: np.ndarray,
    m_small: np.ndarray,
    k_split: int = 16,
    wrong_region_mode: str = "invert",
    small_coef: float = 0.05,
    large_coef: float = 0.05,
) -> torch.Tensor:
    """spectral_evi: [K,H,W] -> modulated [K,H,W]。"""
    k_num, _, _ = spectral_evi.shape
    out = spectral_evi.clone()
    for k in range(k_num):
        w_map = prototype_size_modulation_map(
            m_large, m_small, k, k_split, wrong_region_mode, small_coef, large_coef
        )
        w_t = torch.from_numpy(w_map).to(device=spectral_evi.device, dtype=spectral_evi.dtype)
        out[k] = _apply_evidence_size_modulation(out[k], w_t, wrong_region_mode)
    return out


def _prepare_proto_effective_maps(
    spectral_evi: torch.Tensor,
    ori_image: np.ndarray,
    frame_pad_mask: torch.Tensor,
    pad_h: int,
    pad_w: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], int, int]:
    """返回 rgb, pad_eff, 各 prototype 有效区对齐图与 valid mask。"""
    k_num = int(spectral_evi.shape[0])
    h_map, w_map = int(spectral_evi.shape[-2]), int(spectral_evi.shape[-1])
    rgb = spv._ori_to_vis_rgb(ori_image)
    eff_h, eff_w = rgb.shape[:2]
    pad_eff_bool = spv._downsample_pad_mask(frame_pad_mask, (h_map, w_map))
    pad_eff = cv2.resize(
        pad_eff_bool.astype(np.uint8), (eff_w, eff_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    arrs: List[np.ndarray] = []
    valids: List[np.ndarray] = []
    for k in range(k_num):
        arr_eff = spv._align_map_to_effective(
            spectral_evi[k].detach().cpu().numpy(), eff_h, eff_w, pad_h, pad_w
        )
        arrs.append(arr_eff)
        valids.append(spv._valid_mask_from_pad(pad_eff, arr_eff.shape))
    return rgb, pad_eff, arrs, valids, h_map, w_map


def _proto_ranges_from_arrs(
    arrs: List[np.ndarray],
    valids: List[np.ndarray],
    cmap_mode: str,
    percentile: float,
) -> List[Tuple[float, float]]:
    return [
        spv._prototype_value_range(arr, valid, cmap_mode, percentile)
        for arr, valid in zip(arrs, valids)
    ]


def _mpl_cmap_name(cmap_mode: str) -> str:
    key = cmap_mode.strip().lower()
    if key in ("coolwarm", "bwr"):
        return "coolwarm"
    if key == "rdbu":
        return "RdBu_r"
    if key in ("positive", "magma", "activation"):
        return "magma"
    if key == "viridis":
        return "viridis"
    raise ValueError(f"Unknown cmap: {cmap_mode!r}")


def _save_proto_heatmap_colorbar_panel(
    arr: np.ndarray,
    valid: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_mode: str,
    out_path: str,
    title: str,
    vis_max_size: int,
) -> Dict[str, float]:
    """保存「热力图 + colorbar + 有效区数值直方图」，便于查看值域分布。"""
    vals = arr[valid].reshape(-1).astype(np.float64)
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float64)

    arr_plot = np.ma.masked_where(~valid, arr.astype(np.float32))
    h, w = arr.shape
    fig_w = min(14.0, max(8.0, vis_max_size / 100.0))
    fig_h = max(4.0, fig_w * (h / max(w, 1)) * 0.42)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": [2.2, 1.0]})

    ax0 = axes[0]
    cmap = _mpl_cmap_name(cmap_mode)
    im = ax0.imshow(arr_plot, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label("spectral evidence", fontsize=9)
    ax0.set_title(title, fontsize=10)
    ax0.axis("off")

    ax1 = axes[1]
    ax1.hist(vals, bins=48, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.axvline(vmin, color="#C44E52", linestyle=":", linewidth=1.2, label=f"vmin={vmin:.3g}")
    ax1.axvline(vmax, color="#55A868", linestyle=":", linewidth=1.2, label=f"vmax={vmax:.3g}")
    p50 = float(np.percentile(vals, 50))
    p99 = float(np.percentile(np.abs(vals), 99))
    ax1.axvline(p50, color="#8172B2", linestyle="-.", linewidth=1.0, alpha=0.9, label=f"p50={p50:.3g}")
    ax1.set_xlabel("value")
    ax1.set_ylabel("count")
    ax1.set_title(f"|v| p99={p99:.3g}", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "p50": p50,
        "abs_p99": p99,
        "vmin_vis": float(vmin),
        "vmax_vis": float(vmax),
    }


def _save_proto_level_maps(
    spectral_evi: torch.Tensor,
    ori_image: np.ndarray,
    out_dir: str,
    level_dir_name: str,
    frame_pad_mask: torch.Tensor,
    pad_h: int,
    pad_w: int,
    overlay_alpha: float,
    vis_max_size: int,
    grid_cols: int,
    cmap_mode: str,
    percentile: float,
    weight_maps: Optional[List[np.ndarray]] = None,
    proto_value_ranges: Optional[List[Tuple[float, float]]] = None,
    compare_raw_arrs: Optional[List[np.ndarray]] = None,
    save_diff: bool = False,
    norm_mode: str = "per_prototype",
    save_colorbar: bool = True,
) -> Dict[str, Any]:
    """将 [K,H,W] evidence 保存为逐 prototype 热力图/叠加图及总 grid。"""
    rgb, pad_eff, arrs_eff, valids, h_map, w_map = _prepare_proto_effective_maps(
        spectral_evi, ori_image, frame_pad_mask, pad_h, pad_w
    )
    k_num = len(arrs_eff)
    lvl_dir = os.path.join(out_dir, level_dir_name)
    os.makedirs(lvl_dir, exist_ok=True)

    if proto_value_ranges is None:
        proto_value_ranges = _proto_ranges_from_arrs(arrs_eff, valids, cmap_mode, percentile)

    tiles: List[np.ndarray] = []
    lvl_files: Dict[str, List[str]] = {
        "heatmap": [],
        "overlay": [],
        "weight": [],
        "delta": [],
        "heatmap_cbar": [],
    }
    proto_ranges: List[List[float]] = []
    value_stats: List[Dict[str, float]] = []

    for k in range(k_num):
        arr_eff = arrs_eff[k]
        valid = valids[k]
        vmin, vmax = proto_value_ranges[k]
        proto_ranges.append([vmin, vmax])
        heat_bgr = spv._map_to_color_bgr(arr_eff, valid, vmin, vmax, cmap_mode)
        overlay_rgb = spv._overlay_on_rgb(rgb, heat_bgr, valid, overlay_alpha)

        heat_path = os.path.join(lvl_dir, f"proto{k:03d}__heatmap.jpg")
        ov_path = os.path.join(lvl_dir, f"proto{k:03d}__overlay.jpg")
        heat_out = spv._resize_max_side(spv._label_tile(heat_bgr, str(k)), vis_max_size)
        ov_out = spv._resize_max_side(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), vis_max_size)
        cv2.imwrite(heat_path, heat_out)
        cv2.imwrite(ov_path, ov_out)
        tiles.append(heat_out)
        lvl_files["heatmap"].append(heat_path)
        lvl_files["overlay"].append(ov_path)

        if save_colorbar:
            cbar_path = os.path.join(lvl_dir, f"proto{k:03d}__heatmap_cbar.png")
            cbar_title = f"proto {k}  [{vmin:.3g}, {vmax:.3g}]  ({norm_mode})"
            stats = _save_proto_heatmap_colorbar_panel(
                arr_eff, valid, vmin, vmax, cmap_mode, cbar_path, cbar_title, vis_max_size
            )
            lvl_files["heatmap_cbar"].append(cbar_path)
            value_stats.append(stats)

        if weight_maps is not None and k < len(weight_maps):
            w_arr = weight_maps[k]
            w_norm = np.clip((w_arr + 1.0) * 0.5, 0.0, 1.0)
            w_gray = (w_norm * 255).astype(np.uint8)
            w_lut = spv._get_cmap_lut_bgr("coolwarm")
            w_color = w_lut[w_gray]
            w_path = os.path.join(lvl_dir, f"proto{k:03d}__weight.jpg")
            cv2.imwrite(w_path, spv._resize_max_side(w_color, vis_max_size))
            lvl_files["weight"].append(w_path)

        if save_diff and compare_raw_arrs is not None and k < len(compare_raw_arrs):
            delta = np.abs(compare_raw_arrs[k].astype(np.float32) - arr_eff.astype(np.float32))
            dmax = float(np.percentile(delta[valid], percentile)) if valid.any() else 1.0
            dmax = max(dmax, 1e-6)
            delta_bgr = spv._map_to_color_bgr(delta, valid, 0.0, dmax, "magma")
            d_path = os.path.join(lvl_dir, f"proto{k:03d}__delta.jpg")
            cv2.imwrite(d_path, spv._resize_max_side(spv._label_tile(delta_bgr, "d"), vis_max_size))
            lvl_files["delta"].append(d_path)

    cols = min(grid_cols, max(1, k_num))
    grid_path = os.path.join(lvl_dir, "proto_grid_all.jpg")
    cv2.imwrite(grid_path, spv._make_proto_grid(tiles, cols=cols))

    return {
        "dir": lvl_dir,
        "spatial_hw": [h_map, w_map],
        "num_prototypes": k_num,
        "proto_grid": grid_path,
        "norm_mode": norm_mode,
        "color_ranges": proto_ranges,
        "cmap": cmap_mode,
        "percentile": percentile,
        "files": lvl_files,
        "value_stats": value_stats,
    }


def save_gt_size_proto_maps(
    spectral_evi_raw: torch.Tensor,
    spectral_evi_mod: torch.Tensor,
    ori_image: np.ndarray,
    out_dir: str,
    frame_stem: str,
    frame_pad_mask: torch.Tensor,
    pad_h: int,
    pad_w: int,
    m_large: np.ndarray,
    m_small: np.ndarray,
    level: int,
    overlay_alpha: float,
    vis_max_size: int,
    grid_cols: int,
    cmap_mode: str,
    percentile: float,
    k_split: int,
    save_raw: bool = True,
    small_coef: float = 0.05,
    large_coef: float = 0.05,
    wrong_region_mode: str = "invert",
    norm_mode: str = "shared_raw",
    save_diff: bool = True,
    save_colorbar: bool = True,
    write_rgb: bool = False,
) -> Dict[str, Any]:
    rgb = spv._ori_to_vis_rgb(ori_image)
    eff_h, eff_w = rgb.shape[:2]
    os.makedirs(out_dir, exist_ok=True)

    rgb_path = os.path.join(out_dir, f"{frame_stem}__rgb.jpg")
    if write_rgb:
        rgb_out = spv._resize_max_side(rgb, vis_max_size)
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))

    k_num = int(spectral_evi_mod.shape[0])
    _, _, raw_arrs, raw_valids, _, _ = _prepare_proto_effective_maps(
        spectral_evi_raw, ori_image, frame_pad_mask, pad_h, pad_w
    )
    raw_ranges = _proto_ranges_from_arrs(raw_arrs, raw_valids, cmap_mode, percentile)

    weight_maps: List[np.ndarray] = []
    for k in range(k_num):
        w_map = prototype_size_modulation_map(
            m_large, m_small, k, k_split, wrong_region_mode, small_coef, large_coef
        )
        w_eff = spv._align_map_to_effective(w_map, eff_h, eff_w, pad_h, pad_w)
        weight_maps.append(w_eff)

    meta: Dict[str, Any] = {
        "level": level,
        "rgb": rgb_path if write_rgb else None,
        "spatial_hw": [int(spectral_evi_mod.shape[-2]), int(spectral_evi_mod.shape[-1])],
        "num_prototypes": k_num,
        "wrong_region_mode": wrong_region_mode,
        "norm_mode_mod": norm_mode,
    }

    if save_raw:
        raw_dir_name = f"{frame_stem}__L{level}_raw"
        raw_level = _save_proto_level_maps(
            spectral_evi_raw,
            ori_image,
            out_dir,
            raw_dir_name,
            frame_pad_mask,
            pad_h,
            pad_w,
            overlay_alpha,
            vis_max_size,
            grid_cols,
            cmap_mode,
            percentile,
            weight_maps=None,
            proto_value_ranges=raw_ranges,
            norm_mode="per_prototype",
            save_colorbar=save_colorbar,
        )
        meta["raw"] = raw_level

    mod_ranges = raw_ranges if norm_mode == "shared_raw" else None
    mod_norm_label = "shared_raw" if norm_mode == "shared_raw" else "per_prototype"
    mod_dir_name = f"{frame_stem}__L{level}_mod"
    mod_level = _save_proto_level_maps(
        spectral_evi_mod,
        ori_image,
        out_dir,
        mod_dir_name,
        frame_pad_mask,
        pad_h,
        pad_w,
        overlay_alpha,
        vis_max_size,
        grid_cols,
        cmap_mode,
        percentile,
        weight_maps=weight_maps,
        proto_value_ranges=mod_ranges,
        compare_raw_arrs=raw_arrs if save_diff else None,
        save_diff=save_diff,
        norm_mode=mod_norm_label,
        save_colorbar=save_colorbar,
    )
    meta["modulated"] = mod_level

    m_large_eff = spv._align_map_to_effective(m_large, eff_h, eff_w, pad_h, pad_w)
    m_small_eff = spv._align_map_to_effective(m_small, eff_h, eff_w, pad_h, pad_w)

    mask_dir = os.path.join(out_dir, f"{frame_stem}__L{level}__gt_masks")
    os.makedirs(mask_dir, exist_ok=True)
    for name, arr in (("large", m_large_eff), ("small", m_small_eff)):
        gray = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        path = os.path.join(mask_dir, f"{name}_mask.jpg")
        cv2.imwrite(path, spv._resize_max_side(color, vis_max_size))

    _, pad_eff, mod_arrs, mod_valids, _, _ = _prepare_proto_effective_maps(
        spectral_evi_mod, ori_image, frame_pad_mask, pad_h, pad_w
    )
    tiles_large: List[np.ndarray] = []
    tiles_small: List[np.ndarray] = []
    mod_lvl_dir = mod_level["dir"]
    grid_ranges = mod_ranges if mod_ranges is not None else _proto_ranges_from_arrs(
        mod_arrs, mod_valids, cmap_mode, percentile
    )

    for k in range(k_num):
        arr_eff = mod_arrs[k]
        valid = mod_valids[k]
        vmin, vmax = grid_ranges[k]
        heat = spv._map_to_color_bgr(arr_eff, valid, vmin, vmax, cmap_mode)
        tile = spv._resize_max_side(spv._label_tile(heat, str(k)), vis_max_size)
        if k < k_split:
            tiles_large.append(tile)
        else:
            tiles_small.append(tile)

    grid_large_path = os.path.join(mod_lvl_dir, "proto_grid_large_group.jpg")
    grid_small_path = os.path.join(mod_lvl_dir, "proto_grid_small_group.jpg")
    cv2.imwrite(grid_large_path, spv._make_proto_grid(tiles_large, cols=min(grid_cols, k_split)))
    cv2.imwrite(
        grid_small_path,
        spv._make_proto_grid(tiles_small, cols=min(grid_cols, max(1, k_num - k_split))),
    )

    meta["gt_masks_dir"] = mask_dir
    meta["grid_large_group"] = grid_large_path
    meta["grid_small_group"] = grid_small_path
    meta["k_split"] = k_split
    n_cbar = len(mod_level["files"].get("heatmap_cbar", []))
    print(
        f"[Info] L{level} modulated: {mod_lvl_dir} "
        f"({len(mod_level['files']['heatmap'])} heatmaps, {n_cbar} colorbar panels)"
    )
    return meta


def _parse_levels(raw: Optional[str]) -> List[int]:
    if raw is None or not raw.strip():
        return [0]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_seq_gt_size_vis(
    model: nn.Module,
    seq: str,
    seq_dir: str,
    label_path: str,
    out_dir: str,
    start_frame: int,
    end_frame: int,
    npy2rgb: bool = False,
    dataset_type: Optional[str] = None,
    k_split: int = 16,
    small_coef: float = 0.05,
    large_coef: float = 0.05,
    area_percentile: float = 50.0,
    overlay_alpha: float = 0.38,
    vis_max_size: int = 1200,
    grid_cols: int = 8,
    cmap_mode: str = "coolwarm",
    percentile: float = 99.0,
    save_raw: bool = True,
    heatmap_k: float = 6.5,
    mask_dilate: int = 4,
    mask_blur: float = 1.2,
    norm_mode: str = "shared_raw",
    save_diff: bool = True,
    save_colorbar: bool = True,
    wrong_region_mode: str = "invert",
    levels: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb, dataset_type=dataset_type)
    gt_by_frame = load_gt_by_frame(label_path)

    end_frame = min(end_frame, len(dataset))
    inner = get_model(model)
    if inner.scem_module is None:
        raise RuntimeError("Model has no scem_module.")

    use_prior_map = (
        inner.scem_module is not None
        and inner.scem_module.prior_mode is not None
    )

    results: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for frame_num in range(start_frame, end_frame + 1):
            idx = frame_num - 1
            print(f"[Info] frame {frame_num}: forward + GT size modulation")

            image, ori_image = dataset[idx][0]
            frame = tensor_list_to_nested_tensor([image]).to(device)
            tracks = spv._empty_tracks(model, device)

            fwd = dict(frame=frame, tracks=tracks, debug=True)
            if use_prior_map:
                gmc = np.eye(2, 3, dtype=np.float32)
                fwd["gmc"] = torch.tensor(gmc, dtype=torch.float32).unsqueeze(0).to(device)
            res = model(**fwd)

            token_debug = res.get("scem_token_debug")
            if not isinstance(token_debug, dict):
                raise RuntimeError("Missing scem_token_debug; need debug=True forward.")

            spec_levels = token_debug.get("spectral_evidence_multilevel", [])
            if not spec_levels:
                raise RuntimeError("No spectral_evidence_multilevel in debug.")

            pad_h, pad_w = int(frame.tensors.shape[-2]), int(frame.tensors.shape[-1])
            boxes = gt_by_frame.get(idx, np.zeros((0, 8), dtype=np.float32))
            large_boxes, small_boxes, area_thr = split_boxes_by_area(boxes, area_percentile)

            level_indices = levels if levels is not None else [0]
            frame_stem = f"{seq}__frame{frame_num:06d}"
            frame_dir = os.path.join(out_dir, frame_stem)
            os.makedirs(frame_dir, exist_ok=True)

            frame_meta: Dict[str, Any] = {
                "frame": frame_num,
                "n_gt": int(len(boxes)),
                "n_large": int(len(large_boxes)),
                "n_small": int(len(small_boxes)),
                "area_threshold": area_thr,
                "wrong_region_mode": wrong_region_mode,
                "small_coef": small_coef,
                "large_coef": large_coef,
                "levels": {},
            }
            wrote_rgb = False

            for lvl in level_indices:
                if lvl < 0 or lvl >= len(spec_levels):
                    print(f"[Warn] frame {frame_num}: skip level {lvl} (num_levels={len(spec_levels)})")
                    continue
                spec_lvl = spec_levels[lvl]
                if not torch.is_tensor(spec_lvl) or spec_lvl.dim() != 4:
                    print(f"[Warn] frame {frame_num}: skip level {lvl}: bad shape {type(spec_lvl)}")
                    continue

                spec_evi = spec_lvl[0].detach()  # [K,H,W]
                h_se, w_se = int(spec_evi.shape[-2]), int(spec_evi.shape[-1])
                m_large, m_small = build_large_small_masks(
                    pad_h,
                    pad_w,
                    large_boxes,
                    small_boxes,
                    (h_se, w_se),
                    device,
                    heatmap_k=heatmap_k,
                    mask_dilate=mask_dilate,
                    mask_blur=mask_blur,
                )

                k_num = int(spec_evi.shape[0])
                k_split_eff = min(k_split, k_num)
                spec_mod = modulate_spectral_evidence(
                    spec_evi,
                    m_large,
                    m_small,
                    k_split_eff,
                    wrong_region_mode,
                    small_coef,
                    large_coef,
                )

                lvl_meta = save_gt_size_proto_maps(
                    spectral_evi_raw=spec_evi,
                    spectral_evi_mod=spec_mod,
                    ori_image=ori_image,
                    out_dir=frame_dir,
                    frame_stem=frame_stem,
                    frame_pad_mask=frame.masks[0],
                    pad_h=pad_h,
                    pad_w=pad_w,
                    m_large=m_large,
                    m_small=m_small,
                    level=lvl,
                    overlay_alpha=overlay_alpha,
                    vis_max_size=vis_max_size,
                    grid_cols=grid_cols,
                    cmap_mode=cmap_mode,
                    percentile=percentile,
                    k_split=k_split_eff,
                    small_coef=small_coef,
                    large_coef=large_coef,
                    wrong_region_mode=wrong_region_mode,
                    save_raw=save_raw,
                    norm_mode=norm_mode,
                    save_diff=save_diff,
                    save_colorbar=save_colorbar,
                    write_rgb=not wrote_rgb,
                )
                wrote_rgb = wrote_rgb or bool(lvl_meta.get("rgb"))
                frame_meta["levels"][str(lvl)] = lvl_meta
                print(
                    f"[Info] frame {frame_num} L{lvl}: hw=({h_se},{w_se}), "
                    f"gt L/S={len(large_boxes)}/{len(small_boxes)}, area_thr={area_thr:.1f}"
                )

            if not frame_meta["levels"]:
                raise RuntimeError(
                    f"frame {frame_num}: no valid levels in {level_indices} "
                    f"(spectral_evidence_multilevel has {len(spec_levels)} levels)"
                )
            results.append(frame_meta)
            print(
                f"[Info] frame {frame_num}: saved levels {sorted(frame_meta['levels'].keys())}"
            )
            del res, frame, image

    return results


def _list_sequences(img_root: str, seq_arg: str) -> List[str]:
    if seq_arg.strip().lower() == "all":
        return sorted(
            d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))
        )
    return [s.strip() for s in seq_arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCEM prototype vis with GT-based large/small size modulation."
    )
    parser.add_argument("--train-config", type=str, default="20260511-2.yaml")
    parser.add_argument("--checkpoint", type=str, default="last.pth")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="hsmot_8ch")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seq", type=str, default="data30-10")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frames", type=int, default=5)
    parser.add_argument("--img-format", type=str, default="npy2jpg")
    parser.add_argument("--npy2rgb", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--levels",
        type=str,
        default="0",
        help="SCEM 层索引，逗号分隔，如 0,1；默认 0。每层独立 GT mask 与调制输出。",
    )
    parser.add_argument("--k-split", type=int, default=16, help="前 k 个大目标组 prototype 数。")
    parser.add_argument(
        "--small-coef",
        type=float,
        default=0.05,
        help="大目标组在小目标上的衰减系数（越小抑制越强，0=完全抑制）。",
    )
    parser.add_argument(
        "--large-coef",
        type=float,
        default=0.05,
        help="小目标组在大目标上的衰减系数（越小抑制越强，0=完全抑制）。",
    )
    parser.add_argument(
        "--wrong-region-mode",
        type=str,
        default="invert",
        choices=["invert", "attenuate", "invert_attenuate"],
        help="错尺度 w：invert/attenuate/invert_attenuate；evidence 上 w≠1 时均仅正值×w。",
    )
    parser.add_argument(
        "--area-percentile",
        type=float,
        default=50.0,
        help="按 GT 框面积分位数划分大/小目标（50=中位数）。",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.38)
    parser.add_argument("--vis-max-size", type=int, default=1200)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--cmap", type=str, default="coolwarm", choices=["coolwarm", "rdbu", "positive", "magma", "viridis"])
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument(
        "--no-save-raw",
        action="store_true",
        help="不保存调制前对照图（默认会写 __L{lvl}_raw/）。",
    )
    parser.add_argument(
        "--mask-k",
        type=float,
        default=6.5,
        help="GT 热力图高斯覆盖倍数（原 5.0，越大区域越向外扩）。",
    )
    parser.add_argument(
        "--mask-blur",
        type=float,
        default=1.2,
        help="evidence 分辨率上对 mask 的高斯模糊 σ（0=不模糊）。",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=4,
        help="下采样到 evidence 后的 mask 膨胀半径（像素，0=不膨胀）。",
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        default="shared_raw",
        choices=["shared_raw", "per_prototype"],
        help="调制图色标：shared_raw=与 raw 共用 p99（推荐）；per_prototype=各自拉满（易看不出抑制）。",
    )
    parser.add_argument(
        "--no-save-diff",
        action="store_true",
        help="不保存 |raw-mod| 差分图 proto*k*__delta.jpg。",
    )
    parser.add_argument(
        "--no-save-colorbar",
        action="store_true",
        help="不保存带 colorbar/直方图的热力图 proto*k*__heatmap_cbar.png。",
    )
    args = parser.parse_args()
    level_indices = _parse_levels(args.levels)

    train_cfg_path = _resolve_train_config_path(args.train_config)
    train_config = load_yaml_with_inheritance(path=train_cfg_path)
    if args.data_root is None:
        args.data_root = train_config.get("DATA_ROOT", "") or ""
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(
            os.path.join(os.path.dirname(train_cfg_path), args.data_root)
        )

    config_root = _resolve_config_root(train_cfg_path, args.output_dir)
    out_root = os.path.join(config_root, "scem_proto_gt_size")
    os.makedirs(out_root, exist_ok=True)

    base_name = args.dataset_name.replace("_8ch", "") if "hsmot" in args.dataset_name else args.dataset_name
    img_subdir, dataset_type = _resolve_img_format(args.img_format, train_config)
    img_root = _resolve_img_root(args.data_root, base_name, args.split, img_subdir)

    train_config["MEMOTR_VERSION"] = "20260511_figure"
    model = build_model(config=train_config)
    model.to(torch.device(args.device))
    ckpt = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    print(f"[Info] Loading checkpoint: {ckpt}")
    load_checkpoint(model=model, path=ckpt)
    _init_track_instances_static(train_config, model)

    inner = get_model(model)
    k_total = int(getattr(inner.scem_module, "spectral_database_num", 32))
    k_split = min(args.k_split, k_total)
    print(
        f"[Info] levels={level_indices}, K={k_total}, k_split={k_split}, "
        f"wrong_mode={args.wrong_region_mode}, "
        f"small_coef={args.small_coef}, large_coef={args.large_coef}, "
        f"area_p={args.area_percentile}, mask_k={args.mask_k}, "
        f"mask_blur={args.mask_blur}, mask_dilate={args.mask_dilate}"
    )

    all_summary: Dict[str, Any] = {
        "modulation": "gt_size_split",
        "levels": level_indices,
        "k_split": k_split,
        "wrong_region_mode": args.wrong_region_mode,
        "small_coef": args.small_coef,
        "large_coef": args.large_coef,
        "area_percentile": args.area_percentile,
        "mask_k": args.mask_k,
        "mask_blur": args.mask_blur,
        "mask_dilate": args.mask_dilate,
        "norm_mode": args.norm_mode,
        "per_seq": [],
    }

    for seq in _list_sequences(img_root, args.seq):
        seq_dir = os.path.join(img_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[Warn] skip missing {seq_dir}")
            continue
        label_path = _resolve_label_path(args.data_root, args.split, seq)
        seq_out = os.path.join(out_root, seq)
        os.makedirs(seq_out, exist_ok=True)

        results = run_seq_gt_size_vis(
            model=model,
            seq=seq,
            seq_dir=seq_dir,
            label_path=label_path,
            out_dir=seq_out,
            start_frame=args.start_frame,
            end_frame=args.end_frames,
            npy2rgb=args.npy2rgb,
            dataset_type=dataset_type,
            k_split=k_split,
            wrong_region_mode=args.wrong_region_mode,
            small_coef=args.small_coef,
            large_coef=args.large_coef,
            area_percentile=args.area_percentile,
            overlay_alpha=args.overlay_alpha,
            vis_max_size=args.vis_max_size,
            grid_cols=args.grid_cols,
            cmap_mode=args.cmap,
            percentile=args.percentile,
            save_raw=not args.no_save_raw,
            heatmap_k=args.mask_k,
            mask_dilate=args.mask_dilate,
            mask_blur=args.mask_blur,
            norm_mode=args.norm_mode,
            save_diff=not args.no_save_diff,
            save_colorbar=not args.no_save_colorbar,
            levels=level_indices,
        )
        all_summary["per_seq"].append({"seq": seq, "label": label_path, "frames": results})

    summary_path = os.path.join(out_root, "scem_proto_gt_size_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] output: {out_root}")
    print(f"[Done] summary: {summary_path}")


if __name__ == "__main__":
    main()
