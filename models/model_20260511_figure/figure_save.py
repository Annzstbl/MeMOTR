"""模型内部可视化落盘：SCEM / track spectral / decoder cross-attn。"""

import csv
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from structures.track_instances import TrackInstances
from .figure_context import FigureContext

SaveContext = Dict[str, Any]
MapSaverFn = Callable[[Dict[str, Any], SaveContext], None]


def _skip_image_write(path: str, skip_if_exists: bool) -> bool:
    return bool(skip_if_exists and os.path.isfile(path))


def _iter_token_levels(token_debug: Dict[str, Any]):
    evidence_weights_levels = token_debug.get("evidence_weights", [])
    pool_weights_levels = token_debug.get("pool_weights", [])
    pool_logits_levels = token_debug.get("pool_logits", [])
    gate_levels = token_debug.get("gate", [])
    support_map_levels = token_debug.get("support_map", [])

    for src_lvl, pw_lvl in enumerate(pool_weights_levels):
        if not torch.is_tensor(pw_lvl) or pw_lvl.dim() != 4:
            continue
        ew_lvl = evidence_weights_levels[src_lvl] if src_lvl < len(evidence_weights_levels) else None
        pl_lvl = pool_logits_levels[src_lvl] if src_lvl < len(pool_logits_levels) else None
        gate_lvl = gate_levels[src_lvl] if src_lvl < len(gate_levels) else None
        support_lvl = support_map_levels[src_lvl] if src_lvl < len(support_map_levels) else None
        yield src_lvl, ew_lvl, pw_lvl, pl_lvl, gate_lvl, support_lvl


def _parse_evidence_pos_neg(ew_lvl):
    ew_pos = ew_neg = None
    if torch.is_tensor(ew_lvl):
        if ew_lvl.dim() == 5:
            ew_pos = ew_lvl
    elif isinstance(ew_lvl, dict):
        a_pos = ew_lvl.get("a_pos")
        a_neg = ew_lvl.get("a_neg")
        if torch.is_tensor(a_pos) and a_pos.dim() == 5:
            ew_pos = a_pos
        if torch.is_tensor(a_neg) and a_neg.dim() == 5:
            ew_neg = a_neg
    return ew_pos, ew_neg


def _resize_pad_mask(pad_mask: torch.Tensor, size_hw: Tuple[int, int]) -> np.ndarray:
    if pad_mask.dim() != 2:
        raise ValueError(f"pad_mask expects [H,W], got {tuple(pad_mask.shape)}")
    m = pad_mask.float().unsqueeze(0).unsqueeze(0)
    m = F.interpolate(m, size=size_hw, mode="nearest")
    return m.squeeze().detach().cpu().numpy().astype(bool)


def _build_level_pad_masks(
    frame_pad_mask: Optional[torch.Tensor],
    token_debug: Dict[str, Any],
) -> List[Optional[np.ndarray]]:
    pool_weights_levels = token_debug.get("pool_weights", [])
    if frame_pad_mask is None:
        return [None] * len(pool_weights_levels)
    level_masks: List[Optional[np.ndarray]] = []
    for pw_lvl in pool_weights_levels:
        if not torch.is_tensor(pw_lvl) or pw_lvl.dim() != 4:
            level_masks.append(None)
            continue
        h, w = int(pw_lvl.shape[-2]), int(pw_lvl.shape[-1])
        level_masks.append(_resize_pad_mask(frame_pad_mask, (h, w)))
    return level_masks


def _level_pad_mask(ctx: SaveContext, src_lvl: int, arr_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    level_pad_masks = ctx.get("level_pad_masks") or []
    if src_lvl >= len(level_pad_masks):
        return None
    pad_mask = level_pad_masks[src_lvl]
    if pad_mask is None or pad_mask.shape != arr_shape:
        return None
    return pad_mask


def _upscale_gray(gray: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return gray
    h, w = gray.shape[:2]
    if max(h, w) >= max_side:
        return gray
    scale = max_side / float(max(h, w))
    return cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_NEAREST)


def _upscale_color(color: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return color
    h, w = color.shape[:2]
    if max(h, w) >= max_side:
        return color
    scale = max_side / float(max(h, w))
    return cv2.resize(color, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_NEAREST)


def _normalize_map_to_gray(arr2d: np.ndarray, pad_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr2d, dtype=np.float32)
    valid = np.ones(arr.shape, dtype=bool)
    if pad_mask is not None:
        pad_mask = pad_mask.astype(bool)
        if pad_mask.shape == arr.shape:
            valid = ~pad_mask
    gray = np.zeros(arr.shape, dtype=np.uint8)
    if valid.any():
        valid_vals = arr[valid]
        vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
        if vmax > vmin:
            norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            gray[valid] = (norm[valid] * 255.0).astype(np.uint8)
    return gray, valid


def _save_grayscale_jpg(
    arr2d: np.ndarray,
    path: str,
    skip_existing: bool,
    pad_mask: Optional[np.ndarray] = None,
    vis_max_size: int = 1200,
) -> bool:
    jpg_path = os.path.splitext(path)[0] + ".jpg"
    if _skip_image_write(jpg_path, skip_existing):
        return False
    gray, _ = _normalize_map_to_gray(arr2d, pad_mask)
    gray = _upscale_gray(gray, vis_max_size)
    os.makedirs(os.path.dirname(jpg_path) or ".", exist_ok=True)
    cv2.imwrite(jpg_path, gray)
    return True


def _save_colormap_jpg(
    arr2d: np.ndarray,
    path: str,
    skip_existing: bool,
    pad_mask: Optional[np.ndarray] = None,
    vis_max_size: int = 1200,
    cmap: int = cv2.COLORMAP_JET,
) -> bool:
    jpg_path = os.path.splitext(path)[0] + "_color.jpg"
    if _skip_image_write(jpg_path, skip_existing):
        return False
    gray, valid = _normalize_map_to_gray(arr2d, pad_mask)
    color = cv2.applyColorMap(gray, cmap)
    color[~valid] = 0
    color = _upscale_color(color, vis_max_size)
    os.makedirs(os.path.dirname(jpg_path) or ".", exist_ok=True)
    cv2.imwrite(jpg_path, color)
    return True


def _save_map_jpg(arr2d: np.ndarray, path: str, ctx: SaveContext, src_lvl: int) -> bool:
    pad_mask = _level_pad_mask(ctx, src_lvl, arr2d.shape)
    skip_existing = ctx["skip_existing"]
    vis_max_size = int(ctx.get("vis_max_size", 1200))
    saved_gray = _save_grayscale_jpg(arr2d, path, skip_existing, pad_mask=pad_mask, vis_max_size=vis_max_size)
    saved_color = _save_colormap_jpg(arr2d, path, skip_existing, pad_mask=pad_mask, vis_max_size=vis_max_size)
    return saved_gray or saved_color


def save_pool_weights_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    for src_lvl, _, pw_lvl, _, _, _ in _iter_token_levels(token_debug):
        for tok_idx in range(pw_lvl.shape[1]):
            pw_map = pw_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                ctx["out_dir"],
                f"{ctx['seq']}__{ctx['frame_tag']}__scem_srcL{src_lvl}_tok{tok_idx}__poolW.jpg",
            )
            if _save_map_jpg(pw_map, path, ctx, src_lvl):
                print(f"  saved pool_weights: {path}")


def save_evidence_weights_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    for src_lvl, ew_lvl, pw_lvl, _, _, _ in _iter_token_levels(token_debug):
        ew_pos, ew_neg = _parse_evidence_pos_neg(ew_lvl)
        if ew_pos is None:
            continue
        num_tokens = min(ew_pos.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            ew_pos_mean = ew_pos[0, tok_idx].mean(dim=0).detach().cpu().numpy()
            path = os.path.join(
                ctx["out_dir"],
                f"{ctx['seq']}__{ctx['frame_tag']}__scem_srcL{src_lvl}_tok{tok_idx}__evidenceW_mean.jpg",
            )
            if _save_map_jpg(ew_pos_mean, path, ctx, src_lvl):
                print(f"  saved evidence_weights: {path}")
            if ew_neg is not None and tok_idx < ew_neg.shape[1]:
                ew_neg_mean = ew_neg[0, tok_idx].mean(dim=0).detach().cpu().numpy()
                neg_path = os.path.join(
                    ctx["out_dir"],
                    f"{ctx['seq']}__{ctx['frame_tag']}__scem_srcL{src_lvl}_tok{tok_idx}__evidenceW_neg_mean.jpg",
                )
                if _save_map_jpg(ew_neg_mean, neg_path, ctx, src_lvl):
                    print(f"  saved evidence_weights_neg: {neg_path}")


def save_pool_logits_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    for src_lvl, _, pw_lvl, pl_lvl, _, _ in _iter_token_levels(token_debug):
        if not torch.is_tensor(pl_lvl) or pl_lvl.dim() != 4:
            continue
        num_tokens = min(pl_lvl.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            pl_map = pl_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                ctx["out_dir"],
                f"{ctx['seq']}__{ctx['frame_tag']}__scem_srcL{src_lvl}_tok{tok_idx}__poolLogits.jpg",
            )
            if _save_map_jpg(pl_map, path, ctx, src_lvl):
                print(f"  saved pool_logits: {path}")


def save_gate_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    for src_lvl, _, pw_lvl, _, gate_lvl, _ in _iter_token_levels(token_debug):
        if not torch.is_tensor(gate_lvl) or gate_lvl.dim() != 4:
            continue
        num_tokens = min(gate_lvl.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            gate_map = gate_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                ctx["out_dir"],
                f"{ctx['seq']}__{ctx['frame_tag']}__scem_srcL{src_lvl}_tok{tok_idx}__gate.jpg",
            )
            if _save_map_jpg(gate_map, path, ctx, src_lvl):
                print(f"  saved gate: {path}")


def save_support_map_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    for src_lvl, _, pw_lvl, _, _, support_lvl in _iter_token_levels(token_debug):
        if not torch.is_tensor(support_lvl) or support_lvl.dim() != 4:
            continue
        num_tokens = min(support_lvl.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            smap = support_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                ctx["out_dir"],
                f"{ctx['seq']}__{ctx['frame_tag']}__scem_srcL{src_lvl}_tok{tok_idx}__support_map.jpg",
            )
            if _save_map_jpg(smap, path, ctx, src_lvl):
                print(f"  saved support_map: {path}")


def save_spectral_evidence_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    for lvl, spec_evi in enumerate(token_debug.get("spectral_evidence_multilevel", [])):
        if not torch.is_tensor(spec_evi) or spec_evi.dim() != 4:
            continue
        for ch in range(spec_evi[0].shape[0]):
            arr2d = spec_evi[0, ch].detach().cpu().numpy()
            path = os.path.join(
                ctx["out_dir"],
                f"{ctx['seq']}__{ctx['frame_tag']}__scem_specEvi_L{lvl}_ch{ch}.jpg",
            )
            if _save_map_jpg(arr2d, path, ctx, lvl):
                print(f"  saved spectral_evidence: {path}")


def save_feature_fusion_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    for lvl, ft in enumerate(token_debug.get("feature_fusion_feat_list", [])):
        if not torch.is_tensor(ft) or ft.dim() != 4:
            continue
        energy = torch.sqrt((ft[0] ** 2).sum(dim=0)).detach().cpu().numpy()
        fe_path = os.path.join(
            ctx["out_dir"], f"{ctx['seq']}__{ctx['frame_tag']}__scem_featfuse_featL{lvl}__featE.jpg"
        )
        if _save_map_jpg(energy, fe_path, ctx, lvl):
            print(f"  saved feature_fusion featE: {fe_path}")

    for lvl, sp in enumerate(token_debug.get("feature_fusion_spec_list", [])):
        if not torch.is_tensor(sp) or sp.dim() != 4:
            continue
        for ch in range(sp[0].shape[0]):
            arr2d = sp[0, ch].detach().cpu().numpy()
            path = os.path.join(
                ctx["out_dir"],
                f"{ctx['seq']}__{ctx['frame_tag']}__scem_featfuse_specL{lvl}_ch{ch}.jpg",
            )
            if _save_map_jpg(arr2d, path, ctx, lvl):
                print(f"  saved feature_fusion spec: {path}")


MAP_SAVER_REGISTRY: Dict[str, MapSaverFn] = {
    "pool_weights": save_pool_weights_maps,
    "evidence_weights": save_evidence_weights_maps,
    "pool_logits": save_pool_logits_maps,
    "gate": save_gate_maps,
    "support_map": save_support_map_maps,
    "spectral_evidence": save_spectral_evidence_maps,
    "feature_fusion": save_feature_fusion_maps,
}

# HSMOT 8 通道光谱 band 中心波长 (nm)
SPECTRAL_BAND_CENTERS_NM = np.array(
    [422.5, 487.5, 550.0, 602.5, 660.0, 725.0, 785.0, 887.2],
    dtype=np.float32,
)


def _wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> Tuple[float, float, float]:
    """将中心波长映射到 RGB（0~1）。780nm 以上按 NIR 延伸为暗红/紫红外色。"""
    wl = float(wavelength_nm)
    if wl < 380.0 or wl > 950.0:
        return 0.0, 0.0, 0.0
    if wl >= 780.0:
        # 887.2nm 等近红外：用暗洋红表示，避免纯黑
        t = np.clip((wl - 780.0) / 110.0, 0.0, 1.0)
        r, g, b = 0.55 + 0.25 * t, 0.06, 0.22 + 0.08 * (1.0 - t)
        return r ** gamma, g ** gamma, b ** gamma

    if 380.0 <= wl < 440.0:
        att = 0.3 + 0.7 * (wl - 380.0) / (440.0 - 380.0)
        r, g, b = (-(wl - 440.0) / (440.0 - 380.0)), 0.0, 1.0
    elif 440.0 <= wl < 490.0:
        att = 1.0
        r, g, b = 0.0, (wl - 440.0) / (490.0 - 440.0), 1.0
    elif 490.0 <= wl < 510.0:
        att = 1.0
        r, g, b = 0.0, 1.0, -(wl - 510.0) / (510.0 - 490.0)
    elif 510.0 <= wl < 580.0:
        att = 1.0
        r, g, b = (wl - 510.0) / (580.0 - 510.0), 1.0, 0.0
    elif 580.0 <= wl < 645.0:
        att = 1.0
        r, g, b = 1.0, -(wl - 645.0) / (645.0 - 580.0), 0.0
    else:
        att = 1.0
        r, g, b = 1.0, 0.0, -(wl - 780.0) / (780.0 - 645.0)

    r = np.clip((r * att) ** gamma, 0.0, 1.0)
    g = np.clip((g * att) ** gamma, 0.0, 1.0)
    b = np.clip((b * att) ** gamma, 0.0, 1.0)
    return float(r), float(g), float(b)


def _normalize_spectral_intensities(values: np.ndarray, min_brightness: float = 0.12) -> np.ndarray:
    """将 8 通道相对强度归一化到 [min_brightness, 1] 用于调制亮度。"""
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return vals
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax > vmin:
        norm = (vals - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(vals)
    return min_brightness + (1.0 - min_brightness) * norm


def _render_spectral_band_strip(
    values: np.ndarray,
    band_centers_nm: np.ndarray = SPECTRAL_BAND_CENTERS_NM,
    cell_w: int = 72,
    cell_h: int = 48,
    gap: int = 3,
    label_h: int = 18,
    min_brightness: float = 0.12,
) -> np.ndarray:
    """
    8 通道光谱条带：每格底色=波段中心波长色，亮度=相对强度。
    返回 BGR uint8 图像。
    """
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    n_bands = min(len(vals), len(band_centers_nm))
    if n_bands == 0:
        return np.zeros((cell_h + label_h, cell_w, 3), dtype=np.uint8)

    brightness = _normalize_spectral_intensities(vals[:n_bands], min_brightness=min_brightness)
    img_w = n_bands * cell_w + (n_bands - 1) * gap
    img_h = cell_h + label_h
    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    for i in range(n_bands):
        x0 = i * (cell_w + gap)
        r, g, b = _wavelength_to_rgb(float(band_centers_nm[i]))
        scale = float(brightness[i])
        color_bgr = (
            int(np.clip(b * scale * 255, 0, 255)),
            int(np.clip(g * scale * 255, 0, 255)),
            int(np.clip(r * scale * 255, 0, 255)),
        )
        cv2.rectangle(canvas, (x0, 0), (x0 + cell_w - 1, cell_h - 1), color_bgr, thickness=-1)
        cv2.rectangle(canvas, (x0, 0), (x0 + cell_w - 1, cell_h - 1), (200, 200, 200), thickness=1)
        label = f"{band_centers_nm[i]:.0f}"
        cv2.putText(
            canvas, label, (x0 + 4, cell_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA,
        )
    return canvas


def _render_spectral_band_grid(
    values_2d: np.ndarray,
    band_centers_nm: np.ndarray = SPECTRAL_BAND_CENTERS_NM,
    cell_w: int = 56,
    cell_h: int = 40,
    gap: int = 2,
    label_h: int = 18,
    min_brightness: float = 0.12,
) -> np.ndarray:
    """[L, 8] 或多行 8 列网格，每格按波长着色 × 强度。"""
    arr = np.asarray(values_2d, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_rows, n_cols = arr.shape
    n_cols = min(n_cols, len(band_centers_nm))

    row_h = cell_h + (label_h if n_rows == 1 else 0)
    img_w = n_cols * cell_w + (n_cols - 1) * gap
    img_h = n_rows * row_h + (n_rows - 1) * gap
    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    for row in range(n_rows):
        brightness = _normalize_spectral_intensities(arr[row, :n_cols], min_brightness=min_brightness)
        y0 = row * (row_h + gap)
        for col in range(n_cols):
            x0 = col * (cell_w + gap)
            r, g, b = _wavelength_to_rgb(float(band_centers_nm[col]))
            scale = float(brightness[col])
            color_bgr = (
                int(np.clip(b * scale * 255, 0, 255)),
                int(np.clip(g * scale * 255, 0, 255)),
                int(np.clip(r * scale * 255, 0, 255)),
            )
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), color_bgr, thickness=-1)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (200, 200, 200), thickness=1)
            if row == 0:
                cv2.putText(
                    canvas, f"{band_centers_nm[col]:.0f}", (x0 + 3, y0 + cell_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 40, 40), 1, cv2.LINE_AA,
                )
    return canvas


def _upscale_bgr_image(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    if max(h, w) >= max_side:
        return img
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _save_spectral_band_strip(
    values: np.ndarray,
    path: str,
    skip_existing: bool,
    vis_max_size: int,
) -> bool:
    """保存 8 通道光谱条带：波长色 × 相对强度（不用 jet colorbar）。"""
    spectral_path = os.path.splitext(path)[0] + "_spectral.jpg"
    if _skip_image_write(spectral_path, skip_existing):
        return False
    img = _render_spectral_band_strip(values)
    img = _upscale_bgr_image(img, vis_max_size)
    os.makedirs(os.path.dirname(spectral_path) or ".", exist_ok=True)
    cv2.imwrite(spectral_path, img)
    return True


def _save_spectral_band_grid(
    values_2d: np.ndarray,
    path: str,
    skip_existing: bool,
    vis_max_size: int,
) -> bool:
    spectral_path = os.path.splitext(path)[0] + "_spectral.jpg"
    if _skip_image_write(spectral_path, skip_existing):
        return False
    img = _render_spectral_band_grid(values_2d)
    img = _upscale_bgr_image(img, vis_max_size)
    os.makedirs(os.path.dirname(spectral_path) or ".", exist_ok=True)
    cv2.imwrite(spectral_path, img)
    return True


def _save_vector_strip(arr1d: np.ndarray, path: str, skip_existing: bool, vis_max_size: int) -> bool:
    jpg_path = os.path.splitext(path)[0] + ".jpg"
    if _skip_image_write(jpg_path, skip_existing):
        return False
    strip = arr1d.reshape(1, -1).astype(np.float32)
    gray, _ = _normalize_map_to_gray(strip)
    gray = _upscale_gray(gray, vis_max_size)
    os.makedirs(os.path.dirname(jpg_path) or ".", exist_ok=True)
    cv2.imwrite(jpg_path, gray)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.splitext(path)[0] + "_color.jpg", color)
    return True


def _save_grid_heatmap(arr2d: np.ndarray, path: str, skip_existing: bool, vis_max_size: int) -> bool:
    jpg_path = os.path.splitext(path)[0] + ".jpg"
    if _skip_image_write(jpg_path, skip_existing):
        return False
    arr = np.asarray(arr2d, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.size == 0:
        return False
    gray, _ = _normalize_map_to_gray(arr)
    gray = _upscale_gray(gray, vis_max_size)
    os.makedirs(os.path.dirname(jpg_path) or ".", exist_ok=True)
    cv2.imwrite(jpg_path, gray)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.splitext(path)[0] + "_color.jpg", color)
    return True


def _sample_pooled_spectral_at_query(
    spectral_weights: List[torch.Tensor],
    ref_pts: torch.Tensor,
    query_idx: int,
    n_levels: int,
) -> Optional[np.ndarray]:
    if not spectral_weights or ref_pts.numel() == 0:
        return None
    ref = ref_pts[query_idx].detach()
    if ref.shape[-1] < 2:
        return None
    cx, cy = ref[0].item(), ref[1].item()
    sampled_levels: List[np.ndarray] = []
    for lvl_weight in spectral_weights[:n_levels]:
        if not torch.is_tensor(lvl_weight) or lvl_weight.dim() != 4:
            continue
        _, _c, _h, _w = lvl_weight.shape
        gx = float(np.clip(cx, 0.0, 1.0)) * 2.0 - 1.0
        gy = float(np.clip(cy, 0.0, 1.0)) * 2.0 - 1.0
        grid = torch.tensor([[[[gx, gy]]]], dtype=lvl_weight.dtype, device=lvl_weight.device)
        sampled = F.grid_sample(
            input=lvl_weight, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampled_levels.append(sampled[0, :, 0, 0].detach().cpu().numpy())
    if not sampled_levels:
        return None
    return np.stack(sampled_levels, axis=0)


def _rasterize_deformable_attn_for_query(
    attention_weights: torch.Tensor,
    sampling_locations: torch.Tensor,
    spatial_shapes: torch.Tensor,
    query_idx: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    attn_q = attention_weights[0, query_idx].numpy()
    loc_q = sampling_locations[0, query_idx].numpy()
    n_levels = spatial_shapes.shape[0]
    level_maps: List[np.ndarray] = []
    for lvl in range(n_levels):
        h, w = int(spatial_shapes[lvl, 0]), int(spatial_shapes[lvl, 1])
        acc = np.zeros((h, w), dtype=np.float32)
        attn_lvl = attn_q[:, lvl, :]
        loc_lvl = loc_q[:, lvl, :, :]
        for head in range(attn_lvl.shape[0]):
            for pt in range(attn_lvl.shape[1]):
                x = float(loc_lvl[head, pt, 0]) * (w - 1)
                y = float(loc_lvl[head, pt, 1]) * (h - 1)
                acc[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))] += float(
                    attn_lvl[head, pt]
                )
        level_maps.append(acc)
    if not level_maps:
        return [], np.zeros((1, 1), dtype=np.float32)
    target_h = max(m.shape[0] for m in level_maps)
    target_w = max(m.shape[1] for m in level_maps)
    fused = np.zeros((target_h, target_w), dtype=np.float32)
    for m in level_maps:
        if m.shape != (target_h, target_w):
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        fused += m
    return level_maps, fused


def _collect_track_spectral_vectors(
    res: Dict[str, Any],
    tracks: TrackInstances,
    track_local_idx: int,
    global_q_idx: int,
    n_feature_levels: int,
) -> Dict[str, Any]:
    collected: Dict[str, Any] = {}
    init_q_spec = res.get("init_q_spec")
    obs_q_spec = res.get("obs_q_spec")
    if torch.is_tensor(init_q_spec) and global_q_idx < init_q_spec.shape[1]:
        collected["init_q_spec"] = init_q_spec[0, global_q_idx].detach().cpu().numpy().tolist()
    if torch.is_tensor(obs_q_spec) and global_q_idx < obs_q_spec.shape[1]:
        collected["obs_q_spec"] = obs_q_spec[0, global_q_idx].detach().cpu().numpy().tolist()
    if TrackInstances.use_q_spec and len(tracks.query_q_spec) > track_local_idx:
        collected["query_q_spec_in"] = tracks.query_q_spec[track_local_idx].detach().cpu().numpy().tolist()

    spectral_weights = res.get("spectral_weights")
    ref_tensor = tracks.ref_pts if len(tracks.ref_pts) > track_local_idx else None
    if isinstance(spectral_weights, list) and ref_tensor is not None:
        pooled = _sample_pooled_spectral_at_query(
            spectral_weights=spectral_weights,
            ref_pts=ref_tensor,
            query_idx=track_local_idx,
            n_levels=n_feature_levels,
        )
        if pooled is not None:
            collected["pooled_spectral"] = pooled.tolist()
    return collected


def _save_pooled_spec_rbox_levels(
    res: Dict[str, Any],
    figure: FigureContext,
    prefix: str,
) -> Dict[str, Any]:
    """保存 rbox 旋转框池化 pooled_spec，每个 FPN level 单独一张 8 通道条带图。"""
    assert figure.global_q_idx is not None
    saved: Dict[str, Any] = {}
    for res_key, tag in (
        ("init_pooled_spec_rbox_levels", "init_pooledSpecRbox"),
        ("obs_pooled_spec_rbox_levels", "obs_pooledSpecRbox"),
    ):
        level_tensors = res.get(res_key)
        if not isinstance(level_tensors, list):
            continue
        for lvl, lvl_tensor in enumerate(level_tensors):
            if not torch.is_tensor(lvl_tensor) or figure.global_q_idx >= lvl_tensor.shape[1]:
                continue
            vec = lvl_tensor[0, figure.global_q_idx].detach().cpu().numpy()
            field = f"{tag}_lvl{lvl}"
            path = f"{prefix}__{field}.jpg"
            if _save_spectral_band_strip(vec, path, figure.skip_existing, figure.vis_max_size):
                print(f"  saved {field}: {os.path.splitext(path)[0]}_spectral.jpg")
            saved[field] = vec.tolist()
    return saved


def _save_track_spectral(res: Dict[str, Any], tracks: TrackInstances, figure: FigureContext, n_feature_levels: int) -> Dict[str, Any]:
    assert figure.track_id is not None
    assert figure.track_local_idx is not None
    assert figure.global_q_idx is not None

    prefix = os.path.join(
        figure.track_out_dir,
        f"{figure.seq}__{figure.frame_tag}__trackId{figure.track_id}_q{figure.track_local_idx}",
    )
    vectors = _collect_track_spectral_vectors(
        res=res,
        tracks=tracks,
        track_local_idx=figure.track_local_idx,
        global_q_idx=figure.global_q_idx,
        n_feature_levels=n_feature_levels,
    )
    for name, saver in (
        ("init_q_spec", _save_vector_strip),
        ("obs_q_spec", _save_vector_strip),
        ("query_q_spec_in", _save_vector_strip),
    ):
        if name not in vectors:
            continue
        path = f"{prefix}__{name}.jpg"
        if saver(np.asarray(vectors[name]), path, figure.skip_existing, figure.vis_max_size):
            print(f"  saved {name}: {path}")
    if "pooled_spectral" in vectors:
        path = f"{prefix}__pooled_spectral_Lx8.jpg"
        arr = np.asarray(vectors["pooled_spectral"], dtype=np.float32)
        if _save_spectral_band_grid(arr, path, figure.skip_existing, figure.vis_max_size):
            print(f"  saved pooled_spectral: {os.path.splitext(path)[0]}_spectral.jpg")
    vectors.update(_save_pooled_spec_rbox_levels(res=res, figure=figure, prefix=prefix))
    return vectors


def _save_cross_attn(
    cross_attn_debug: Dict[int, Dict[str, Any]],
    figure: FigureContext,
) -> None:
    assert figure.track_id is not None
    assert figure.global_q_idx is not None

    ctx: SaveContext = {
        "skip_existing": figure.skip_existing,
        "vis_max_size": figure.vis_max_size,
        "level_pad_masks": [],
    }
    for layer_idx in sorted(cross_attn_debug.keys()):
        payload = cross_attn_debug[layer_idx]
        attn = payload["attention_weights"]
        loc = payload["sampling_locations"]
        shapes = payload["spatial_shapes"]
        if figure.global_q_idx >= attn.shape[1]:
            print(
                f"[Warn] frame {figure.frame_num} layer {layer_idx}: "
                f"query idx {figure.global_q_idx} out of range {attn.shape[1]}"
            )
            continue
        level_maps, fused = _rasterize_deformable_attn_for_query(attn, loc, shapes, figure.global_q_idx)
        prefix = os.path.join(
            figure.track_out_dir,
            f"{figure.seq}__{figure.frame_tag}__trackId{figure.track_id}__decL{layer_idx}_crossAttn",
        )
        if _save_map_jpg(fused, f"{prefix}__fused.jpg", ctx, 0):
            print(f"  saved cross_attn fused: {prefix}__fused.jpg")
        for lvl, lvl_map in enumerate(level_maps):
            if _save_map_jpg(lvl_map, f"{prefix}__lvl{lvl}.jpg", ctx, 0):
                print(f"  saved cross_attn level: {prefix}__lvl{lvl}.jpg")


def save_scem_maps(res: Dict[str, Any], figure: FigureContext) -> None:
    token_debug = res.get("scem_token_debug")
    if not isinstance(token_debug, dict):
        print(f"[Warn] No scem_token_debug at frame {figure.frame_num}")
        return
    ctx: SaveContext = {
        "seq": figure.seq,
        "frame_num": figure.frame_num,
        "frame_tag": figure.frame_tag,
        "out_dir": figure.scem_out_dir,
        "skip_existing": figure.skip_existing,
        "level_pad_masks": _build_level_pad_masks(figure.frame_pad_mask, token_debug),
        "vis_max_size": figure.vis_max_size,
    }
    print(f"[Info] Saving SCEM maps frame {figure.frame_num}: {list(figure.save_maps)}")
    for name in figure.save_maps:
        saver = MAP_SAVER_REGISTRY.get(name)
        if saver is None:
            print(f"[Warn] Unknown save map: {name}")
            continue
        try:
            saver(token_debug, ctx)
        except Exception as e:
            print(f"[Warn] saver '{name}' failed for seq={figure.seq} frame={figure.frame_num}: {e}")


def save_figure_outputs(
    res: Dict[str, Any],
    figure: FigureContext,
    tracks: List[TrackInstances],
    cross_attn_debug: Optional[Dict[int, Dict[str, Any]]] = None,
    n_feature_levels: int = 4,
) -> Optional[Dict[str, Any]]:
    if not figure.enabled:
        return None

    timeline_item: Optional[Dict[str, Any]] = None
    if figure.should_save_scem():
        save_scem_maps(res, figure)

    if figure.should_save_track():
        timeline_item = {
            "frame": figure.frame_num,
            "track_local_idx": figure.track_local_idx,
            "global_q_idx": figure.global_q_idx,
        }
        if figure.save_track_spectral:
            vectors = _save_track_spectral(res, tracks[0], figure, n_feature_levels)
            timeline_item.update(vectors)
        if figure.save_cross_attn and cross_attn_debug:
            _save_cross_attn(cross_attn_debug, figure)

        if figure.spectral_timeline is not None and timeline_item is not None:
            figure.spectral_timeline.append(timeline_item)

    return timeline_item


_SPEC_RBOX_FIELD_RE = re.compile(r"^(init|obs)_pooledSpecRbox_lvl(\d+)$")


def _parse_spec_rbox_field(field: str) -> Optional[Tuple[str, int]]:
    match = _SPEC_RBOX_FIELD_RE.match(field)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _save_spec_rbox_timeline_csv(timeline: List[Dict[str, Any]], figure: FigureContext) -> None:
    """将时序 init/obs pooledSpecRbox 全量导出为 CSV（long + wide 两种格式）。"""
    if not timeline or figure.track_id is None:
        return

    base = os.path.join(
        figure.track_out_dir,
        f"{figure.seq}__trackId{figure.track_id}__timeline_specRbox",
    )
    long_path = f"{base}_long.csv"
    wide_path = f"{base}_wide.csv"
    if figure.skip_existing and os.path.isfile(long_path) and os.path.isfile(wide_path):
        return

    long_rows: List[List[Any]] = []
    wide_records: List[Dict[str, Any]] = []
    all_wide_cols: set = set()

    for item in timeline:
        frame = item.get("frame")
        track_local_idx = item.get("track_local_idx")
        global_q_idx = item.get("global_q_idx")
        wide_row: Dict[str, float] = {}

        for key, val in item.items():
            parsed = _parse_spec_rbox_field(key)
            if parsed is None:
                continue
            spec_type, fpn_level = parsed
            vec = np.asarray(val, dtype=np.float32).reshape(-1)
            n_bands = min(len(vec), len(SPECTRAL_BAND_CENTERS_NM))
            for band_idx in range(n_bands):
                band_nm = float(SPECTRAL_BAND_CENTERS_NM[band_idx])
                value = float(vec[band_idx])
                long_rows.append([
                    frame, track_local_idx, global_q_idx,
                    spec_type, fpn_level, band_idx, band_nm, value,
                ])
                col = f"{spec_type}_lvl{fpn_level}_{band_nm:g}nm"
                wide_row[col] = value

        if wide_row:
            all_wide_cols.update(wide_row.keys())
            wide_records.append({
                "frame": frame,
                "track_local_idx": track_local_idx,
                "global_q_idx": global_q_idx,
                **wide_row,
            })

    if not long_rows:
        return

    os.makedirs(figure.track_out_dir or ".", exist_ok=True)

    with open(long_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "track_local_idx", "global_q_idx",
            "spec_type", "fpn_level", "band_index", "band_center_nm", "value",
        ])
        writer.writerows(long_rows)
    print(f"[Info] saved specRbox timeline csv (long): {long_path}")

    if wide_records and all_wide_cols:
        wide_header = ["frame", "track_local_idx", "global_q_idx"] + sorted(all_wide_cols)
        with open(wide_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(wide_header)
            for rec in wide_records:
                writer.writerow([rec.get(col, "") for col in wide_header])
        print(f"[Info] saved specRbox timeline csv (wide): {wide_path}")


def save_track_spectral_timeline(timeline: List[Dict[str, Any]], figure: FigureContext) -> None:
    if not timeline or figure.track_id is None:
        return

    def _stack_field(field: str) -> Optional[np.ndarray]:
        rows = []
        for item in timeline:
            if field not in item:
                continue
            row = np.asarray(item[field], dtype=np.float32).reshape(-1)
            rows.append(row)
        if not rows:
            return None
        max_len = max(row.shape[0] for row in rows)
        padded = []
        for row in rows:
            if row.shape[0] < max_len:
                row = np.concatenate([row, np.zeros((max_len - row.shape[0],), dtype=np.float32)])
            padded.append(row)
        return np.stack(padded, axis=0)

    field_names = sorted({
        key
        for item in timeline
        for key in item
        if key not in ("frame", "track_local_idx", "global_q_idx")
    })
    for name in field_names:
        mat = _stack_field(name)
        if mat is None:
            continue
        path = os.path.join(figure.track_out_dir, f"{figure.seq}__trackId{figure.track_id}__timeline_{name}.jpg")
        use_spectral = (
            name == "pooled_spectral"
            or name.startswith("init_pooledSpecRbox")
            or name.startswith("obs_pooledSpecRbox")
        )
        if use_spectral and mat.shape[1] == len(SPECTRAL_BAND_CENTERS_NM):
            saved = _save_spectral_band_grid(mat, path, figure.skip_existing, figure.vis_max_size)
            out_path = f"{os.path.splitext(path)[0]}_spectral.jpg"
        else:
            saved = _save_grid_heatmap(mat, path, figure.skip_existing, figure.vis_max_size)
            out_path = path
        if saved:
            print(f"[Info] saved timeline {name}: {out_path}")

    meta_path = os.path.join(figure.track_out_dir, f"{figure.seq}__trackId{figure.track_id}__timeline_meta.json")
    if not (figure.skip_existing and os.path.isfile(meta_path)):
        with open(meta_path, "w") as f:
            json.dump(timeline, f, indent=2)
        print(f"[Info] saved timeline meta: {meta_path}")

    _save_spec_rbox_timeline_csv(timeline=timeline, figure=figure)
