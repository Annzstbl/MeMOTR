"""
加载 checkpoint，对一条或多条视频序列做序贯推理，保存 SCEM token_debug 中的可视化图。

默认只保存 pool_weights；通过 --save-maps 可扩展其它图（evidence_weights / gate / support_map 等）。
每条序列可单独指定起始帧与结束帧（1-indexed，含结束帧）。
"""

"""
# 单序列，推理第 1 帧，保存 pool_weights（默认）
python debug/20260526/test.py \
  --train-config debug/20260331_99/debug_20260331_4.yaml \
  --checkpoint last.pth \
  --seq data39-1 \
  --end-frames 1
# 多序列，各自推理到不同帧
python debug/20260526/test.py \
  --seq data39-1,data48-1 \
  --end-frames 30,50
# 从第 5 帧推理到第 20 帧
python debug/20260526/test.py \
  --seq data39-1 \
  --start-frames 5 \
  --end-frames 20
# 同时保存多种 SCEM debug 图
python debug/20260526/test.py \
  --seq data39-1 \
  --end-frames 10 \
  --save-maps pool_weights,evidence_weights,gate
# 保存全部支持的图
python debug/20260526/test.py --save-maps all --end-frames none
"""
"""
CUDA_VISIBLE_DEVICES=1 \
python debug/20260526/test.py \
  --train-config /data1/users/litianhao01/hsmot/MeMOTR/debug/20260526/20260511-2.yaml \
  --checkpoint last.pth \
  --seq data39-3 \
  --end-frames 55

CUDA_VISIBLE_DEVICES=1 \
python debug/20260526/test.py \
  --train-config /data1/users/litianhao01/hsmot/MeMOTR/debug/20260526/20260511-2.yaml \
  --checkpoint last.pth \
  --seq data39-3,data28-5,data37-2,data30-3,data47-3,data31-1,data48-1,data33-1,data36-13,data37-1 \
  --end-frames 55,15,90,10,15,58,15,140,10,155 \
  --save-last-n-frames 20

  CUDA_VISIBLE_DEVICES=1 \
python debug/20260526/test.py \
  --train-config /data1/users/litianhao01/hsmot/MeMOTR/debug/20260526/20260511-2.yaml \
  --checkpoint last.pth \
  --seq data30-4\
  --end-frames 5 \
  --save-last-n-frames 20


"""


import os
import sys
import argparse
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

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


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

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


def _skip_image_write(path: str, skip_if_exists: bool) -> bool:
    return bool(skip_if_exists and os.path.isfile(path))


def _frame_tag(frame_num: int) -> str:
    return f"f{frame_num:06d}"


def _normalize_img_format(fmt: str) -> Tuple[str, str]:
    """
    解析图像目录格式。
    返回 (split 下子目录名, SeqDataset dataset_type)。
    """
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


def _init_track_instances_static(train_config: dict, model: nn.Module) -> None:
    """与 submit_engine._init_track_instances_static 保持一致。"""
    rect_bbox = is_rect_memotr_version(train_config.get("MEMOTR_VERSION", ""))
    TrackInstances.set_static_properties(
        use_spectral_decoder=train_config.get("DECODER_SPECTRAL", True),
        use_dab=train_config["USE_DAB"],
        use_q_spec=bool(getattr(get_model(model), "use_q_spec", False)),
        bbox_dim=4 if rect_bbox else 5,
    )


# ---------------------------------------------------------------------------
# 序列配置：支持 per-seq 起止帧
# ---------------------------------------------------------------------------

def _parse_seq_specs(
    seq_arg: str,
    start_frames_arg: Optional[str],
    end_frames_arg: Optional[str],
    default_start: int,
    default_end: Optional[int],
) -> List[Dict[str, Any]]:
    """
    解析序列列表及每条序列的起止帧（1-indexed，含结束帧）。

    --seq data39-1,data48-1
    --start-frames 1,5          # 与 seq 一一对应；仅一个值则复用到全部
    --end-frames 30,50          # 与 seq 一一对应；仅一个值则复用到全部
    """
    seq_list = [s.strip() for s in seq_arg.split(",") if s.strip()]
    if not seq_list:
        raise ValueError("Empty --seq.")

    def _expand_values(raw: Optional[str], n: int, name: str) -> List[Optional[int]]:
        if raw is None:
            return [None] * n
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == 1:
            return [int(parts[0])] * n
        if len(parts) != n:
            raise ValueError(f"--{name} count ({len(parts)}) must match --seq count ({n}) or be 1.")
        return [int(p) for p in parts]

    starts = _expand_values(start_frames_arg, len(seq_list), "start-frames")
    ends = _expand_values(end_frames_arg, len(seq_list), "end-frames")

    specs: List[Dict[str, Any]] = []
    for i, seq in enumerate(seq_list):
        start = starts[i] if starts[i] is not None else default_start
        end = ends[i] if ends[i] is not None else default_end
        if start < 1:
            raise ValueError(f"start frame must be >= 1, got {start} for seq {seq}")
        if end is not None and end < start:
            raise ValueError(f"end frame ({end}) must be >= start frame ({start}) for seq {seq}")
        specs.append({"seq": seq, "start_frame": start, "end_frame": end})
    return specs


# ---------------------------------------------------------------------------
# 序贯推理
# ---------------------------------------------------------------------------

def _build_frame_meta(frame_num: int, outputs: Dict[str, Any]) -> Dict[str, Any]:
    frame_meta: Dict[str, Any] = {"frame": frame_num, "outputs": {}}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            frame_meta["outputs"][k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        elif k == "scem_token_debug" and isinstance(v, dict):
            frame_meta["outputs"][k] = {
                sub_k: ("tensor" if torch.is_tensor(sub_v) else type(sub_v).__name__)
                for sub_k, sub_v in v.items()
            }
    return frame_meta


def run_forward_sequential(
    model: nn.Module,
    train_config: dict,
    seq_dir: str,
    npy2rgb: bool,
    dataset_type: Optional[str],
    start_frame: int,
    end_frame: Optional[int],
    seq: str,
    map_out_dir: str,
    save_maps: List[str],
    save_last_n_frames: int = 30,
    skip_existing_images: bool = False,
    vis_max_size: int = 1200,
) -> List[Dict[str, Any]]:
    """
    序贯处理 [start_frame, end_frame]（1-indexed，含两端）。
    推理跑完整段区间以维持 track 状态，但仅保存末尾 save_last_n_frames 帧的 SCEM 图。
    """
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
    if save_last_n_frames > 0:
        save_from_frame = max(start_frame, actual_end_frame - save_last_n_frames + 1)
    else:
        save_from_frame = start_frame

    print(
        f"[Info] Processing frames {start_frame}-{actual_end_frame} "
        f"({num_frames} frames, dataset total={len(dataset)})"
    )
    if save_from_frame > start_frame:
        print(
            f"[Info] Save maps for last {save_last_n_frames} frames only: "
            f"{save_from_frame}-{actual_end_frame}"
        )
    else:
        print(f"[Info] Save maps for frames {save_from_frame}-{actual_end_frame}")

    inner_model = get_model(model)
    use_dab = train_config["USE_DAB"]
    decoder_spectral = train_config.get("DECODER_SPECTRAL", True)

    tracker = RuntimeTracker(
        det_score_thresh=0.5,
        track_score_thresh=0.5,
        miss_tolerance=30,
        use_motion=False,
        motion_min_length=3,
        motion_max_length=5,
        visualize=False,
        use_dab=use_dab,
        decoder_spectral=decoder_spectral,
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

    saved_meta: List[Dict[str, Any]] = []
    prev_frame = None

    model.eval()
    with torch.no_grad():
        for i in range(num_frames):
            dataset_idx = start_idx + i
            frame_num = start_frame + i
            should_save = frame_num >= save_from_frame
            log_tag = "forward+save" if should_save else "forward"
            print(f"[Info] {log_tag} frame {frame_num} (index {dataset_idx})")

            image, ori_image = dataset[dataset_idx][0]
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

            if gmc is not None:
                res = model(frame=frame, tracks=tracks, gmc=gmc, debug=should_save)
            else:
                res = model(frame=frame, tracks=tracks, debug=should_save)

            previous_tracks, new_tracks = tracker.update(model_outputs=res, tracks=tracks)
            tracks = inner_model.postprocess_single_frame(previous_tracks, new_tracks, None)

            if should_save and isinstance(res, dict):
                token_debug = res.get("scem_token_debug")
                if token_debug is not None:
                    print(f"[Info] Saving SCEM maps frame {frame_num}: {save_maps}")
                    save_scem_token_debug_maps(
                        token_debug=token_debug,
                        out_dir=map_out_dir,
                        seq=seq,
                        frame_num=frame_num,
                        save_maps=save_maps,
                        skip_existing=skip_existing_images,
                        frame_pad_mask=frame.masks[0].detach(),
                        vis_max_size=vis_max_size,
                    )
                    saved_meta.append(_build_frame_meta(frame_num, res))
                else:
                    print(f"[Warn] No scem_token_debug at frame {frame_num}")

            del res, frame, image

    return saved_meta


# ---------------------------------------------------------------------------
# SCEM 可视化：可扩展 saver 注册表
# ---------------------------------------------------------------------------

SaveContext = Dict[str, Any]
MapSaverFn = Callable[[Dict[str, Any], SaveContext], None]


def _iter_token_levels(token_debug: Dict[str, Any]):
    """yield (src_lvl, ew_lvl, pw_lvl, pl_lvl, gate_lvl, support_lvl)"""
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
    """将输入 padding mask 下采样到目标尺度。True 表示无效 padding 区域。"""
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
    if pad_mask is None:
        return None
    if pad_mask.shape != arr_shape:
        return None
    return pad_mask


def _upscale_gray(gray: np.ndarray, max_side: int) -> np.ndarray:
    """放大灰度图，对齐原 debug.py 的 figsize=4, dpi=300（约 1200px）。"""
    if max_side <= 0:
        return gray
    h, w = gray.shape[:2]
    if max(h, w) >= max_side:
        return gray
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _upscale_color(color: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return color
    h, w = color.shape[:2]
    if max(h, w) >= max_side:
        return color
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _normalize_map_to_gray(
    arr2d: np.ndarray,
    pad_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """归一化到 uint8 灰度，并返回 valid mask（True=有效区域）。"""
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
    """保存无 colorbar 的 JPG 灰度图；padding/mask 区域置 0，并按 vis_max_size 放大。"""
    jpg_path = os.path.splitext(path)[0] + ".jpg"
    if _skip_image_write(jpg_path, skip_existing):
        return False

    gray, _valid = _normalize_map_to_gray(arr2d, pad_mask)
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
    """保存 jet 伪彩色 JPG（无 colorbar，仅内容）；padding/mask 区域置黑。"""
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
    saved_gray = _save_grayscale_jpg(
        arr2d, path, skip_existing, pad_mask=pad_mask, vis_max_size=vis_max_size
    )
    saved_color = _save_colormap_jpg(
        arr2d, path, skip_existing, pad_mask=pad_mask, vis_max_size=vis_max_size
    )
    return saved_gray or saved_color


def save_pool_weights_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    seq = ctx["seq"]
    frame_tag = ctx["frame_tag"]
    out_dir = ctx["out_dir"]
    skip_existing = ctx["skip_existing"]

    for src_lvl, _, pw_lvl, _, _, _ in _iter_token_levels(token_debug):
        num_tokens = pw_lvl.shape[1]
        for tok_idx in range(num_tokens):
            pw_map = pw_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                out_dir,
                f"{seq}__{frame_tag}__scem_srcL{src_lvl}_tok{tok_idx}__poolW.jpg",
            )
            if _save_map_jpg(pw_map, path, ctx, src_lvl):
                print(f"  saved pool_weights: {path}")


def save_evidence_weights_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    seq = ctx["seq"]
    frame_tag = ctx["frame_tag"]
    out_dir = ctx["out_dir"]
    skip_existing = ctx["skip_existing"]

    for src_lvl, ew_lvl, pw_lvl, _, _, _ in _iter_token_levels(token_debug):
        ew_pos, ew_neg = _parse_evidence_pos_neg(ew_lvl)
        if ew_pos is None:
            continue
        num_tokens = min(ew_pos.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            ew_pos_mean = ew_pos[0, tok_idx].mean(dim=0).detach().cpu().numpy()
            path = os.path.join(
                out_dir,
                f"{seq}__{frame_tag}__scem_srcL{src_lvl}_tok{tok_idx}__evidenceW_mean.jpg",
            )
            if _save_map_jpg(ew_pos_mean, path, ctx, src_lvl):
                print(f"  saved evidence_weights: {path}")

            if ew_neg is not None and tok_idx < ew_neg.shape[1]:
                ew_neg_mean = ew_neg[0, tok_idx].mean(dim=0).detach().cpu().numpy()
                neg_path = os.path.join(
                    out_dir,
                    f"{seq}__{frame_tag}__scem_srcL{src_lvl}_tok{tok_idx}__evidenceW_neg_mean.jpg",
                )
                if _save_map_jpg(ew_neg_mean, neg_path, ctx, src_lvl):
                    print(f"  saved evidence_weights_neg: {neg_path}")


def save_pool_logits_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    seq = ctx["seq"]
    frame_tag = ctx["frame_tag"]
    out_dir = ctx["out_dir"]
    skip_existing = ctx["skip_existing"]

    for src_lvl, _, pw_lvl, pl_lvl, _, _ in _iter_token_levels(token_debug):
        if not torch.is_tensor(pl_lvl) or pl_lvl.dim() != 4:
            continue
        num_tokens = min(pl_lvl.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            pl_map = pl_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                out_dir,
                f"{seq}__{frame_tag}__scem_srcL{src_lvl}_tok{tok_idx}__poolLogits.jpg",
            )
            if _save_map_jpg(pl_map, path, ctx, src_lvl):
                print(f"  saved pool_logits: {path}")


def save_gate_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    seq = ctx["seq"]
    frame_tag = ctx["frame_tag"]
    out_dir = ctx["out_dir"]
    skip_existing = ctx["skip_existing"]

    for src_lvl, _, pw_lvl, _, gate_lvl, _ in _iter_token_levels(token_debug):
        if not torch.is_tensor(gate_lvl) or gate_lvl.dim() != 4:
            continue
        num_tokens = min(gate_lvl.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            gate_map = gate_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                out_dir,
                f"{seq}__{frame_tag}__scem_srcL{src_lvl}_tok{tok_idx}__gate.jpg",
            )
            if _save_map_jpg(gate_map, path, ctx, src_lvl):
                print(f"  saved gate: {path}")


def save_support_map_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    seq = ctx["seq"]
    frame_tag = ctx["frame_tag"]
    out_dir = ctx["out_dir"]
    skip_existing = ctx["skip_existing"]

    for src_lvl, _, pw_lvl, _, _, support_lvl in _iter_token_levels(token_debug):
        if not torch.is_tensor(support_lvl) or support_lvl.dim() != 4:
            continue
        num_tokens = min(support_lvl.shape[1], pw_lvl.shape[1])
        for tok_idx in range(num_tokens):
            smap = support_lvl[0, tok_idx].detach().cpu().numpy()
            path = os.path.join(
                out_dir,
                f"{seq}__{frame_tag}__scem_srcL{src_lvl}_tok{tok_idx}__support_map.jpg",
            )
            if _save_map_jpg(smap, path, ctx, src_lvl):
                print(f"  saved support_map: {path}")


def save_spectral_evidence_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    seq = ctx["seq"]
    frame_tag = ctx["frame_tag"]
    out_dir = ctx["out_dir"]
    skip_existing = ctx["skip_existing"]

    for lvl, spec_evi in enumerate(token_debug.get("spectral_evidence_multilevel", [])):
        if not torch.is_tensor(spec_evi) or spec_evi.dim() != 4:
            continue
        spec_evi_0 = spec_evi[0]
        for ch in range(spec_evi_0.shape[0]):
            arr2d = spec_evi_0[ch].detach().cpu().numpy()
            path = os.path.join(
                out_dir,
                f"{seq}__{frame_tag}__scem_specEvi_L{lvl}_ch{ch}.jpg",
            )
            if _save_map_jpg(arr2d, path, ctx, lvl):
                print(f"  saved spectral_evidence: {path}")


def save_feature_fusion_maps(token_debug: Dict[str, Any], ctx: SaveContext) -> None:
    seq = ctx["seq"]
    frame_tag = ctx["frame_tag"]
    out_dir = ctx["out_dir"]
    skip_existing = ctx["skip_existing"]

    for lvl, ft in enumerate(token_debug.get("feature_fusion_feat_list", [])):
        if not torch.is_tensor(ft) or ft.dim() != 4:
            continue
        energy = torch.sqrt((ft[0] ** 2).sum(dim=0)).detach().cpu().numpy()
        fe_path = os.path.join(out_dir, f"{seq}__{frame_tag}__scem_featfuse_featL{lvl}__featE.jpg")
        if _save_map_jpg(energy, fe_path, ctx, lvl):
            print(f"  saved feature_fusion featE: {fe_path}")

    for lvl, sp in enumerate(token_debug.get("feature_fusion_spec_list", [])):
        if not torch.is_tensor(sp) or sp.dim() != 4:
            continue
        sp0 = sp[0]
        for ch in range(sp0.shape[0]):
            arr2d = sp0[ch].detach().cpu().numpy()
            path = os.path.join(
                out_dir,
                f"{seq}__{frame_tag}__scem_featfuse_specL{lvl}_ch{ch}.jpg",
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


def register_map_saver(name: str, fn: MapSaverFn) -> None:
    """扩展入口：注册新的 SCEM debug 可视化 saver。"""
    MAP_SAVER_REGISTRY[name] = fn


def _resolve_save_maps(raw: str) -> List[str]:
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if not names:
        return ["pool_weights"]
    if len(names) == 1 and names[0].lower() == "all":
        return list(MAP_SAVER_REGISTRY.keys())
    unknown = [n for n in names if n not in MAP_SAVER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown save-maps: {unknown}. Available: {list(MAP_SAVER_REGISTRY.keys())}, all"
        )
    return names


def save_scem_token_debug_maps(
    token_debug: Dict[str, Any],
    out_dir: str,
    seq: str,
    frame_num: int,
    save_maps: List[str],
    skip_existing: bool = False,
    frame_pad_mask: Optional[torch.Tensor] = None,
    vis_max_size: int = 1200,
) -> None:
    if not isinstance(token_debug, dict):
        return

    ctx: SaveContext = {
        "seq": seq,
        "frame_num": frame_num,
        "frame_tag": _frame_tag(frame_num),
        "out_dir": out_dir,
        "skip_existing": skip_existing,
        "level_pad_masks": _build_level_pad_masks(frame_pad_mask, token_debug),
        "vis_max_size": vis_max_size,
    }
    for name in save_maps:
        saver = MAP_SAVER_REGISTRY[name]
        try:
            saver(token_debug, ctx)
        except Exception as e:
            print(f"[Warn] saver '{name}' failed for seq={seq} frame={frame_num}: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load model, run sequential inference, save SCEM pool_weights and optional debug maps."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="debug_20260331_4.yaml",
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
    parser.add_argument(
        "--seq",
        type=str,
        default="data39-1",
        help="Comma-separated sequence names.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="Default start frame (1-indexed) when --start-frames is not set.",
    )
    parser.add_argument(
        "--start-frames",
        type=str,
        default=None,
        help="Per-seq start frames, comma-separated; one value applies to all.",
    )
    parser.add_argument(
        "--end-frames",
        type=str,
        default="1",
        help="Per-seq end frames (1-indexed, inclusive); one value applies to all. Omit to run until sequence end.",
    )
    parser.add_argument(
        "--img-format",
        type=str,
        default="npy2jpg",
        choices=["npy2jpg", "npy"],
        help="Image directory format under split/: npy2jpg (3JPG, default) or npy (NPY).",
    )
    parser.add_argument(
        "--npy2rgb",
        action="store_true",
        help="Use 3-channel subset (channels 1,2,4) instead of full 8-channel input.",
    )
    parser.add_argument(
        "--save-maps",
        type=str,
        default="pool_weights",
        help=(
            "Comma-separated map types to save. "
            f"Available: {','.join(MAP_SAVER_REGISTRY.keys())}, all"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root; default: {script_dir}/{config_basename}/heatmaps",
    )
    parser.add_argument(
        "--save-last-n-frames",
        type=int,
        default=30,
        help="Only save SCEM maps for the last N frames in each sequence (default: 30). "
        "Inference still runs from start_frame for correct tracking. Use 0 to save all frames.",
    )
    parser.add_argument("--skip-existing-images", action="store_true")
    parser.add_argument(
        "--vis-max-size",
        type=int,
        default=1200,
        help="Max side length when upscaling saved JPG maps (default 1200, same as debug.py figsize=4 @ dpi=300).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dump-json", type=str, default="", help="Dump per-frame metadata json.")

    args = parser.parse_args()
    save_maps = _resolve_save_maps(args.save_maps)

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

    config_name = os.path.splitext(os.path.basename(train_cfg_path))[0]
    heatmap_root = args.output_dir or os.path.join(CURRENT_DIR, config_name, "heatmaps")
    os.makedirs(heatmap_root, exist_ok=True)

    end_frames_raw = args.end_frames
    if end_frames_raw is not None and end_frames_raw.strip().lower() in ("none", "all"):
        end_frames_raw = None

    base_name = args.dataset_name.replace("_8ch", "") if "hsmot" in args.dataset_name else args.dataset_name
    img_subdir, dataset_type = _resolve_img_format(args.img_format, train_config)
    img_root = _resolve_img_root(args.data_root, base_name, args.split, img_subdir)
    print(f"[Info] img format={args.img_format} -> {img_root} (dataset_type={dataset_type})")

    seq_arg = args.seq
    if seq_arg.lower() == "all":
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"image root not found: {img_root}")
        seq_arg = ",".join(
            sorted(d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d)))
        )
        print(f"[Info] --seq all -> {len(seq_arg.split(','))} sequences")

    seq_specs = _parse_seq_specs(
        seq_arg=seq_arg,
        start_frames_arg=args.start_frames,
        end_frames_arg=end_frames_raw,
        default_start=args.start_frame,
        default_end=None,
    )

    print(f"[Info] Building model from: {train_cfg_path}")
    model = build_model(config=train_config)
    device = torch.device(args.device)
    model.to(device)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[Info] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    _init_track_instances_static(train_config, model)

    inner = get_model(model)
    if not getattr(inner, "scem_module", None):
        print("[Warn] Model has no scem_module; scem_token_debug may be absent.")

    all_meta: Dict[str, Any] = {"config": train_cfg_path, "checkpoint": checkpoint_path, "per_seq": []}

    for spec in seq_specs:
        seq = spec["seq"]
        seq_dir = os.path.join(img_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[Warn] Skip missing seq dir: {seq_dir}")
            continue

        print(f"\n[Info] seq={seq}, frames [{spec['start_frame']}, {spec['end_frame'] or 'end'}]")
        map_out_dir = os.path.join(heatmap_root, seq)
        os.makedirs(map_out_dir, exist_ok=True)

        saved_frames = run_forward_sequential(
            model=model,
            train_config=train_config,
            seq_dir=seq_dir,
            npy2rgb=args.npy2rgb,
            dataset_type=dataset_type,
            start_frame=spec["start_frame"],
            end_frame=spec["end_frame"],
            seq=seq,
            map_out_dir=map_out_dir,
            save_maps=save_maps,
            save_last_n_frames=args.save_last_n_frames,
            skip_existing_images=args.skip_existing_images,
            vis_max_size=args.vis_max_size,
        )

        all_meta["per_seq"].append({"seq": seq, "frames": saved_frames})

    if args.dump_json:
        try:
            with open(args.dump_json, "w") as f:
                json.dump(all_meta, f, indent=2)
            print(f"[Info] Dumped metadata to {args.dump_json}")
        except Exception as e:
            print(f"[Warn] Failed to dump json: {e}")

    print(f"\n[Done] Outputs under: {heatmap_root}")


if __name__ == "__main__":
    main()
