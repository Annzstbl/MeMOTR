"""
加载 checkpoint，对一条或多条视频序列做序贯推理，保存 SCEM token_debug 中的可视化图。

使用 model_20260511_figure：可视化由模型 forward 内直接落盘。
默认只保存 pool_weights；通过 --save-maps 可扩展其它图。

Track query 可视化示例：
CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/test2.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data31-1 \
  --end-frames 69 \
  --save-last-n-frames 0 \
  --vis-track-id 0 \
  --vis-track-spectral \
  --vis-cross-attn \
  --save-maps none

  CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/test2.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data33-1 \
  --end-frames 137 \
  --save-last-n-frames 0 \
  --vis-track-id 4 \
  --vis-track-spectral \
  --vis-cross-attn \
  --save-maps none

CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/test3.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data36-4 \
  --end-frames 30 \
  --save-last-n-frames 0 \
  --vis-track-id 3 \
  --vis-cross-attn \
  --vis-track-spectral \
  --save-maps none


SE 逐像素统计示例（跨 n 条序列、所有帧，统计 stem SE 8 通道 mean/var）：
conda activate hsmot
cd /data1/users/litianhao01/hsmot/MeMOTR
CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/test3.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data31-1,data33-1,data39-1 \
  --end-frames all \
  --save-maps none \
  --analyze-se \
  --se-save-vis

输出目录：debug/20260526/<config>/se_stats/
  - se_all_mean.npy   [8, H/2, W/2] 每通道逐像素均值
  - se_all_var.npy    [8, H/2, W/2] 每通道逐像素方差
  - se_all_std.npy    [8, H/2, W/2] 标准差
  - se_all_count.npy  [H/2, W/2]     每像素累计帧数
  - se_all_mean_std_grid.png         8 通道可视化（可选 --se-save-vis）
  加 --se-per-seq 可额外保存每条序列单独的统计
  加 --se-apply-sigmoid 可对 sig_raw 做 sigmoid 后再统计

Track ID 预览（先确认要画哪个 id）：
CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/test3.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data36-4 \
  --end-frames 5 \
  --save-maps none \
  --vis-track-preview \
  --vis-track-preview-n 5
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
from models.model_20260511_figure.figure_context import FigureContext
from models.model_20260511_figure.figure_save import MAP_SAVER_REGISTRY, save_track_spectral_timeline
from utils.utils import load_yaml_with_inheritance
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.GMC import compute_gmc_sequence
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset
from hsmot.datasets.pipelines.channel import rotate_norm_boxes_to_boxes
from hsmot.mmlab.hs_mmrotate import obb2poly
from hsmot.mmlab.hs_rectmot import denormalize_cxcywh
from utils.box_ops import box_cxcywh_to_xyxy

SPECTRAL_BAND_CENTERS_NM = [422.5, 487.5, 550.0, 602.5, 660.0, 725.0, 785.0, 887.2]


# ---------------------------------------------------------------------------
# SE (stem ConvMSI_SE) 逐像素统计
# ---------------------------------------------------------------------------

class SEStatsAccumulator:
    """Welford 在线算法，逐像素、逐通道累计 mean / variance。"""

    def __init__(self, n_channels: int = 8):
        self.n_channels = n_channels
        self.count: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.num_updates = 0

    def _init_buffers(self, se: torch.Tensor) -> None:
        _, c, h, w = se.shape
        if c != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} SE channels, got {c}.")
        self.count = torch.zeros(h, w, dtype=torch.int64, device=se.device)
        self.mean = torch.zeros(c, h, w, dtype=torch.float64, device=se.device)
        self.m2 = torch.zeros(c, h, w, dtype=torch.float64, device=se.device)

    def update(self, se: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> None:
        if se.dim() == 3:
            se = se.unsqueeze(0)
        if se.shape[0] != 1:
            raise ValueError(f"SEStatsAccumulator expects batch size 1, got {se.shape[0]}.")

        se = se.detach().to(dtype=torch.float64)
        if self.mean is None:
            self._init_buffers(se)
        elif se.shape[-2:] != self.mean.shape[-2:]:
            raise ValueError(
                f"SE spatial size mismatch: current {tuple(se.shape[-2:])}, "
                f"expected {tuple(self.mean.shape[-2:])}."
            )

        x = se[0]
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[-2:], dtype=torch.bool, device=x.device)
        else:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            if valid_mask.shape != x.shape[-2:]:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} != SE spatial {tuple(x.shape[-2:])}."
                )

        if not valid_mask.any():
            return

        self.num_updates += 1
        flat_valid = valid_mask.reshape(-1)
        x_valid = x.reshape(self.n_channels, -1)[:, flat_valid]
        count_valid = self.count.reshape(-1)[flat_valid].to(dtype=torch.float64)
        mean_valid = self.mean.reshape(self.n_channels, -1)[:, flat_valid]
        m2_valid = self.m2.reshape(self.n_channels, -1)[:, flat_valid]

        n_new = count_valid + 1.0
        delta = x_valid - mean_valid
        mean_valid = mean_valid + delta / n_new.unsqueeze(0)
        m2_valid = m2_valid + delta * (x_valid - mean_valid)

        self.count.reshape(-1)[flat_valid] = n_new.to(dtype=torch.int64)
        self.mean.reshape(self.n_channels, -1)[:, flat_valid] = mean_valid
        self.m2.reshape(self.n_channels, -1)[:, flat_valid] = m2_valid

    def finalize(self) -> Dict[str, np.ndarray]:
        if self.mean is None:
            raise RuntimeError("No SE samples accumulated.")
        count = self.count.cpu().numpy().astype(np.int32)
        mean = self.mean.cpu().numpy().astype(np.float32)
        var = np.zeros_like(mean, dtype=np.float32)
        valid = count > 1
        if valid.any():
            var[:, valid] = (self.m2.cpu().numpy()[:, valid] / (count[valid] - 1)).astype(np.float32)
        var[:, count == 1] = 0.0
        return {"count": count, "mean": mean, "var": var, "std": np.sqrt(np.maximum(var, 0.0))}

    def save(self, out_dir: str, prefix: str = "se") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        stats = self.finalize()
        paths = {}
        for key in ("mean", "var", "std", "count"):
            path = os.path.join(out_dir, f"{prefix}_{key}.npy")
            np.save(path, stats[key])
            paths[key] = path

        meta = {
            "num_frame_updates": self.num_updates,
            "spatial_shape_hw": list(stats["mean"].shape[-2:]),
            "n_channels": self.n_channels,
            "band_centers_nm": SPECTRAL_BAND_CENTERS_NM,
            "files": paths,
        }
        meta_path = os.path.join(out_dir, f"{prefix}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        paths["meta"] = meta_path

        channel_summary = summarize_se_spatial(stats)
        ch_json = os.path.join(out_dir, f"{prefix}_channel_summary.json")
        ch_payload = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in channel_summary.items()
        }
        with open(ch_json, "w", encoding="utf-8") as f:
            json.dump(ch_payload, f, indent=2, ensure_ascii=False)
        paths["channel_summary"] = ch_json
        return paths


def summarize_se_spatial(
    stats: Dict[str, np.ndarray],
    min_count: int = 1,
    weighted: bool = True,
) -> Dict[str, Any]:
    """[8,H,W] 逐像素统计 -> [8] 通道均值/方差（去掉空间维）。"""
    mean_map = stats["mean"]
    var_map = stats["var"]
    count_map = stats["count"]
    valid = count_map >= min_count
    if not np.any(valid):
        raise ValueError(f"No pixel with count >= {min_count}")

    mu = mean_map[:, valid].astype(np.float64)
    var_pix = var_map[:, valid].astype(np.float64)
    if weighted:
        w = count_map[valid].astype(np.float64)
        w = w / w.sum()
    else:
        w = np.full(int(valid.sum()), 1.0 / valid.sum(), dtype=np.float64)

    channel_mean = (mu * w[None]).sum(axis=1)
    channel_var_temporal = (var_pix * w[None]).sum(axis=1)
    channel_var_spatial = (w[None] * (mu - channel_mean[:, None]) ** 2).sum(axis=1)
    channel_var_total = channel_var_temporal + channel_var_spatial

    return {
        "n_channels": mean_map.shape[0],
        "n_valid_pixels": int(valid.sum()),
        "min_count": min_count,
        "weighted_by_frame_count": weighted,
        "band_centers_nm": SPECTRAL_BAND_CENTERS_NM,
        "channel_mean": channel_mean.astype(np.float32),
        "channel_var_temporal": channel_var_temporal.astype(np.float32),
        "channel_var_spatial": channel_var_spatial.astype(np.float32),
        "channel_var_total": channel_var_total.astype(np.float32),
        "channel_std_total": np.sqrt(np.maximum(channel_var_total, 0)).astype(np.float32),
    }


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
    """pad_mask: True=padding；返回 valid_mask: True=有效像素。"""
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


def save_se_stats_visualization(stats: Dict[str, np.ndarray], out_dir: str, prefix: str = "se") -> str:
    import matplotlib.pyplot as plt

    mean = stats["mean"]
    std = stats["std"]
    c, h, w = mean.shape
    fig, axes = plt.subplots(2, c, figsize=(2.4 * c, 4.8), squeeze=False)
    for ch in range(c):
        wl = SPECTRAL_BAND_CENTERS_NM[ch]
        im0 = axes[0, ch].imshow(mean[ch], cmap="viridis", aspect="auto")
        axes[0, ch].set_title(f"mean ch{ch}\n{wl:.1f}nm", fontsize=9)
        axes[0, ch].axis("off")
        fig.colorbar(im0, ax=axes[0, ch], fraction=0.046, pad=0.02)

        im1 = axes[1, ch].imshow(std[ch], cmap="magma", aspect="auto")
        axes[1, ch].set_title(f"std ch{ch}", fontsize=9)
        axes[1, ch].axis("off")
        fig.colorbar(im1, ax=axes[1, ch], fraction=0.046, pad=0.02)

    fig.suptitle(f"{prefix}: per-pixel SE mean (top) / std (bottom)", fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_mean_std_grid.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


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


def _resolve_track_vis_out_dir(config_root: str, seq: str, track_id: int) -> str:
    return os.path.join(config_root, "track", seq, f"id{track_id}")


def _resolve_track_preview_out_dir(config_root: str, seq: str) -> str:
    return os.path.join(config_root, "track_preview", seq)


def _color_by_track_id(track_id: int) -> Tuple[int, int, int]:
    hue = int(180 * (track_id % 100) / 100)
    hsv = np.uint8([[[hue, 220, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in bgr)


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


def _draw_track_id_label(
    img: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    x, y = anchor
    y = max(16, y)
    cv2.putText(
        img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
    )


def _save_track_preview_frame(
    ori_image: np.ndarray,
    tracks: TrackInstances,
    frame_num: int,
    seq: str,
    out_dir: str,
    rect_bbox: bool,
    track_score_thresh: float,
) -> str:
    eff_h, eff_w = int(ori_image.shape[0]), int(ori_image.shape[1])
    vis = np.ascontiguousarray(_ori_to_vis_rgb(ori_image))

    active = 0
    for i in range(len(tracks)):
        obj_id = int(tracks.ids[i].item())
        if obj_id < 0:
            continue
        score = float(torch.max(tracks.scores[i]).item())
        if score < track_score_thresh:
            continue

        color = _color_by_track_id(obj_id)
        label = str(obj_id)

        if rect_bbox:
            box = denormalize_cxcywh(tracks.boxes[i : i + 1].cpu(), (eff_h, eff_w))
            x1, y1, x2, y2 = box_cxcywh_to_xyxy(box)[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
            _draw_track_id_label(vis, label, (x1, y1), color)
        else:
            box_xywha = rotate_norm_boxes_to_boxes(
                tracks.boxes[i : i + 1].cpu(), (eff_h, eff_w), version="le135"
            )
            poly = obb2poly(box_xywha)[0].detach().cpu().numpy().reshape(-1, 2).astype(np.int32)
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=1)
            anchor = (int(poly[0, 0]), int(poly[0, 1]))
            _draw_track_id_label(vis, label, anchor, color)
        active += 1

    frame_tag = f"frame{frame_num:04d}"
    header = f"{seq} {frame_tag}  active_tracks={active}"
    cv2.putText(
        vis, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq}__{frame_tag}__all_track_ids.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return out_path


def _init_track_instances_static(train_config: dict, model: nn.Module) -> None:
    rect_bbox = is_rect_memotr_version(train_config.get("MEMOTR_VERSION", ""))
    TrackInstances.set_static_properties(
        use_spectral_decoder=train_config.get("DECODER_SPECTRAL", True),
        use_dab=train_config["USE_DAB"],
        use_q_spec=bool(getattr(get_model(model), "use_q_spec", False)),
        bbox_dim=4 if rect_bbox else 5,
    )


def _parse_seq_specs(
    seq_arg: str,
    start_frames_arg: Optional[str],
    end_frames_arg: Optional[str],
    default_start: int,
    default_end: Optional[int],
) -> List[Dict[str, Any]]:
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


def _resolve_save_maps(raw: str) -> List[str]:
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if not names:
        return ["pool_weights"]
    if len(names) == 1 and names[0].lower() == "none":
        return []
    if len(names) == 1 and names[0].lower() == "all":
        return list(MAP_SAVER_REGISTRY.keys())
    unknown = [n for n in names if n not in MAP_SAVER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown save-maps: {unknown}. Available: {list(MAP_SAVER_REGISTRY.keys())}, all, none"
        )
    return names


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


def _find_track_local_index(tracks: TrackInstances, track_id: int) -> Optional[int]:
    if len(tracks) == 0:
        return None
    ids = tracks.ids.detach().cpu()
    matches = (ids == track_id).nonzero(as_tuple=False).view(-1)
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"[Warn] Multiple track queries with id={track_id}, use first index {matches[0].item()}")
    return int(matches[0].item())


def _global_query_index(n_det_queries: int, track_local_idx: int) -> int:
    return n_det_queries + track_local_idx


def _build_figure_context(
    *,
    enabled: bool,
    seq: str,
    frame_num: int,
    map_out_dir: str,
    track_vis_out_dir: Optional[str],
    save_maps: List[str],
    vis_track_id: Optional[int],
    track_local_idx: Optional[int],
    global_q_idx: Optional[int],
    vis_track_spectral: bool,
    vis_cross_attn: bool,
    skip_existing: bool,
    vis_max_size: int,
    frame_pad_mask: Optional[torch.Tensor],
    spectral_timeline: Optional[List[Dict[str, Any]]],
) -> Optional[FigureContext]:
    if not enabled:
        return None
    return FigureContext(
        enabled=True,
        seq=seq,
        frame_num=frame_num,
        scem_out_dir=map_out_dir,
        track_out_dir=track_vis_out_dir or "",
        save_maps=tuple(save_maps),
        track_id=vis_track_id,
        track_local_idx=track_local_idx,
        global_q_idx=global_q_idx,
        save_track_spectral=vis_track_spectral,
        save_cross_attn=vis_cross_attn,
        skip_existing=skip_existing,
        vis_max_size=vis_max_size,
        frame_pad_mask=frame_pad_mask,
        spectral_timeline=spectral_timeline,
    )


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
    vis_track_id: Optional[int] = None,
    vis_track_spectral: bool = False,
    vis_cross_attn: bool = False,
    track_vis_out_dir: Optional[str] = None,
    vis_track_preview: bool = False,
    vis_track_preview_n: int = 5,
    track_preview_out_dir: Optional[str] = None,
    rect_bbox: bool = False,
    se_hook: Optional[SECaptureHook] = None,
    se_accumulators: Optional[List[SEStatsAccumulator]] = None,
    se_apply_sigmoid: bool = False,
) -> List[Dict[str, Any]]:
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
    save_from_frame = (
        max(start_frame, actual_end_frame - save_last_n_frames + 1)
        if save_last_n_frames > 0
        else start_frame
    )

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

    saved_meta: List[Dict[str, Any]] = []
    prev_frame = None
    track_vis_enabled = vis_track_id is not None and (vis_track_spectral or vis_cross_attn)
    preview_end_frame = min(actual_end_frame, start_frame + max(vis_track_preview_n, 0) - 1)
    preview_frames = (
        set(range(start_frame, preview_end_frame + 1))
        if vis_track_preview and vis_track_preview_n > 0
        else set()
    )
    spectral_timeline: List[Dict[str, Any]] = []
    if vis_track_preview:
        if not track_preview_out_dir:
            raise ValueError("track_preview_out_dir is required when --vis-track-preview is set.")
        os.makedirs(track_preview_out_dir, exist_ok=True)
        print(
            f"[Info] Track ID preview: frames {start_frame}-{preview_end_frame} "
            f"({len(preview_frames)} frames), out={track_preview_out_dir}"
        )
    if track_vis_enabled:
        if not track_vis_out_dir:
            raise ValueError("track_vis_out_dir is required when track visualization is enabled.")
        os.makedirs(track_vis_out_dir, exist_ok=True)
        print(
            f"[Info] Track vis enabled: id={vis_track_id}, "
            f"spectral={vis_track_spectral}, cross_attn={vis_cross_attn}, "
            f"out={track_vis_out_dir}"
        )

    model.eval()
    with torch.no_grad():
        for i in range(num_frames):
            dataset_idx = start_idx + i
            frame_num = start_frame + i
            should_save = frame_num >= save_from_frame
            need_debug = should_save or track_vis_enabled
            log_tag = "forward+save" if should_save else ("forward+track_vis" if track_vis_enabled else "forward")
            print(f"[Info] {log_tag} frame {frame_num} (index {dataset_idx})")

            track_local_idx = None
            global_q_idx = None
            if track_vis_enabled:
                track_local_idx = _find_track_local_index(tracks[0], vis_track_id)
                if track_local_idx is not None:
                    global_q_idx = _global_query_index(inner_model.n_det_queries, track_local_idx)
                    print(
                        f"[Info] frame {frame_num}: track id {vis_track_id} "
                        f"local={track_local_idx}, global_q={global_q_idx}"
                    )
                else:
                    print(f"[Warn] frame {frame_num}: track id {vis_track_id} not in input tracks, skip track vis")

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

            figure = _build_figure_context(
                enabled=should_save or track_vis_enabled,
                seq=seq,
                frame_num=frame_num,
                map_out_dir=map_out_dir,
                track_vis_out_dir=track_vis_out_dir,
                save_maps=save_maps if should_save else [],
                vis_track_id=vis_track_id,
                track_local_idx=track_local_idx,
                global_q_idx=global_q_idx,
                vis_track_spectral=vis_track_spectral,
                vis_cross_attn=vis_cross_attn,
                skip_existing=skip_existing_images,
                vis_max_size=vis_max_size,
                frame_pad_mask=frame.masks[0].detach() if should_save else None,
                spectral_timeline=spectral_timeline if track_vis_enabled else None,
            )

            forward_kwargs = dict(frame=frame, tracks=tracks, debug=need_debug)
            if figure is not None:
                forward_kwargs["figure"] = figure
            if gmc is not None:
                forward_kwargs["gmc"] = gmc
            res = model(**forward_kwargs)

            if se_accumulators and se_hook is not None:
                se_raw = se_hook.pop()
                if se_raw is not None:
                    se = _prepare_se_tensor(
                        se_raw,
                        stem_returns_gate=se_hook.stem_returns_gate,
                        apply_sigmoid=se_apply_sigmoid,
                    )
                    valid_mask = _downsample_valid_mask(frame.masks[0], se.shape[-2:])
                    for acc in se_accumulators:
                        acc.update(se, valid_mask=valid_mask)

            previous_tracks, new_tracks = tracker.update(model_outputs=res, tracks=tracks)
            tracks = inner_model.postprocess_single_frame(previous_tracks, new_tracks, None)

            if frame_num in preview_frames:
                out_path = _save_track_preview_frame(
                    ori_image=ori_image,
                    tracks=tracks[0],
                    frame_num=frame_num,
                    seq=seq,
                    out_dir=track_preview_out_dir,
                    rect_bbox=rect_bbox,
                    track_score_thresh=tracker.track_score_thresh,
                )
                print(f"[Info] Saved track ID preview: {out_path}")

            if should_save and isinstance(res, dict):
                saved_meta.append(_build_frame_meta(frame_num, res))

            del res, frame, image

    if track_vis_enabled and vis_track_spectral and track_vis_out_dir and vis_track_id is not None:
        save_track_spectral_timeline(
            timeline=spectral_timeline,
            figure=FigureContext(
                enabled=True,
                seq=seq,
                track_id=vis_track_id,
                track_out_dir=track_vis_out_dir,
                skip_existing=skip_existing_images,
                vis_max_size=vis_max_size,
            ),
        )

    return saved_meta


def main():
    parser = argparse.ArgumentParser(
        description="Load model_20260511_figure, run sequential inference, save debug figures."
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
    parser.add_argument("--seq", type=str, default="data39-1", help="Comma-separated sequence names.")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--start-frames", type=str, default=None)
    parser.add_argument("--end-frames", type=str, default="1")
    parser.add_argument(
        "--img-format",
        type=str,
        default="npy2jpg",
        choices=["npy2jpg", "npy"],
    )
    parser.add_argument("--npy2rgb", action="store_true")
    parser.add_argument(
        "--save-maps",
        type=str,
        default="pool_weights",
        help=f"Comma-separated SCEM map types; use none to skip SCEM. Available: {','.join(MAP_SAVER_REGISTRY.keys())}, all, none",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--save-last-n-frames",
        type=int,
        default=30,
        help="Only save SCEM maps for the last N frames. Use 0 to save all frames.",
    )
    parser.add_argument("--skip-existing-images", action="store_true")
    parser.add_argument("--vis-max-size", type=int, default=1200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dump-json", type=str, default="")
    parser.add_argument("--vis-track-id", type=int, default=None)
    parser.add_argument("--vis-track-spectral", action="store_true")
    parser.add_argument(
        "--vis-cross-attn",
        action="store_true",
        help="保存 decoder 各层 cross-attention 空间 attn 图（由 model_20260511_figure 内部捕获）。",
    )
    parser.add_argument(
        "--vis-track-preview",
        action="store_true",
        help="保存前 N 帧原图叠加全部 active track id，用于挑选 --vis-track-id。",
    )
    parser.add_argument(
        "--vis-track-preview-n",
        type=int,
        default=5,
        help="与 --vis-track-preview 联用，保存前 N 帧（默认 5）。",
    )
    parser.add_argument(
        "--analyze-se",
        action="store_true",
        help="统计 stem SE 在逐像素、8 通道上的 mean/var（跨所选序列的全部帧）。",
    )
    parser.add_argument(
        "--se-apply-sigmoid",
        action="store_true",
        help="对 sig_raw（conv3d_se_v4）做 sigmoid 后再统计；默认使用 raw SE。",
    )
    parser.add_argument(
        "--se-save-vis",
        action="store_true",
        help="与 --analyze-se 联用，保存 8 通道 mean/std 热力图。",
    )
    parser.add_argument(
        "--se-per-seq",
        action="store_true",
        help="与 --analyze-se 联用，除全局汇总外，每条序列单独保存统计结果。",
    )

    args = parser.parse_args()
    save_maps = _resolve_save_maps(args.save_maps)
    if args.vis_track_id is not None and not (args.vis_track_spectral or args.vis_cross_attn):
        raise ValueError("--vis-track-id 需要同时指定 --vis-track-spectral 和/或 --vis-cross-attn")

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

    config_root = _resolve_config_root(train_cfg_path, args.output_dir)
    heatmap_root = os.path.join(config_root, "heatmaps")
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
    train_config["MEMOTR_VERSION"] = "20260511_figure"
    print("[Info] Using MEMOTR_VERSION='20260511_figure'")
    model = build_model(config=train_config)
    model.to(torch.device(args.device))

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[Info] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    _init_track_instances_static(train_config, model)

    se_hook: Optional[SECaptureHook] = None
    global_se_acc: Optional[SEStatsAccumulator] = None
    se_stats_root = os.path.join(config_root, "se_stats")
    if args.analyze_se:
        se_hook = SECaptureHook(model)
        global_se_acc = SEStatsAccumulator()
        os.makedirs(se_stats_root, exist_ok=True)
        print(
            f"[Info] SE stats enabled: stem_returns_gate={se_hook.stem_returns_gate}, "
            f"apply_sigmoid={args.se_apply_sigmoid}, out={se_stats_root}"
        )

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

        track_vis_out_dir = None
        if args.vis_track_id is not None:
            track_vis_out_dir = _resolve_track_vis_out_dir(config_root, seq, args.vis_track_id)

        track_preview_out_dir = None
        if args.vis_track_preview:
            track_preview_out_dir = _resolve_track_preview_out_dir(config_root, seq)

        rect_bbox = is_rect_memotr_version(train_config.get("MEMOTR_VERSION", ""))

        seq_se_acc = SEStatsAccumulator() if args.analyze_se and args.se_per_seq else None
        se_accumulators: List[SEStatsAccumulator] = []
        if global_se_acc is not None:
            se_accumulators.append(global_se_acc)
        if seq_se_acc is not None:
            se_accumulators.append(seq_se_acc)

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
            vis_track_id=args.vis_track_id,
            vis_track_spectral=args.vis_track_spectral,
            vis_cross_attn=args.vis_cross_attn,
            track_vis_out_dir=track_vis_out_dir,
            vis_track_preview=args.vis_track_preview,
            vis_track_preview_n=args.vis_track_preview_n,
            track_preview_out_dir=track_preview_out_dir,
            rect_bbox=rect_bbox,
            se_hook=se_hook,
            se_accumulators=se_accumulators or None,
            se_apply_sigmoid=args.se_apply_sigmoid,
        )
        all_meta["per_seq"].append({"seq": seq, "frames": saved_frames})

        if seq_se_acc is not None:
            seq_out = os.path.join(se_stats_root, seq)
            paths = seq_se_acc.save(seq_out, prefix="se")
            print(f"[Info] SE stats for seq={seq}: {paths['meta']}")
            if args.se_save_vis:
                vis_path = save_se_stats_visualization(seq_se_acc.finalize(), seq_out, prefix="se")
                print(f"[Info] SE vis for seq={seq}: {vis_path}")

    if args.analyze_se and global_se_acc is not None:
        paths = global_se_acc.save(se_stats_root, prefix="se_all")
        print(f"[Info] Global SE stats ({len(seq_specs)} seq): {paths['meta']}")
        if args.se_save_vis:
            vis_path = save_se_stats_visualization(global_se_acc.finalize(), se_stats_root, prefix="se_all")
            print(f"[Info] Global SE vis: {vis_path}")

    if se_hook is not None:
        se_hook.close()

    if args.dump_json:
        try:
            with open(args.dump_json, "w") as f:
                json.dump(all_meta, f, indent=2)
            print(f"[Info] Dumped metadata to {args.dump_json}")
        except Exception as e:
            print(f"[Warn] Failed to dump json: {e}")

    print(f"\n[Done] SCEM heatmaps under: {heatmap_root}")
    if args.analyze_se:
        print(f"[Done] SE per-pixel stats under: {se_stats_root}")
    if args.vis_track_id is not None:
        print(f"[Done] Track vis under: {os.path.join(config_root, 'track')}")


if __name__ == "__main__":
    main()
