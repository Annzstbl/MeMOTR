#!/usr/bin/env python3
"""
可视化 GMC（sparseOptFlow）将上一帧 box 对齐到当前帧的效果。

每个序列只跑前 N 帧（默认 20），保存到：
  <output_dir>/<seq_name>/frame_0001.jpg ...

模式:
  - 默认: 上一帧 GT 经 GMC 到当前帧（绿色=当前帧 GT，橙色=GMC 后的上一帧 GT）
  - --use-model: 上一帧 track ref_pts 经 GMC（与 model_20260511_gmc.forward 一致）
      不画 GT；第 1 帧仅画模型 track 输出，后续帧画 GMC/ref/预测
      蓝色=历史 track ref_pts（GMC 前）  橙色=历史 ref_pts+GMC
      紫色=历史 track 当前帧预测  绿色=本帧 newborn（标签 H/N + id + 真实得分）

用法:
  cd MeMOTR
  # GT 模式
  python scripts/vis_gmc_box_warp.py --data-root ... --output-dir ... --num-frames 20

  # 模型 ref_pts 模式
  python scripts/vis_gmc_box_warp.py --use-model \\
    --config-path configs_hsmot_spectral_embed_252/20260511-1-gmc-submit.yaml \\
    --data-root /data/users/litianhao01/hsmot/data --output-dir .../gmc_vis_model \\
    --num-frames 20 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MEMOTR_ROOT = os.path.dirname(_SCRIPT_DIR)
if _MEMOTR_ROOT not in sys.path:
    sys.path.insert(0, _MEMOTR_ROOT)

from data.seq_dataset import SeqDataset
from hsmot.datasets.pipelines.channel import rotate_norm_boxes_to_boxes
from hsmot.mmlab.hs_mmrotate import obb2poly_np
from models.runtime_tracker import RuntimeTracker
from models.utils import get_model, load_checkpoint, logits_to_scores
from structures.track_instances import TrackInstances
from submit_engine import (
    _init_track_instances_static,
    _overlay_submit_train_config,
    list_submit_sequences,
    resolve_submit_checkpoint_and_output,
    resolve_submit_dataset_root,
    resolve_submit_split_dir,
)
from utils.GMC import compute_gmc_sequence
from utils.batch_vis_result import draw_rotated_bbox
from utils.nested_tensor import tensor_list_to_nested_tensor_with_shared_shapes
from utils.utils import load_train_config, load_yaml_with_inheritance


def load_mot_labels(label_file: str) -> dict[int, list[np.ndarray]]:
    """frame_idx (0-based) -> list of [x0..y3, track_id, cls]."""
    label_full: dict[int, list[np.ndarray]] = defaultdict(list)
    if not os.path.isfile(label_file):
        return label_full
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 13:
                continue
            t = int(float(parts[0])) - 1
            track_id = int(float(parts[1]))
            coords = list(map(float, parts[2:10]))
            cls = int(float(parts[11]))
            label_full[t].append(np.array([*coords, track_id, cls], dtype=np.float32))
    return label_full


def ori_to_bgr(ori_image: np.ndarray) -> np.ndarray:
    """8ch HSMOT ori -> BGR uint8 for display (与 submit 画图通道一致)."""
    if hasattr(ori_image, "detach"):
        ori_image = ori_image.detach().cpu().numpy()
    img = np.ascontiguousarray(ori_image)
    if img.ndim != 3:
        raise ValueError(f"Expect HWC ori image, got shape {img.shape}")
    c = img.shape[2]
    if c >= 8:
        vis = np.stack([img[:, :, 4], img[:, :, 2], img[:, :, 1]], axis=-1).astype(np.float32)
    elif c >= 3:
        vis = np.ascontiguousarray(img[:, :, :3], dtype=np.float32)
    else:
        vis = np.repeat(img[:, :, :1], 3, axis=2).astype(np.float32)
    vmin, vmax = vis.min(), vis.max()
    if vmax > vmin:
        vis = (vis - vmin) / (vmax - vmin) * 255.0
    else:
        vis = np.zeros_like(vis)
    return np.ascontiguousarray(vis.clip(0, 255).astype(np.uint8))


def warp_poly_xyxyxyxy(poly8: np.ndarray, gmc_matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly8[:8], dtype=np.float32).reshape(4, 2)
    pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
    new_pts = (gmc_matrix @ pts_h.T).T
    return new_pts.reshape(-1)


def ref_pts_to_polys(ref_pts: torch.Tensor, eff_hw: Tuple[int, int]) -> List[np.ndarray]:
    """ref_pts (inverse_sigmoid) -> 8 点多边形，有效区域像素坐标（与 submit / pred 一致）。"""
    if ref_pts.numel() == 0:
        return []
    norm = ref_pts.sigmoid().detach().cpu()
    boxes = rotate_norm_boxes_to_boxes(norm, eff_hw, version="le135")
    boxes_np = boxes.numpy().astype(np.float32)
    boxes_np = np.concatenate([boxes_np, np.ones((len(boxes_np), 1), dtype=np.float32)], axis=1)
    polys = obb2poly_np(boxes_np, version="le135")
    if polys.ndim == 1:
        polys = polys.reshape(1, -1)
    return [polys[i, :8] for i in range(polys.shape[0])]


def norm_boxes_to_polys(norm_boxes: torch.Tensor, eff_hw: Tuple[int, int]) -> List[np.ndarray]:
    if norm_boxes.numel() == 0:
        return []
    boxes = rotate_norm_boxes_to_boxes(norm_boxes.detach().cpu(), eff_hw, version="le135")
    boxes_np = boxes.numpy().astype(np.float32)
    boxes_np = np.concatenate([boxes_np, np.ones((len(boxes_np), 1), dtype=np.float32)], axis=1)
    polys = obb2poly_np(boxes_np, version="le135")
    if polys.ndim == 1:
        polys = polys.reshape(1, -1)
    return [polys[i, :8] for i in range(polys.shape[0])]


def track_instance_score(track: TrackInstances, idx: int) -> float:
    """与 RuntimeTracker 一致：取当前 label 上的 sigmoid 得分。"""
    lbl = int(track.labels[idx].item())
    row = track.scores[idx]
    if row.ndim == 0:
        return float(row.item())
    return float(row[lbl].item())


def draw_poly(
    img: np.ndarray,
    poly8: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    score: float,
    track_id: Optional[Union[int, str]] = None,
    role: str = "",
) -> None:
    x1, y1, x2, y2, x3, y3, x4, y4 = poly8[:8].tolist()
    if role and track_id is not None:
        label_id: Union[int, str] = f"{role}{track_id}"
    elif role:
        label_id = role
    else:
        label_id = track_id
    draw_rotated_bbox(
        img, label_id, x1, y1, x2, y2, x3, y3, x4, y4, score,
        thickness=thickness, font_scale=0.45, color=color,
    )


def ref_polys_with_meta(
    ref_pts: torch.Tensor,
    eff_hw: Tuple[int, int],
    track: TrackInstances,
    active: torch.Tensor,
) -> List[Tuple[np.ndarray, float, int]]:
    if not active.any():
        return []
    idxs = active.nonzero(as_tuple=False).flatten().tolist()
    polys = ref_pts_to_polys(ref_pts[active], eff_hw)
    items: List[Tuple[np.ndarray, float, int]] = []
    for poly, idx in zip(polys, idxs):
        if int(track.ids[idx].item()) < 0:
            continue
        items.append((poly, track_instance_score(track, idx), int(track.ids[idx].item())))
    return items


def track_boxes_to_draw_items(
    track: TrackInstances,
    eff_hw: Tuple[int, int],
    *,
    active_only: bool = True,
) -> List[Tuple[np.ndarray, float, int]]:
    items: List[Tuple[np.ndarray, float, int]] = []
    for i in range(len(track)):
        if active_only and int(track.ids[i].item()) < 0:
            continue
        polys = norm_boxes_to_polys(track.boxes[i : i + 1], eff_hw)
        if not polys:
            continue
        items.append((polys[0], track_instance_score(track, i), int(track.ids[i].item())))
    return items


def forward_track_pred_items(
    res: dict,
    n_dets: int,
    track: TrackInstances,
    eff_hw: Tuple[int, int],
) -> List[Tuple[np.ndarray, float, int]]:
    """当前帧 forward 对已有 track query 的预测（与 tracks 槽位一一对应）。"""
    track_logits = res["pred_logits"][0, n_dets:]
    scores = logits_to_scores(track_logits)
    boxes = res["pred_bboxes"][0, n_dets:]
    items: List[Tuple[np.ndarray, float, int]] = []
    n_slots = min(len(track), boxes.shape[0])
    for i in range(n_slots):
        if int(track.ids[i].item()) < 0:
            continue
        lbl = int(track.labels[i].item())
        sc = float(scores[i, lbl].item())
        polys = norm_boxes_to_polys(boxes[i : i + 1], eff_hw)
        if not polys:
            continue
        items.append((polys[0], sc, int(track.ids[i].item())))
    return items


def load_frame_pair(dataset: SeqDataset, frame_idx: int):
    item = dataset[frame_idx]
    if len(item) == 2:
        (image_tensor, ori_image), _info = item
    else:
        raise ValueError(f"Unexpected dataset item length: {len(item)}")
    if isinstance(image_tensor, tuple):
        image_tensor, ori_image = image_tensor
    return image_tensor, ori_image


def effective_hwc_from_ori(ori_image: np.ndarray) -> Tuple[int, int, int]:
    """与 submit_engine 一致：有效区域 (H, W, C)，不含右侧/下侧 pad。"""
    if hasattr(ori_image, "shape") and len(ori_image.shape) == 4:
        return int(ori_image.shape[1]), int(ori_image.shape[2]), int(ori_image.shape[3])
    return int(ori_image.shape[0]), int(ori_image.shape[1]), int(ori_image.shape[2])


def build_nested_frame(image_tensor: torch.Tensor, ori_image: np.ndarray, device: torch.device):
    effective_img_shape = effective_hwc_from_ori(ori_image)
    padded_img_shape = (
        int(image_tensor.shape[1]),
        int(image_tensor.shape[2]),
        int(image_tensor.shape[0]),
    )
    frame = tensor_list_to_nested_tensor_with_shared_shapes(
        [image_tensor],
        effective_img_shape=effective_img_shape,
        padded_img_shape=padded_img_shape,
    ).to(device)
    eff_hw = (effective_img_shape[0], effective_img_shape[1])
    return frame, eff_hw


def draw_canvas_header(canvas, title: str, subtitle: str) -> None:
    cv2.putText(
        canvas, title, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, subtitle, (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA,
    )


def process_sequence_gt(
    seq_name: str,
    seq_dir: str,
    mot_dir: str,
    output_dir: str,
    num_frames: int,
    dataset_type: str,
    gmc_method: str,
    gmc_downscale: int,
) -> int:
    label_file = os.path.join(mot_dir, f"{seq_name}.txt")
    labels = load_mot_labels(label_file)
    dataset = SeqDataset(seq_dir=seq_dir, dataset_type=dataset_type)
    n = min(num_frames, len(dataset))
    if n <= 0:
        return 0

    seq_out = os.path.join(output_dir, seq_name)
    os.makedirs(seq_out, exist_ok=True)

    prev_tensor = None
    prev_polys: list[np.ndarray] = []
    saved = 0

    for frame_idx in range(n):
        image_tensor, ori_image = load_frame_pair(dataset, frame_idx)
        canvas = ori_to_bgr(ori_image)
        gmc = np.eye(2, 3, dtype=np.float32)

        if prev_tensor is not None:
            gmc = compute_gmc_sequence(
                images=[prev_tensor, image_tensor],
                method=gmc_method,
                downscale=gmc_downscale,
            )[-1]

        for row in labels.get(frame_idx, []):
            draw_poly(
                canvas, row[:8], color=(0, 220, 0), thickness=2,
                score=0.0, track_id=int(row[8]), role="GT",
            )

        for row in prev_polys:
            warped = warp_poly_xyxyxyxy(row[:8], gmc)
            draw_poly(
                canvas, warped, color=(0, 140, 255), thickness=2,
                score=0.0, track_id=int(row[8]), role="GT",
            )

        if frame_idx == 0:
            sub = "green=GT"
        else:
            sub = "green=GT  orange=prev GT+GMC"
        draw_canvas_header(canvas, f"{seq_name}  frame={frame_idx + 1}  [GT]", sub)

        cv2.imwrite(os.path.join(seq_out, f"frame_{frame_idx + 1:04d}.jpg"), canvas)
        saved += 1
        prev_tensor = image_tensor.detach().clone()
        prev_polys = [r.copy() for r in labels.get(frame_idx, [])]

    return saved


@torch.no_grad()
def process_sequence_model(
    seq_name: str,
    seq_dir: str,
    output_dir: str,
    num_frames: int,
    dataset_type: str,
    gmc_method: str,
    gmc_downscale: int,
    model: torch.nn.Module,
    tracker: RuntimeTracker,
    device: torch.device,
) -> int:
    inner = get_model(model)
    dataset = SeqDataset(seq_dir=seq_dir, dataset_type=dataset_type)
    n = min(num_frames, len(dataset))
    if n <= 0:
        return 0

    seq_out = os.path.join(output_dir, seq_name)
    os.makedirs(seq_out, exist_ok=True)

    tracks = [
        TrackInstances(
            hidden_dim=inner.hidden_dim,
            num_classes=inner.num_classes,
        ).to(device)
    ]
    prev_tensor = None
    saved = 0

    # BGR: 历史 ref 前/后、历史 pred、newborn
    COLOR_REF_BEFORE = (255, 180, 0)
    COLOR_REF_AFTER = (0, 140, 255)
    COLOR_HIST_PRED = (220, 0, 220)
    COLOR_NEWBORN = (0, 220, 0)

    for frame_idx in range(n):
        image_tensor, ori_image = load_frame_pair(dataset, frame_idx)
        frame, eff_hw = build_nested_frame(image_tensor, ori_image, device)

        gmc_np = np.eye(2, 3, dtype=np.float32)
        if prev_tensor is not None:
            gmc_np = compute_gmc_sequence(
                images=[prev_tensor, image_tensor],
                method=gmc_method,
                downscale=gmc_downscale,
            )[-1]
        gmc_t = torch.tensor(gmc_np, dtype=torch.float32, device=device).unsqueeze(0)

        ref_before_items: List[Tuple[np.ndarray, float, int]] = []
        ref_after_items: List[Tuple[np.ndarray, float, int]] = []
        hist_pred_items: List[Tuple[np.ndarray, float, int]] = []
        if len(tracks[0]) > 0:
            active = tracks[0].ids >= 0
        else:
            active = torch.zeros(0, dtype=torch.bool, device=device)

        if frame_idx > 0 and active.any():
            ref_before_items = ref_polys_with_meta(tracks[0].ref_pts, eff_hw, tracks[0], active)

        res = inner(frame=frame, tracks=tracks, gmc=gmc_t)
        n_dets = len(res["det_query_embed"])

        if frame_idx > 0 and active.any():
            ref_after_items = ref_polys_with_meta(tracks[0].ref_pts, eff_hw, tracks[0], active)
            hist_pred_items = forward_track_pred_items(res, n_dets, tracks[0], eff_hw)

        previous_tracks, new_tracks = tracker.update(model_outputs=res, tracks=tracks)
        tracks = inner.postprocess_single_frame(previous_tracks, new_tracks, None)

        canvas = ori_to_bgr(ori_image)

        if frame_idx == 0:
            for poly, sc, tid in track_boxes_to_draw_items(tracks[0], eff_hw):
                draw_poly(
                    canvas, poly, COLOR_NEWBORN, thickness=2, score=sc,
                    track_id=tid, role="N",
                )
            sub = "green N|id|score=model tracks (frame1, no GT)"
        else:
            for poly, sc, tid in ref_before_items:
                draw_poly(
                    canvas, poly, COLOR_REF_BEFORE, thickness=2, score=sc,
                    track_id=tid, role="Rb",
                )
            for poly, sc, tid in ref_after_items:
                draw_poly(
                    canvas, poly, COLOR_REF_AFTER, thickness=2, score=sc,
                    track_id=tid, role="Ra",
                )
            for poly, sc, tid in hist_pred_items:
                draw_poly(
                    canvas, poly, COLOR_HIST_PRED, thickness=1, score=sc,
                    track_id=tid, role="H",
                )
            for poly, sc, tid in track_boxes_to_draw_items(new_tracks[0], eff_hw, active_only=False):
                draw_poly(
                    canvas, poly, COLOR_NEWBORN, thickness=2, score=sc,
                    track_id=tid, role="N",
                )
            sub = "Rb/Ra=hist ref  H=hist pred  N=newborn (real score, no GT)"

        draw_canvas_header(canvas, f"{seq_name}  frame={frame_idx + 1}  [model]", sub)
        cv2.imwrite(os.path.join(seq_out, f"frame_{frame_idx + 1:04d}.jpg"), canvas)
        saved += 1
        prev_tensor = frame.tensors[0].detach().clone()

    return saved


def build_model_and_tracker(
    config_path: str,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> tuple[torch.nn.Module, RuntimeTracker, dict]:
    submit_cfg = load_yaml_with_inheritance(config_path)
    checkpoint_root, _output_root = resolve_submit_checkpoint_and_output(submit_cfg)
    if checkpoint_path is None:
        ckpt_name = submit_cfg.get("SUBMIT_MODEL", "last.pth")
        checkpoint_path = os.path.join(checkpoint_root, ckpt_name)

    train_config = load_train_config(os.path.join(checkpoint_root, "train/config.yaml"))
    _overlay_submit_train_config(train_config=train_config, config=submit_cfg)

    train_config["DEVICE"] = str(device)
    if train_config.get("AVAILABLE_GPUS"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(train_config["AVAILABLE_GPUS"]).split(",")[0]

    from models import build_model as build_model_fn

    model = build_model_fn(config=train_config)
    load_checkpoint(model=model, path=checkpoint_path)
    model = model.to(device)
    model.eval()
    _init_track_instances_static(train_config, model)

    tracker = RuntimeTracker(
        det_score_thresh=submit_cfg.get("DET_SCORE_THRESH", 0.5),
        track_score_thresh=submit_cfg.get("TRACK_SCORE_THRESH", 0.5),
        miss_tolerance=submit_cfg.get("MISS_TOLERANCE", 30),
        use_motion=submit_cfg.get("USE_MOTION", False),
        motion_min_length=submit_cfg.get("MOTION_MIN_LENGTH", 3),
        motion_max_length=submit_cfg.get("MOTION_MAX_LENGTH", 5),
        use_dab=train_config.get("USE_DAB", True),
        decoder_spectral=train_config.get("DECODER_SPECTRAL", True),
    )
    return model, tracker, train_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize GMC-warped boxes (first N frames per seq).")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default="hsmot_8ch")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--dataset-type", default="3JPG")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--seq", nargs="*", default=None, help="只处理指定序列名，默认全部")
    parser.add_argument("--gmc-method", default="sparseOptFlow")
    parser.add_argument("--gmc-downscale", type=int, default=1)
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="用 model_20260511_gmc 推理 track，可视化 ref_pts 的 GMC 前后",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="--use-model 时必填，如 configs_hsmot_spectral_embed_252/20260511-1-gmc-submit.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="覆盖配置中的 SUBMIT_MODEL 路径",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.use_model and not args.config_path:
        parser.error("--use-model 需要同时指定 --config-path")

    data_split_dir = resolve_submit_split_dir(
        data_root=args.data_root,
        dataset_name=args.dataset,
        dataset_split=args.split,
        dataset_version=args.dataset_version,
        dataset_type=args.dataset_type,
    )
    dataset_root = resolve_submit_dataset_root(
        data_root=args.data_root,
        dataset_name=args.dataset,
        dataset_split=args.split,
        dataset_version=args.dataset_version,
        dataset_type=args.dataset_type,
    )
    mot_dir = os.path.join(dataset_root, args.split, "mot")
    if not os.path.isdir(mot_dir):
        mot_dir = os.path.join(os.path.dirname(data_split_dir), "mot")

    seq_names = args.seq or list_submit_sequences(data_split_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    model = tracker = None
    if args.use_model:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        config_path = args.config_path
        if not os.path.isabs(config_path):
            config_path = os.path.join(_MEMOTR_ROOT, config_path)
        model, tracker, _ = build_model_and_tracker(
            config_path=config_path,
            checkpoint_path=args.checkpoint,
            device=device,
        )
        print(f"[model] loaded from config={config_path}, device={device}")

    total_saved = 0
    desc = "GMC vis (model)" if args.use_model else "GMC vis (GT)"
    for seq_name in tqdm(seq_names, desc=desc):
        seq_dir = os.path.join(data_split_dir, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"[skip] missing seq dir: {seq_dir}")
            continue
        if args.use_model:
            n_saved = process_sequence_model(
                seq_name=seq_name,
                seq_dir=seq_dir,
                output_dir=args.output_dir,
                num_frames=args.num_frames,
                dataset_type=args.dataset_type,
                gmc_method=args.gmc_method,
                gmc_downscale=args.gmc_downscale,
                model=model,
                tracker=tracker,
                device=torch.device(args.device if torch.cuda.is_available() else "cpu"),
            )
        else:
            n_saved = process_sequence_gt(
                seq_name=seq_name,
                seq_dir=seq_dir,
                mot_dir=mot_dir,
                output_dir=args.output_dir,
                num_frames=args.num_frames,
                dataset_type=args.dataset_type,
                gmc_method=args.gmc_method,
                gmc_downscale=args.gmc_downscale,
            )
        total_saved += n_saved
        print(f"  {seq_name}: saved {n_saved} frames -> {os.path.join(args.output_dir, seq_name)}")

    print(f"Done. total frames saved: {total_saved} under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
