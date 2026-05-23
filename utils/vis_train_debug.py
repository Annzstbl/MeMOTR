"""训练时将 transform 后的 batch 保存到 workdir/debug，用于核对图像与标注。"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

VT_TINY_CATEGORY_NAMES = [
    "ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane",
]

VT_TINY_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255, 127.5]
VT_TINY_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255, 127.5]

HSMOT_8CH_MEAN = [
    0.27358221, 0.28804452, 0.28133921, 0.26906377,
    0.28309119, 0.26928305, 0.28372527, 0.27149373,
]
HSMOT_8CH_STD = [
    0.19756629, 0.17432339, 0.16413284, 0.17581682,
    0.18366176, 0.1536845, 0.15964683, 0.16557951,
]


def _denorm_stats(config: dict) -> tuple[np.ndarray, np.ndarray]:
    dataset = config.get("DATASET", "")
    input_channels = int(config.get("INPUT_CHANNELS", 3))
    if dataset in ("vt_tiny_mot", "VT-Tiny-MOT") or input_channels == 4:
        mean = np.array(VT_TINY_MEAN[:input_channels], dtype=np.float32)
        std = np.array(VT_TINY_STD[:input_channels], dtype=np.float32)
    elif input_channels == 8:
        mean = np.array([v * 255 for v in HSMOT_8CH_MEAN], dtype=np.float32)
        std = np.array([v * 255 for v in HSMOT_8CH_STD], dtype=np.float32)
    else:
        mean = np.full(input_channels, 127.5, dtype=np.float32)
        std = np.full(input_channels, 127.5, dtype=np.float32)
    return mean, std


def tensor_to_vis_img(img_tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    c = img.shape[2]
    img = img * std[:c] + mean[:c]
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_boxes_xyxy(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    obj_ids: torch.Tensor,
    category_names: list[str] | None = None,
) -> np.ndarray:
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] >= 3:
        vis = vis[:, :, :3].copy()
    elif vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if boxes.numel() == 0:
        return vis

    boxes_np = boxes.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    ids_np = obj_ids.detach().cpu().numpy()

    for box, label, obj_id in zip(boxes_np, labels_np, ids_np):
        x1, y1, x2, y2 = map(int, box[:4])
        color = (
            int(37 * (int(obj_id) % 7 + 1) % 255),
            int(17 * (int(label) + 3) % 255),
            int(29 * (int(obj_id) + 5) % 255),
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cls_name = category_names[int(label)] if category_names and 0 <= int(label) < len(category_names) else str(int(label))
        text = f"id={int(obj_id)} {cls_name}"
        cv2.putText(
            vis, text, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return vis


def _save_frame(
    img_tensor: torch.Tensor,
    target: dict,
    save_stem: Path,
    mean: np.ndarray,
    std: np.ndarray,
    category_names: list[str] | None,
):
    img_hwc = tensor_to_vis_img(img_tensor, mean, std)
    boxes = target.get("boxes")
    labels = target.get("labels")
    obj_ids = target.get("obj_ids")
    if boxes is None or labels is None or obj_ids is None:
        cv2.imwrite(str(save_stem.with_suffix(".jpg")), cv2.cvtColor(img_hwc[:, :, :3], cv2.COLOR_RGB2BGR))
        return

    rgb = _draw_boxes_xyxy(img_hwc[:, :, :3], boxes, labels, obj_ids, category_names)
    cv2.imwrite(str(save_stem.with_name(save_stem.name + "_rgb.jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if img_hwc.shape[2] >= 4:
        ir = img_hwc[:, :, 3]
        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        ir_vis = _draw_boxes_xyxy(ir_bgr, boxes, labels, obj_ids, category_names)
        cv2.imwrite(str(save_stem.with_name(save_stem.name + "_ir.jpg")), ir_vis)


def save_train_batch_debug(
    batch: dict,
    save_dir: str | Path,
    epoch: int,
    iter_idx: int,
    config: dict,
):
    """保存一个 training batch（transform 后）到 ``save_dir/epoch_{e}/iter_{i}/``。"""
    save_root = Path(save_dir) / f"epoch_{epoch:03d}" / f"iter_{iter_idx:04d}"
    save_root.mkdir(parents=True, exist_ok=True)

    mean, std = _denorm_stats(config)
    category_names = VT_TINY_CATEGORY_NAMES if config.get("DATASET") in ("vt_tiny_mot", "VT-Tiny-MOT") else None

    batch_size = len(batch["imgs"])
    clip_len = len(batch["imgs"][0])
    info_lines = [
        f"epoch={epoch}",
        f"iter={iter_idx}",
        f"batch_size={batch_size}",
        f"clip_len={clip_len}",
        f"dataset={config.get('DATASET')}",
    ]

    for b in range(batch_size):
        for f in range(clip_len):
            img = batch["imgs"][b][f]
            target = batch["infos"][b][f]
            meta = batch["img_metas"][b][f] if "img_metas" in batch else {}

            n_box = int(target["boxes"].shape[0]) if target.get("boxes") is not None else 0
            img_shape = meta.get("img_shape")
            pad_shape = meta.get("pad_shape")
            info_lines.append(
                f"  b{b}_f{f}: img={tuple(img.shape)} boxes={n_box} "
                f"img_shape={img_shape.tolist() if torch.is_tensor(img_shape) else img_shape} "
                f"pad_shape={pad_shape.tolist() if torch.is_tensor(pad_shape) else pad_shape}"
            )
            if n_box > 0:
                info_lines.append(f"    labels={target['labels'].tolist()}")
                info_lines.append(f"    obj_ids={target['obj_ids'].tolist()}")

            _save_frame(
                img_tensor=img,
                target=target,
                save_stem=save_root / f"b{b:02d}_f{f:02d}",
                mean=mean,
                std=std,
                category_names=category_names,
            )

    with open(save_root / "info.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(info_lines) + "\n")
