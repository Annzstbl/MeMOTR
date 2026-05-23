#!/usr/bin/env python3
"""可视化 VTTinyMOT 数据集采样结果。

用法 (在 MeMOTR 目录下):
    conda activate hsmot
    python scripts/vis_vt_tiny_mot.py
    python scripts/vis_vt_tiny_mot.py --ann-mode plain --split test --num-samples 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ME_MOTR_ROOT = Path(__file__).resolve().parents[1]
if str(ME_MOTR_ROOT) not in sys.path:
    sys.path.insert(0, str(ME_MOTR_ROOT))

from data.vt_tiny_mot import VTTinyMOT, transforms_for_train

CATEGORY_NAMES = [
    "ship",
    "car",
    "cyclist",
    "pedestrian",
    "bus",
    "drone",
    "plane",
]

DEFAULT_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255, 127.5]
DEFAULT_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255, 127.5]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VTTinyMOT dataset samples")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(ME_MOTR_ROOT.parent / "data"),
        help="数据集根目录 (包含 VT-Tiny-MOT)",
    )
    parser.add_argument("--dataset-dir", type=str, default="VT-Tiny-MOT")
    parser.add_argument("--ann-mode", type=str, default="plain", choices=["00", "01", "plain"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=3, help="可视化 clip 数量")
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None, help="指定 dataset 索引")
    parser.add_argument("--output-dir", type=str, default=str(ME_MOTR_ROOT / "debug_vt_tiny_mot"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize-w", type=int, nargs=2, default=[480, 640], metavar=("MIN", "MAX"))
    parser.add_argument("--resize-h", type=int, nargs=2, default=[384, 512], metavar=("MIN", "MAX"))
    parser.add_argument("--crop-w", type=int, nargs=2, default=[416, 640], metavar=("MIN", "MAX"))
    parser.add_argument("--crop-h", type=int, nargs=2, default=[448, 512], metavar=("MIN", "MAX"))
    parser.add_argument("--flip-ratio", type=float, default=0.5)
    parser.add_argument("--sample-length", type=int, default=2)
    return parser.parse_args()


def build_dataset(args) -> VTTinyMOT:
    config = {
        "DATA_ROOT": args.data_root,
        "DATASET_DIR": args.dataset_dir,
        "ANN_MODE": args.ann_mode,
        "SAMPLE_STEPS": [999],
        "SAMPLE_INTERVALS": [4],
        "SAMPLE_MODES": ["random_interval"],
        "SAMPLE_LENGTHS": [args.sample_length],
        "NPY2RGB": False,
        "TRANSFORMS_CONFIG": {
            "RESIZE": {"W": list(args.resize_w), "H": list(args.resize_h)},
            "CROP_SIZE": {"W": list(args.crop_w), "H": list(args.crop_h)},
            "FLIP_RATIO": args.flip_ratio,
        },
        "DECODER_SPECTRAL_REFINE": False,
    }
    transform = transforms_for_train(transform_config=config["TRANSFORMS_CONFIG"])
    return VTTinyMOT(config=config, split=args.split, transform=transform, logger=None)


def tensor_to_vis_img(img_tensor: torch.Tensor, mean=None, std=None) -> np.ndarray:
    """(C,H,W) tensor -> uint8 HWC, 反归一化。"""
    mean = np.array(mean if mean is not None else DEFAULT_MEAN, dtype=np.float32)
    std = np.array(std if std is not None else DEFAULT_STD, dtype=np.float32)

    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    c = img.shape[2]
    img = img * std[:c] + mean[:c]
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_boxes_xyxy(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    obj_ids: torch.Tensor,
) -> np.ndarray:
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] == 4:
        vis = vis[:, :, :3].copy()
    elif vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    boxes_np = boxes.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    ids_np = obj_ids.detach().cpu().numpy()

    for box, label, obj_id in zip(boxes_np, labels_np, ids_np):
        x1, y1, x2, y2 = map(int, box)
        color = (
            int(37 * (int(obj_id) % 7 + 1) % 255),
            int(17 * (int(label) + 3) % 255),
            int(29 * (int(obj_id) + 5) % 255),
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cls_name = CATEGORY_NAMES[int(label)] if 0 <= int(label) < len(CATEGORY_NAMES) else str(int(label))
        text = f"id={int(obj_id)} {cls_name}"
        cv2.putText(
            vis, text, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return vis


def save_frame_views(
    img_tensor: torch.Tensor,
    target: dict,
    save_stem: Path,
):
    img_hwc = tensor_to_vis_img(img_tensor)
    rgb = draw_boxes_xyxy(
        img_hwc[:, :, :3], target["boxes"], target["labels"], target["obj_ids"],
    )
    cv2.imwrite(str(save_stem.with_name(save_stem.name + "_rgb.jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if img_hwc.shape[2] >= 4:
        ir = img_hwc[:, :, 3]
        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        ir_vis = draw_boxes_xyxy(
            ir_bgr, target["boxes"], target["labels"], target["obj_ids"],
        )
        cv2.imwrite(str(save_stem.with_name(save_stem.name + "_ir.jpg")), ir_vis)


def visualize_sample(dataset: VTTinyMOT, sample_idx: int, output_dir: Path):
    vid, begin_frame = dataset.sample_begin_frames[sample_idx]
    item = dataset[sample_idx]
    images = item["images"]
    targets = item["targets"]

    clip_dir = output_dir / f"idx{sample_idx:05d}_{vid}_f{begin_frame}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    info_lines = [
        f"sample_idx={sample_idx}",
        f"video={vid}",
        f"begin_frame={begin_frame}",
        f"clip_len={len(images)}",
    ]
    for fi, (img, target) in enumerate(zip(images, targets)):
        n_box = int(target["boxes"].shape[0])
        info_lines.append(
            f"  frame[{fi}]: img={tuple(img.shape)} boxes={n_box} "
            f"labels={target['labels'].tolist() if n_box else []}"
        )
        save_frame_views(
            img_tensor=img,
            target=target,
            save_stem=clip_dir / f"frame{fi:02d}",
        )

    with open(clip_dir / "info.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")

    print(f"[saved] {clip_dir}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    print(
        f"Dataset: split={args.split}, ann_mode={args.ann_mode}, "
        f"videos={len(dataset.vid_idx)}, samples={len(dataset)}"
    )

    if args.sample_indices:
        candidate_indices = [int(i) for i in args.sample_indices]
    else:
        rng = np.random.default_rng(args.seed)
        candidate_indices = rng.permutation(len(dataset)).tolist()

    saved = 0
    for idx in candidate_indices:
        if not args.sample_indices and saved >= args.num_samples:
            break
        try:
            visualize_sample(dataset, int(idx), output_dir)
            saved += 1
        except (AssertionError, FileNotFoundError, KeyError) as exc:
            print(f"[skip] sample_idx={idx}: {exc}")

    print(f"Done. saved={saved}, output={output_dir}")


if __name__ == "__main__":
    main()
