"""VT-Tiny-MOT 双波段 MOT 数据集。

- 图像：每帧同时读取 RGB(/00/) 与 IR(/01/)，拼接为 4 通道 (H, W, 4)
- 标注：COCO-MOT JSON，可通过 ANN_MODE 选择 00 / 01 / plain
- 框类型：水平正框 (xyxy)，预处理走 hsmot MotH* pipeline
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from math import floor
from random import randint

import numpy as np
import os.path as osp

from hsmot.datasets.pipelines.channel import RectMotToMemotr
from hsmot.datasets.pipelines.compose import MotCompose
from hsmot.datasets.pipelines.formatting import MotCollect, MotDefaultFormatBundle
from hsmot.datasets.pipelines.loading import MotLoadAnnotations, MotLoadRgbIrImageFromJPG
from hsmot.datasets.pipelines.transforms import (
    MotHRandomCrop,
    MotHRandomFlip,
    MotHResize,
    MotNormalize,
    MotPad,
)

from .mot import MOTDataset
from .utils import resolve_stage_scalar

ANN_FILE_TEMPLATES = {
    "00": "instances_00_{split}2017.json",
    "01": "instances_01_{split}2017.json",
    "plain": "instances_{split}2017.json",
}


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y + h]


def parse_scene_from_file_name(file_name: str) -> str:
    return file_name.split("/")[0]


def parse_frame_id(img: dict) -> int:
    if "frame_id" in img:
        return int(img["frame_id"])
    if "mot_frame_id" in img:
        return int(img["mot_frame_id"])
    return int(osp.splitext(img["file_name"].split("/")[-1])[0])


def parse_file_frame_id(img: dict) -> int:
    """从 file_name 解析磁盘上的帧编号（如 00445.jpg -> 445）。"""
    return int(osp.splitext(img["file_name"].split("/")[-1])[0])


def load_coco_annotations(
    ann_path: str,
    ann_mode: str,
    ir_ann_path: str | None = None,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, dict[int, int]]]:
    """解析 COCO-MOT JSON。

    返回:
        labels_full[scene][mot_frame_id]: 该 MOT 帧上的 GT 列表
        frame_file_ids[scene][mot_frame_id]: MOT 帧序号 -> 磁盘文件名编号
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    ir_images_by_id: dict[int, dict] = {}
    if ann_mode == "plain":
        if ir_ann_path is None:
            raise ValueError("plain 模式需要提供 ir_ann_path (instances_01_*.json)")
        with open(ir_ann_path, "r", encoding="utf-8") as f:
            ir_data = json.load(f)
        ir_images_by_id = {img["id"]: img for img in ir_data["images"]}

    labels_full: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    frame_file_ids: dict[str, dict[int, int]] = defaultdict(dict)
    skipped = 0

    for img in data["images"]:
        if "/00/" not in img["file_name"]:
            continue
        scene = parse_scene_from_file_name(img["file_name"])
        mot_frame_id = parse_frame_id(img)
        frame_file_ids[scene][mot_frame_id] = parse_file_frame_id(img)

    for ann in data["annotations"]:
        img = images_by_id.get(ann["image_id"])
        if img is None:
            if ann_mode == "plain" and ann.get("type") == 2:
                img = ir_images_by_id.get(ann["image_id"])
            if img is None:
                skipped += 1
                continue

        scene = parse_scene_from_file_name(img["file_name"])
        frame_id = parse_frame_id(img)
        x, y, w, h = ann["bbox"]
        labels_full[scene][frame_id].append(
            np.array([x, y, w, h, ann["track_id"], ann["category_id"]], dtype=np.float32)
        )

    if skipped:
        print(f"[VT-Tiny-MOT] skipped {skipped} annotations with unresolved image_id in {ann_path}")

    return labels_full, frame_file_ids


class VTTinyMOT(MOTDataset):
    def __init__(self, config: dict, split: str, transform, logger=None, dataset_version=None):
        super().__init__(config=config, split=split, transform=transform)

        self.config = config
        self.transform = transform
        self.dataset_version = (
            dataset_version if dataset_version is not None else config.get("DATASET_VERSION")
        )
        assert split in ("train", "test"), f"Split {split} is not supported!"

        self.ann_mode = config.get("ANN_MODE", "plain").lower()
        assert self.ann_mode in ANN_FILE_TEMPLATES, (
            f"ANN_MODE={self.ann_mode} is not supported, choose from {list(ANN_FILE_TEMPLATES)}"
        )

        self.sample_steps: list = config["SAMPLE_STEPS"]
        self.sample_intervals: list = config["SAMPLE_INTERVALS"]
        self.sample_modes: list = config["SAMPLE_MODES"]
        self.sample_lengths: list = config["SAMPLE_LENGTHS"]
        self.clip_begin_strides = config.get("CLIP_BEGIN_STRIDE", 1)
        self.sample_stage = None
        self.sample_begin_frames = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.clip_begin_stride = 1
        self.sample_vid_tmax = None

        self.npy2rgb = config.get("NPY2RGB", False)
        self.frame_ext = ".jpg"

        dataset_dir = config.get("DATASET_DIR", "VT-Tiny-MOT")
        base_dataset_dir = os.path.join(config["DATA_ROOT"], dataset_dir)
        if self.dataset_version:
            base_dataset_dir = os.path.join(base_dataset_dir, self.dataset_version)
        self.split_dir = os.path.join(base_dataset_dir, f"{split}2017")
        self.ann_dir = os.path.join(base_dataset_dir, "annotations")
        assert os.path.exists(self.split_dir), f"Dir {self.split_dir} does not exist."
        assert os.path.exists(self.ann_dir), f"Dir {self.ann_dir} does not exist."

        def _log(msg: str):
            if logger is not None:
                logger.show(head=msg)
                logger.write(head=msg, filename="log.txt", mode="a")

        if self.dataset_version:
            _log(f"VT-Tiny-MOT dataset_version={self.dataset_version}, base_dir={base_dataset_dir}")

        ann_path = os.path.join(
            self.ann_dir,
            ANN_FILE_TEMPLATES[self.ann_mode].format(split=split),
        )
        ir_ann_path = None
        if self.ann_mode == "plain":
            ir_ann_path = os.path.join(
                self.ann_dir,
                ANN_FILE_TEMPLATES["01"].format(split=split),
            )
        assert os.path.exists(ann_path), f"Annotation file not found: {ann_path}"
        _log(f"VT-Tiny-MOT ann_mode={self.ann_mode}, ann_file={ann_path}")

        vid_white_list_cfg = config.get("VID_WHITE_LIST")
        vid_white_list = set(vid_white_list_cfg) if vid_white_list_cfg is not None else None

        self.train_half = bool(config.get("TRAIN_HALF", False)) and split == "train"
        self.train_half_list = None
        if self.train_half:
            self.train_half_list = set()
            self.train_half_file = os.path.join(base_dataset_dir, "train_half.txt")
            if os.path.exists(self.train_half_file):
                with open(self.train_half_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.train_half_list.add(line)
                _log(f"Use train half list from {self.train_half_file}, total {len(self.train_half_list)} vids.")
            else:
                _log(f"TRAIN_HALF enabled but {self.train_half_file} not found, ignore TRAIN_HALF.")
                self.train_half = False

        self.labels_full, self.frame_file_ids = load_coco_annotations(
            ann_path, self.ann_mode, ir_ann_path
        )
        self.vid_idx: dict[str, int] = {}
        self.idx_vid: dict[int, str] = {}

        kept_vids = 0
        skipped_by_whitelist = 0
        skipped_by_train_half = 0
        skipped_by_missing_dir = 0

        for scene in sorted(self.labels_full.keys()):
            if vid_white_list is not None and scene not in vid_white_list:
                skipped_by_whitelist += 1
                continue
            if self.train_half and self.train_half_list is not None and scene not in self.train_half_list:
                skipped_by_train_half += 1
                continue
            scene_dir = os.path.join(self.split_dir, scene, "00")
            if not os.path.isdir(scene_dir):
                skipped_by_missing_dir += 1
                continue

            self.vid_idx[scene] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[scene]] = scene
            kept_vids += 1

        _log(
            f"VT-Tiny-MOT video filtering: kept={kept_vids}, "
            f"skip_whitelist={skipped_by_whitelist}, skip_train_half={skipped_by_train_half}, "
            f"skip_missing_dir={skipped_by_missing_dir}"
        )
        assert kept_vids > 0, "No valid videos found for VTTinyMOT."

        self.set_epoch(0)

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        data_info = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        results = self.transform(data_info)
        if self.npy2rgb:
            images = [img[[0, 1, 2], ...] for img in results[0]]
        else:
            images = results[0]
        return {
            "images": images,
            "targets": results[1],
            "img_metas": results[2],
        }

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_epoch to init VTTinyMOT Dataset."
        return len(self.sample_begin_frames)

    def sample_frames_idx(self, vid: str, begin_frame: int) -> list[int]:
        if self.sample_length == 1:
            return [begin_frame]
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            return [begin_frame + interval * i for i in range(self.sample_length)]
        raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch: int):
        self.sample_begin_frames = []
        self.sample_vid_tmax = {}
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        self.clip_begin_stride = max(
            1, resolve_stage_scalar(self.clip_begin_strides, self.sample_stage, key="CLIP_BEGIN_STRIDE")
        )

        for vid in self.vid_idx.keys():
            mot_frame_ids = self.frame_file_ids.get(vid, {})
            assert mot_frame_ids, f"No frame index mapping found for video {vid}."
            t_min = min(mot_frame_ids.keys())
            t_max = max(mot_frame_ids.keys())
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1, self.clip_begin_stride):
                self.sample_begin_frames.append((vid, t))

    def get_vid_tmax(self, vid: str) -> int:
        mot_frame_ids = self.frame_file_ids.get(vid, {})
        assert mot_frame_ids, f"No frame index mapping found for video {vid}."
        return max(mot_frame_ids.keys())

    def get_single_frame(self, vid: str, idx: int):
        file_id = self.frame_file_ids[vid][idx]
        rgb_path = os.path.join(self.split_dir, vid, "00", f"{file_id:05d}{self.frame_ext}")
        data_info = {"filename": rgb_path, "ann": {}}

        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        obj_idx_offset = self.vid_idx[vid] * 100000

        for x, y, w, h, track_id, cls in self.labels_full[vid].get(idx, []):
            gt_bboxes.append(xywh_to_xyxy(x, y, w, h))
            gt_labels.append(int(cls))
            gt_ids.append(int(track_id) + obj_idx_offset)

        if gt_bboxes:
            data_info["ann"]["bboxes"] = np.array(gt_bboxes, dtype=np.float32)
            data_info["ann"]["labels"] = np.array(gt_labels, dtype=np.int64)
            data_info["ann"]["trackids"] = np.array(gt_ids, dtype=np.int64)
        else:
            data_info["ann"]["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            data_info["ann"]["labels"] = np.array([], dtype=np.int64)
            data_info["ann"]["trackids"] = np.zeros((0,), dtype=np.int64)

        results = dict(img_info=data_info, ann_info=data_info["ann"])
        results["img_prefix"] = None
        results["seg_prefix"] = None
        results["proposal_file"] = None
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        return results

    def get_multi_frames(self, vid: str, idxs: list[int]):
        return [self.get_single_frame(vid=vid, idx=i) for i in idxs]


def parse_crop_size(crop_size_cfg) -> tuple[int, int, int, int]:
    """解析 CROP_SIZE，返回 (w_min, w_max, h_min, h_max)。"""
    if isinstance(crop_size_cfg, dict):
        w_min, w_max = crop_size_cfg["W"]
        h_range = crop_size_cfg.get("H", crop_size_cfg["W"])
        h_min, h_max = h_range
    elif len(crop_size_cfg) == 4:
        w_min, w_max, h_min, h_max = crop_size_cfg
    elif len(crop_size_cfg) == 2:
        w_min, w_max = crop_size_cfg
        h_min, h_max = w_min, w_max
    else:
        raise ValueError(
            "CROP_SIZE must be {W:[min,max], H:[min,max]}, "
            f"[w_min, w_max, h_min, h_max], or [w_min, w_max], got {crop_size_cfg}."
        )
    return int(w_min), int(w_max), int(h_min), int(h_max)


def parse_resize_config(resize_cfg, resize_h_cfg=None):
    """解析 RESIZE。

    返回:
        ('range', (w_min, w_max, h_min, h_max)) — 宽高独立随机采样
        ('value', [(w, h), ...]) — 从固定尺度列表中随机选一个（兼容旧配置）
    """
    if isinstance(resize_cfg, dict):
        w_min, w_max = resize_cfg["W"]
        h_range = resize_cfg.get("H", resize_cfg["W"])
        h_min, h_max = h_range
        return "range", (int(w_min), int(w_max), int(h_min), int(h_max))
    if resize_cfg and isinstance(resize_cfg[0], (list, tuple)):
        scales = [(int(w), int(h)) for h, w in resize_cfg]
        return "value", scales
    if resize_h_cfg is not None:
        if len(resize_h_cfg) != len(resize_cfg):
            raise ValueError(
                f"RESIZE_H length {len(resize_h_cfg)} must match RESIZE length {len(resize_cfg)}."
            )
        scales = [(int(w), int(h)) for h, w in zip(resize_h_cfg, resize_cfg)]
        return "value", scales
    scales = [(int(w), int(w)) for w in resize_cfg]
    return "value", scales


def transforms_for_train(get_spectral_weights=False, transform_config=None):
    # 4 通道 (RGB + IR) 占位统计量，后续可替换为数据集真实 mean/std
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255, 127.5]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255, 127.5]

    resize_mode, resize_cfg = parse_resize_config(
        transform_config["RESIZE"],
        transform_config.get("RESIZE_H"),
    )
    crop_size = parse_crop_size(transform_config["CROP_SIZE"])
    flip_ratio = transform_config.get("FLIP_RATIO", 0.0)

    if resize_mode == "range":
        w_min, w_max, h_min, h_max = resize_cfg
        resize_transform = MotHResize(
            img_scale=[(w_min, h_min), (w_max, h_max)],
            multiscale_mode="range",
            bbox_clip_border=False,
            keep_ratio=False,
            resize_hw_range=resize_cfg,
        )
    else:
        resize_transform = MotHResize(
            multiscale_mode="value",
            img_scale=resize_cfg,
            bbox_clip_border=False,
            keep_ratio=False,
        )

    return MotCompose([
        MotLoadRgbIrImageFromJPG(),
        MotLoadAnnotations(poly2mask=False),
        MotHRandomFlip(direction=["horizontal"], flip_ratio=[flip_ratio]),
        MotHRandomCrop(
            crop_size=crop_size,
            crop_type="absolute_w_range",
            allow_negative_crop=False,
            iof_thr=0.5,
            keep_ratio=False,
        ),
        resize_transform,
        # TODO: ColorAug
        MotNormalize(mean=mean, std=std, to_rgb=False),
        MotPad(size_divisor=64),
        MotDefaultFormatBundle(),
        MotCollect(keys=["img", "gt_bboxes", "gt_labels", "gt_trackids"]),
        RectMotToMemotr(get_spectral_weights=get_spectral_weights),
        # TODO: ReverseClip
    ])


def build(config: dict, split: str, logger):
    if split != "train":
        raise ValueError(f"Data split {split} is not supported for VTTinyMOT yet.")

    return VTTinyMOT(
        config=config,
        split=split,
        transform=transforms_for_train(
            get_spectral_weights=config.get("DECODER_SPECTRAL_REFINE", False),
            transform_config=config["TRANSFORMS_CONFIG"],
        ),
        logger=logger,
        dataset_version=config.get("DATASET_VERSION"),
    )
