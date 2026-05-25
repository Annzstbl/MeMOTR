# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import cv2

import torchvision.transforms.functional as F

from torch.utils.data import Dataset
import numpy as np
import mmcv
from hsmot.mmlab.hs_mmdet import to_tensor
from hsmot.datasets.pipelines.channel import HeatmapFromRotateGt
import torch
from collections import defaultdict


def _detect_dataset_type(seq_dir: str, dataset_type=None) -> str:
    if dataset_type is not None:
        return dataset_type.upper()

    file_names = sorted(os.listdir(seq_dir))
    if any(file_name.endswith('.npy') for file_name in file_names):
        return "NPY"
    if any(file_name.endswith('.jpg') and '_p1' in os.path.splitext(file_name)[0] for file_name in file_names):
        return "3JPG"
    raise ValueError(f"Cannot infer dataset type from {seq_dir}")


def _collect_image_paths(seq_dir: str, dataset_type: str):
    file_names = sorted(os.listdir(seq_dir))
    if dataset_type == "NPY":
        return [os.path.join(seq_dir, file_name) for file_name in file_names if file_name.endswith('.npy')]
    if dataset_type == "3JPG":
        return [
            os.path.join(seq_dir, file_name)
            for file_name in file_names
            if file_name.endswith('.jpg') and '_p1' in os.path.splitext(file_name)[0]
        ]
    raise ValueError(f"Unsupported DATASET_TYPE: {dataset_type}")


def _load_multichannel_image(path: str, dataset_type: str):
    if dataset_type == "NPY":
        image = np.load(path)
        assert image is not None
        return image

    if dataset_type == "3JPG":
        stem, ext = os.path.splitext(path)
        base_stem = stem.rsplit('_', 1)[0] if stem.endswith(('_p1', '_p2', '_p3')) else stem
        part_paths = [f'{base_stem}_p1{ext}', f'{base_stem}_p2{ext}', f'{base_stem}_p3{ext}']
        part_images = []
        for part_path in part_paths:
            image = mmcv.imread(part_path)
            assert image is not None, f"Failed to load image: {part_path}"
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            part_images.append(image)
        return np.concatenate([part_images[0], part_images[1], part_images[2][:, :, :2]], axis=2)

    raise ValueError(f"Unsupported DATASET_TYPE: {dataset_type}")


class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, stride=64, npy2rgb=False, dataset_type=None):
        # a hack implementation for BDD100K and others:
        # if "BDD100K" in seq_dir:
        #     image_paths = sorted(os.listdir(os.path.join(seq_dir)))
        #     image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        # else:
        #     image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
        #     image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        self.dataset_type = _detect_dataset_type(seq_dir, dataset_type)
        self.image_paths = _collect_image_paths(seq_dir, self.dataset_type)
        self.image_height = 900
        self.image_width = 1200
        mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
        std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
        mean = [_*255 for _ in mean]
        std = [_*255 for _ in std]
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.stride=stride
        self.npy2rgb=npy2rgb

        return

    @staticmethod
    def load(path, dataset_type):
        return _load_multichannel_image(path, dataset_type)

    def process_image(self, image):
        ori_image = image.copy()

        # 首先归一化  然后resize 然后padding
        image = mmcv.imnormalize(image, self.mean, self.std, to_rgb=False)
        # image = mmcv.imresize(image, (self.scale_size_w, self.scale_size_h))
        image = mmcv.impad_to_multiple(image, self.stride, pad_val=0)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = to_tensor(image)
        if self.npy2rgb:
            image = image[[1,2,4], :, :]
        # image = image.unsqueeze(0)
        return image, ori_image

        # ori_image = image.copy()
        # h, w = image.shape[:2]
        # scale = self.image_height / min(h, w)
        # if max(h, w) * scale > self.image_width:
        #     scale = self.image_width / max(h, w)
        # target_h = int(h * scale)
        # target_w = int(w * scale)
        # image = cv2.resize(image, (target_w, target_h))
        # image = F.normalize(F.to_tensor(image), self.mean, self.std)
        # return image, ori_image

    def __getitem__(self, item):
        image = self.load(self.image_paths[item], self.dataset_type)
        info = self.image_paths[item]
        return self.process_image(image=image), info

    def __len__(self):
        return len(self.image_paths)


class VtTinySeqDataset(Dataset):
    """VT-Tiny-MOT 推理序列：读取 scene/00 与 scene/01 JPG，拼接为 4 通道。"""

    VT_TINY_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255, 127.5]
    VT_TINY_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255, 127.5]

    def __init__(self, seq_dir: str, stride: int = 64):
        rgb_dir = os.path.join(seq_dir, "00")
        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"VT-Tiny RGB dir not found: {rgb_dir}")
        self.image_paths = sorted(
            os.path.join(rgb_dir, file_name)
            for file_name in os.listdir(rgb_dir)
            if file_name.endswith(".jpg")
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No RGB jpg found under {rgb_dir}")
        self.mean = np.array(self.VT_TINY_MEAN, dtype=np.float32)
        self.std = np.array(self.VT_TINY_STD, dtype=np.float32)
        self.stride = stride

    @staticmethod
    def load_rgb_ir(rgb_path: str) -> np.ndarray:
        ir_path = rgb_path.replace("/00/", "/01/")
        if not os.path.exists(ir_path):
            raise FileNotFoundError(f"IR image not found: {ir_path}")
        rgb = mmcv.imread(rgb_path)
        assert rgb is not None, f"Failed to load RGB image: {rgb_path}"
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)

        ir = mmcv.imread(ir_path, flag="grayscale")
        assert ir is not None, f"Failed to load IR image: {ir_path}"
        if ir.ndim == 2:
            ir = ir[..., np.newaxis]
        return np.concatenate([rgb, ir], axis=2)

    def process_image(self, image: np.ndarray):
        ori_image = image.copy()
        image = mmcv.imnormalize(image, self.mean, self.std, to_rgb=False)
        image = mmcv.impad_to_multiple(image, self.stride, pad_val=0)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = to_tensor(image)
        return image, ori_image

    def __getitem__(self, item):
        image = self.load_rgb_ir(self.image_paths[item])
        return self.process_image(image), self.image_paths[item]

    def __len__(self):
        return len(self.image_paths)


class SeqDataset_HeatmapGT(Dataset):
    def __init__(self, seq_dir: str, label_file:str, stride=64, npy2rgb=False, dataset_type=None):
        # a hack implementation for BDD100K and others:
        # if "BDD100K" in seq_dir:
        #     image_paths = sorted(os.listdir(os.path.join(seq_dir)))
        #     image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        # else:
        #     image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
        #     image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        self.dataset_type = _detect_dataset_type(seq_dir, dataset_type)
        self.image_paths = _collect_image_paths(seq_dir, self.dataset_type)
        self.image_height = 900
        self.image_width = 1200
        mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
        std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
        mean = [_*255 for _ in mean]
        std = [_*255 for _ in std]
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.stride=stride
        self.npy2rgb=npy2rgb

        # 读取label
        # frameid 从1开始，这里减去1 从0开始
        self.label_full = defaultdict(list)  # frame_id -> list of boxes
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                t, i, *x0y0x1y1x2y2x3y3, _, cls, trunc = l.strip().split(',')[:13] 
                t, i, cls = map(int, (t, i, cls))
                x0, y0, x1, y1, x2, y2, x3, y3 = map(float, (x0y0x1y1x2y2x3y3))
                self.label_full[t-1].append(np.array([x0, y0, x1, y1, x2, y2, x3, y3, i, cls], dtype=np.float32))

    @staticmethod
    def load(path, dataset_type):
        return _load_multichannel_image(path, dataset_type)

    def process_image(self, image):
        ori_image = image.copy()

        # 首先归一化  然后resize 然后padding
        image = mmcv.imnormalize(image, self.mean, self.std, to_rgb=False)
        # image = mmcv.imresize(image, (self.scale_size_w, self.scale_size_h))
        image = mmcv.impad_to_multiple(image, self.stride, pad_val=0)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = to_tensor(image)
        if self.npy2rgb:
            image = image[[1,2,4], :, :]
        # image = image.unsqueeze(0)
        return image, ori_image



    def __getitem__(self, item):
        image = self.load(self.image_paths[item], self.dataset_type)
        info = self.image_paths[item]
        labels = np.stack(self.label_full[item]) # xyxyxyxy id cls  (N, 10)
        xyxyxyxy = torch.tensor(labels[:, :8], dtype=torch.float32)

        image, ori_image = self.process_image(image=image)
        heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xyxyxyxy(xyxyxyxy, image.shape[1:], 'le135')
        return (image, ori_image), info, heatmap

    def __len__(self):
        return len(self.image_paths)