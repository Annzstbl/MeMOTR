import os
from math import floor
from random import randint

import torch
from PIL import Image
import data.transforms as T
# from typing import List
# from torch.utils.data import Dataset
from .mot import MOTDataset
from .utils import resolve_stage_scalar
from collections import defaultdict

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
from hsmot.mmlab.hs_mmrotate import poly2obb, poly2obb_np
import os.path as osp
from hsmot.datasets.pipelines.compose import MotCompose, MotRandomChoice
from hsmot.datasets.pipelines.channel import MotrToMmrotate, MmrotateToMotr, MmrotateToMemotr, MotipToMmrotate
from hsmot.datasets.pipelines.loading import (
    MotLoadAnnotations,
    MotLoadImageFromFile,
    MotLoadMultichannelImageFrom3JPG,
    MotLoadMultichannelImageFromNpy,
)
from hsmot.datasets.pipelines.transforms import MotRRsize, MotRRandomFlip, MotRRandomCrop, MotNormalize, MotPad
from hsmot.datasets.pipelines.formatting import MotCollect, MotDefaultFormatBundle, MotShow


class hsmot_8ch(MOTDataset):
    def __init__(self, config: dict, split: str, transform, version='le135', logger=None, dataset_version=None):
        super(hsmot_8ch, self).__init__(config=config, split=split, transform=transform)

        self.config = config
        self.transform = transform
        self.dataset_name = config["DATASET"]
        self.dataset_name = self.dataset_name.replace("_8ch", "")
        assert split == "train" or split == "test", f"Split {split} is not supported!"
        self.version = version

        # Sampling setting.
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

        self.npy2rgb = config["NPY2RGB"]
        self.dataset_type = config.get("DATASET_TYPE", "NPY").upper()

        self.gts = defaultdict(lambda: defaultdict(list))
        self.vid_idx = dict()
        self.idx_vid = dict()
        self.dataset_version = dataset_version

        # 构造基础数据集目录，若提供 version，则路径变为 DATASET/version
        base_dataset_dir = os.path.join(config["DATA_ROOT"], self.dataset_name)
        if self.dataset_version is not None:
            base_dataset_dir = os.path.join(base_dataset_dir, self.dataset_version)

        if self.dataset_type == "NPY":
            self.data_subdir = "npy"
            self.frame_ext = ".npy"
        elif self.dataset_type == "3JPG":
            self.data_subdir = "npy2jpg"
            self.frame_ext = ".jpg"
        else:
            raise ValueError(f"Unsupported DATASET_TYPE: {self.dataset_type}")

        self.split_dir = os.path.join(base_dataset_dir, split, self.data_subdir)
        assert os.path.exists(self.split_dir), f"Dir {self.split_dir} is not exist."
        self.labels_dir = os.path.join(base_dataset_dir, split, "mot")

        def _log(msg: str):
            if logger is not None:
                logger.show(head=msg)
                logger.write(head=msg, filename="log.txt", mode="a")

        # 过滤优先级规则：
        # 1) dataset_version 只决定 base_dataset_dir（已在上方处理）
        # 2) VID_WHITE_LIST 和 TRAIN_HALF 都是“过滤器”，最终取交集
        # 3) TRAIN_HALF 仅在 train split 生效
        vid_white_list_cfg = config.get("VID_WHITE_LIST")
        vid_white_list = set(vid_white_list_cfg) if vid_white_list_cfg is not None else None

        self.train_half = bool(config.get("TRAIN_HALF", False)) and split == "train"
        self.train_half_list = None
        if self.train_half:
            self.train_half_list = set()
            self.train_half_file = os.path.join(base_dataset_dir, "train_half.txt")
            assert os.path.exists(self.train_half_file), f"File {self.train_half_file} is not exist."
            with open(self.train_half_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.train_half_list.add(line)
            _log(f"Use train half list from {self.train_half_file}, total {len(self.train_half_list)} vids.")
        elif bool(config.get("TRAIN_HALF", False)) and split != "train":
            _log("TRAIN_HALF is enabled but split is not train, ignore TRAIN_HALF.")

        self.labels_full = defaultdict(lambda: defaultdict(list))
        total_vids = 0
        kept_vids = 0
        skipped_by_whitelist = 0
        skipped_by_train_half = 0
        for vid in os.listdir(self.labels_dir):
            if not vid.endswith(".txt"):
                continue
            total_vids += 1
            vid_name = os.path.splitext(vid)[0]

            if vid_white_list is not None and vid_name not in vid_white_list:
                skipped_by_whitelist += 1
                continue
            if self.train_half and (self.train_half_list is not None) and vid_name not in self.train_half_list:
                skipped_by_train_half += 1
                continue

            kept_vids += 1
            gt_path = os.path.join(self.labels_dir, vid)
            with open(gt_path, "r") as f:
                for l in f:
                    t, i, *x0y0x1y1x2y2x3y3, _, cls, trunc = l.strip().split(',')[:13]
                    t, i, cls = map(int, (t, i, cls))
                    x0, y0, x1, y1, x2, y2, x3, y3 = map(float, (x0y0x1y1x2y2x3y3))
                    self.labels_full[vid][t].append(np.array([x0, y0, x1, y1, x2, y2, x3, y3, i, cls], dtype=np.float32))

        _log(
            f"Video filtering done: total={total_vids}, kept={kept_vids}, "
            f"skip_whitelist={skipped_by_whitelist}, skip_train_half={skipped_by_train_half}"
        )
        vid_files = list(self.labels_full.keys())

        for vid in vid_files:
            self.vid_idx[vid] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[vid]] = vid

        self.set_epoch(0)

        return

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        data_info = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        results = self.transform(data_info)
        if self.npy2rgb:
            images = [img[[1,2,4],...] for img in results[0]]
        else:
            images = results[0]
        return{
            "images": images,
            "targets": results[1],
            "img_metas": results[2]
        }

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frames)

    def sample_frames_idx(self, vid: int, begin_frame: int) -> list[int]:
        if self.sample_length == 1:
            return [begin_frame]

        elif self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idxs = [begin_frame + interval * i for i in range(self.sample_length)]
            return frame_idxs
        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch: int):
        self.sample_begin_frames = list()
        self.sample_vid_tmax = dict()
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
            t_min = min(self.labels_full[vid].keys())
            t_max = self.get_vid_tmax(vid)
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1, self.clip_begin_stride):
                self.sample_begin_frames.append((vid, t))

        return

    def get_vid_tmax(self, vid: str) -> int:
        vid_dir = os.path.join(self.split_dir, osp.splitext(vid)[0])
        assert os.path.exists(vid_dir), f"Dir {vid_dir} is not exist."

        if self.dataset_type == "NPY":
            frame_ids = [
                int(osp.splitext(file_name)[0])
                for file_name in os.listdir(vid_dir)
                if file_name.endswith(self.frame_ext)
            ]
        else:
            frame_ids = [
                int(osp.splitext(file_name)[0].rsplit('_', 1)[0])
                for file_name in os.listdir(vid_dir)
                if file_name.endswith(self.frame_ext) and '_p' in osp.splitext(file_name)[0]
            ]

        assert frame_ids, f"No valid frames found in {vid_dir}"
        return max(frame_ids)

    def get_single_frame(self, vid: str, idx: int):
        #确认源代码的 frame_idx 现在看不需要
        '''
            info["boxes"] = list()
            info["ids"] = list()
            info["labels"] = list()
            info["areas"] = list()
            info["frame_idx"] = torch.as_tensor(idx)

        '''
        if self.dataset_type == "3JPG":
            frame_name = f'{idx:06d}_p1{self.frame_ext}'
        else:
            frame_name = f'{idx:06d}{self.frame_ext}'
        img_path = os.path.join(self.split_dir, osp.splitext(vid)[0], frame_name)
        data_info = {}
        data_info['filename'] = img_path
        data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_polygons = []
        obj_idx_offset = self.vid_idx[vid] * 100000
        
        for *xyxyxyxy, id, cls in self.labels_full[vid][idx]:
            x, y, w, h, a = poly2obb_np(np.array(xyxyxyxy, dtype=np.float32), self.version)
            gt_bboxes.append([x, y, w, h, a])
            gt_labels.append(cls)
            gt_polygons.append(xyxyxyxy)
            gt_ids.append(id+obj_idx_offset)
        
        if gt_bboxes:
            data_info['ann']['bboxes'] = np.array(
                gt_bboxes, dtype=np.float32)
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64)
            data_info['ann']['polygons'] = np.array(
                gt_polygons, dtype=np.float32)
            data_info['ann']['trackids'] = np.array(gt_ids, dtype=np.int64)
        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                    dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
            data_info['ann']['polygons'] = np.zeros((0, 8),
                                                    dtype=np.float32)
            data_info['ann']['trackids'] = np.zeros((0), dtype=np.int64)
        
        img_info = data_info
        ann_info = data_info['ann']
        results = dict(img_info=img_info, ann_info=ann_info)

        # """Prepare results dict for pipeline."""
        results['img_prefix'] = None
        results['seg_prefix'] = None
        results['proposal_file'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
        return results

    def get_multi_frames(self, vid: str, idxs: list[int]):
        return [self.get_single_frame(vid=vid, idx=i) for i in idxs]


def transforms_for_train(use_cache=True, cache_path=None, spectral_method=None, spectral_n_clusters=None,
                         get_spectral_weights=True, transform_config=None, dataset_type="NPY"):
    mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
    std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
    mean = [_*255 for _ in mean]
    std = [_*255 for _ in std]

    assert transform_config["RESIZE"] is not None 
    if transform_config["RESIZE"] is not None:
        resize = transform_config["RESIZE"]
    scale_w = resize
    scale_h = [int(w/4*3) for w in scale_w]
    scales = list(zip(scale_h, scale_w))

    crop_size = tuple(transform_config["CROP_SIZE"])
    flip_ratio = transform_config["FLIP_RATIO"]

    if dataset_type == "3JPG":
        load_image = MotLoadMultichannelImageFrom3JPG()
    elif dataset_type == "NPY":
        load_image = MotLoadMultichannelImageFromNpy()
    else:
        raise ValueError(f"Unsupported DATASET_TYPE: {dataset_type}")

    return MotCompose([
                MotipToMmrotate(),
                load_image,
                MotLoadAnnotations(poly2mask=False),
                MotRRandomFlip(direction=['horizontal'], flip_ratio=[flip_ratio], version='le135'),
                MotRRandomCrop(crop_size=crop_size, crop_type='absolute_w_range', version='le135',
                               allow_negative_crop=False, iof_thr=0.5, keep_ratio=True),
                MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),       
                # 缺少一个颜色预训练
                MotNormalize(mean=mean, std=std, to_rgb=False),
                MotPad(size_divisor=64),
                # MotShow(save_path='/data4/litianhao/hsmot/memotr/debug99/debug_img', version='le135', mean=mean, std=std, img_name_tail='2026_3_3', show_proposals=False, to_bgr=False),
                MotDefaultFormatBundle(),
                MotCollect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_trackids']),
                MmrotateToMemotr(use_cache=use_cache, cache_path=cache_path, spectral_method=spectral_method, spectral_n_clusters=spectral_n_clusters, mean=mean, std=std, get_spectral_weights=get_spectral_weights)
                #TODO 缺少一个reverse clip 但实际参数是0所以暂不实现

            ])


def build(config: dict, split: str, logger):
    resize = config["RESIZE"] if "RESIZE" in config else None
    if split == "train":
        return hsmot_8ch(
            config=config,
            split=split,
            transform=transforms_for_train(
                cache_path=os.path.join(config["DATA_ROOT"], config["DATASET"].replace("_8ch", "")),
                get_spectral_weights=config["DECODER_SPECTRAL_REFINE"],
                use_cache=config["DECODER_SPECTRAL_USE_CACHE"],
                spectral_n_clusters=config["DECODER_SPECTRAL_CLUSTERS"],
                spectral_method=config["DECODER_SPECTRAL_METHOD"],
                transform_config=config["TRANSFORMS_CONFIG"],
                dataset_type=config.get("DATASET_TYPE", "NPY").upper()
            ),
            logger = logger,
            dataset_version=config["DATASET_VERSION"]
        )
    else:
        raise ValueError(f"Data split {split} is not supported.")
