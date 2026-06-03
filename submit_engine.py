# Copyright (c) Ruopeng Gao. All Rights Reserved.
from __future__ import annotations

import os
import multiprocessing as mp
import queue
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from os import path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from data.seq_dataset import SeqDataset, SeqDataset_HeatmapGT, VtTinySeqDataset
from hsmot.datasets.pipelines.channel import rotate_norm_boxes_to_boxes
from hsmot.eval.validator import val_folder
from hsmot.eval.vt_tiny_coco import list_vt_tiny_channel_eval_runs, val_vt_tiny_coco_det
from hsmot.mmlab.hs_mmrotate import obb2poly
from hsmot.mmlab.hs_rectmot import denormalize_cxcywh
from log.logger import Logger
from models import build_model
from models.matcher import is_rect_memotr_version
from models.runtime_tracker import RuntimeTracker
from models.utils import get_model, load_checkpoint
from structures.track_instances import TrackInstances
from utils.batch_vis_result import draw_rotated_bbox
from utils.box_ops import box_cxcywh_to_xyxy, box_cxcywh_to_xywh
from utils.GMC import compute_gmc_sequence
from utils.nested_tensor import tensor_list_to_nested_tensor_with_shared_shapes
from utils.utils import (
    distributed_rank,
    distributed_world_size,
    inverse_sigmoid,
    is_distributed,
    load_train_config,
)


def resolve_two_stage_dir(base_dir: str, prefer: str = "stage2", logger: "Logger | None" = None) -> str:
    """两阶段训练（DETR pretrain + MOT finetune）下，自动在 base_dir 之下定位实际的训练输出目录。

    评测时 yaml/命令行通常只给出顶层 ``SUBMIT_DIR`` / ``EVAL_DIR``，而两阶段训练会把产物落到
    ``<base>/stage1_detr/`` 与 ``<base>/stage2_mot/``。本函数按以下顺序选择实际可用的目录：

      1. ``base_dir`` 本身存在 ``train/config.yaml`` → 视作单阶段训练目录，原样返回（兼容旧 yaml）。
      2. 否则按 ``prefer`` 顺序检查 ``stage2_mot``、``stage1_detr``，返回第一个含 ``train/config.yaml`` 的子目录。
      3. 都找不到 → 抛出 ``FileNotFoundError`` 并打印诊断信息。

    参数:
        base_dir: 顶层路径，对应 yaml/命令行的 ``SUBMIT_DIR`` / ``EVAL_DIR``。
        prefer:   ``"stage2"`` 优先评 MOT（默认）；``"stage1"`` 优先评 DETR。
        logger:   若提供，则用 logger 打印 fallback 信息；否则用 ``print``。
    """
    if base_dir is None:
        raise ValueError("submit/eval dir must not be None.")

    flag_rel = path.join("train", "config.yaml")

    if path.exists(path.join(base_dir, flag_rel)):
        return base_dir

    order = ["stage2_mot", "stage1_detr"] if prefer == "stage2" else ["stage1_detr", "stage2_mot"]
    for sub in order:
        full = path.join(base_dir, sub)
        if path.exists(path.join(full, flag_rel)):
            msg = (
                f"[two-stage] base_dir='{base_dir}' 无 train/config.yaml，"
                f"自动 fallback 到 '{full}' (prefer={prefer})。"
                " 如需评估另一个阶段，请显式 --submit-dir/--eval-dir 指向该子目录。"
            )
            if logger is not None:
                logger.show(head=msg)
            else:
                print(msg)
            return full

    raise FileNotFoundError(
        "Cannot resolve submit/eval dir for two-stage training. None of the following exists:\n"
        f"  - {path.join(base_dir, flag_rel)}\n"
        f"  - {path.join(base_dir, 'stage2_mot', flag_rel)}\n"
        f"  - {path.join(base_dir, 'stage1_detr', flag_rel)}\n"
        "请确认训练是否完成、目录是否正确，或手动 --submit-dir/--eval-dir 指到正确路径。"
    )


def is_vt_tiny_dataset(dataset_name: str) -> bool:
    return dataset_name in ("vt_tiny_mot", "VT-Tiny-MOT")


def resolve_vt_tiny_submit_split_dir(
    data_root: str,
    dataset_split: str,
    dataset_dir: str = "VT-Tiny-MOT",
    dataset_version: str | None = None,
) -> str:
    base_dataset_dir = path.join(data_root, dataset_dir)
    if dataset_version:
        base_dataset_dir = path.join(base_dataset_dir, dataset_version)
    split_dir = path.join(base_dataset_dir, f"{dataset_split}2017")
    if not path.isdir(split_dir):
        raise FileNotFoundError(f"VT-Tiny split dir not found: {split_dir}")
    return split_dir


def list_submit_sequences(data_split_dir: str, dataset_name: str) -> list[str]:
    if is_vt_tiny_dataset(dataset_name):
        seq_names = []
        for name in sorted(os.listdir(data_split_dir)):
            seq_path = path.join(data_split_dir, name)
            if path.isdir(seq_path) and path.isdir(path.join(seq_path, "00")):
                seq_names.append(name)
        return seq_names
    return sorted(os.listdir(data_split_dir))


def resolve_submit_dataset_root(
    data_root: str,
    dataset_name: str,
    dataset_split: str,
    dataset_version: str | None = None,
    dataset_type: str | None = None,
    dataset_dir: str | None = None,
) -> str:
    """Return dataset root used by submit / during-train eval."""
    if is_vt_tiny_dataset(dataset_name):
        base_dataset_dir = path.join(data_root, dataset_dir or "VT-Tiny-MOT")
        if dataset_version:
            base_dataset_dir = path.join(base_dataset_dir, dataset_version)
        return base_dataset_dir

    dataset_root = path.join(data_root, dataset_name.replace("_8ch", ""))
    if dataset_version is not None:
        dataset_root = path.join(dataset_root, dataset_version)
    return dataset_root


class Submitter:
    """Run MeMOTR inference on a single HSMOT sequence and dump tracking/detection results."""
    def __init__(self, dataset_name: str, split_dir: str, seq_name: str, outputs_dir: str, model: nn.Module,
                 dataset,
                 det_score_thresh: float = 0.7, track_score_thresh: float = 0.6, result_score_thresh: float = 0.7,
                 miss_tolerance: int = 5,
                 use_motion: bool = False, motion_lambda: float = 0.5,
                 motion_min_length: int = 3, motion_max_length: int = 5,
                 use_dab: bool = False,
                 visualize: bool = False,
                 decoder_spectral: bool = True,
                 use_scem_gt: bool = False,
                 only_train_detr: bool = False,
                 epoch: int = None,
                 draw_pic_dir: str = None,
                 rect_bbox: bool = False,
                 dataloader_num_workers: int = 4):
        self.dataset_name = dataset_name
        self.rect_bbox = rect_bbox
        self.seq_name = seq_name
        self.seq_dir = path.join(split_dir, seq_name)
        self.outputs_dir = outputs_dir
        self.predict_dir = path.join(self.outputs_dir, "tracker")
        self.predict_det_dir = path.join(self.outputs_dir, "det")
        self.model = model
        self.tracker = RuntimeTracker(det_score_thresh=det_score_thresh, track_score_thresh=track_score_thresh,
                                      miss_tolerance=miss_tolerance,
                                      use_motion=use_motion,
                                      motion_min_length=motion_min_length, motion_max_length=motion_max_length,
                                      visualize=visualize, use_dab=use_dab, decoder_spectral=decoder_spectral)
        self.result_score_thresh = result_score_thresh
        self.motion_lambda = motion_lambda
        self.use_scem_gt = use_scem_gt
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset, batch_size=1, num_workers=dataloader_num_workers, shuffle=False,
        )
        self.device = next(self.model.parameters()).device
        self.use_dab = use_dab
        self.use_motion = use_motion
        self.visualize = visualize
        self.decoder_spectral = decoder_spectral
        self.only_train_detr = only_train_detr
        self.draw_pic_dir = draw_pic_dir
        self.epoch = epoch

        os.makedirs(self.predict_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_dir, f'{self.seq_name}.txt')):
            os.remove(os.path.join(self.predict_dir, f'{self.seq_name}.txt'))
        os.makedirs(self.predict_det_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_det_dir, f'{self.seq_name}_det.txt')):
            os.remove(os.path.join(self.predict_det_dir, f'{self.seq_name}_det.txt'))
        self.model.eval()

        self.use_prior_map = False

        #如果有scem_module在model中
        if hasattr(get_model(self.model), 'scem_module') and get_model(self.model).scem_module is not None and get_model(self.model).scem_module.prior_mode is not None:
            self.use_prior_map = True

    @staticmethod
    def _effective_hw(ori_image):
        """Get effective (pre-pad) image size from original frame."""
        return int(ori_image.shape[1]), int(ori_image.shape[2])

    def _tracks_to_txt_lines(self, tracks_result: TrackInstances, frame_idx: int, eff_h: int, eff_w: int) -> list[str]:
        lines = []
        if len(tracks_result) == 0:
            return lines

        if self.rect_bbox:
            boxes = denormalize_cxcywh(tracks_result.boxes.cpu(), (eff_h, eff_w))
            boxes_xywh = box_cxcywh_to_xywh(boxes)
            for track, box in zip(tracks_result, boxes_xywh):
                x, y, w, h = box.tolist()
                obj_id = track.ids.item()
                conf = torch.max(track.scores, dim=-1).values.item()
                label = track.labels.item()
                lines.append(
                    f"{frame_idx + 1},{obj_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.3f},{label},-1,-1\n"
                )
            return lines

        boxes_xyxyxyxy = rotate_norm_boxes_to_boxes(tracks_result.boxes.cpu(), (eff_h, eff_w), version='le135')
        boxes_xyxyxyxy = obb2poly(boxes_xyxyxyxy)
        for track, xyxyxyxy in zip(tracks_result, boxes_xyxyxyxy):
            x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy.tolist()
            obj_id = track.ids.item()
            conf = torch.max(track.scores, dim=-1).values.item()
            label = track.labels.item()
            lines.append(
                f"{frame_idx + 1:6d},{obj_id:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},"
                f"{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n"
            )
        return lines

    def _pred_to_det_txt_lines(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        frame_idx: int,
        eff_h: int,
        eff_w: int,
    ) -> list[str]:
        lines = []
        if pred_boxes.numel() == 0:
            return lines

        if self.rect_bbox:
            boxes = denormalize_cxcywh(pred_boxes.cpu(), (eff_h, eff_w))
            boxes_xywh = box_cxcywh_to_xywh(boxes)
            det_labels = torch.max(pred_scores, dim=-1).indices
            det_confs = torch.max(pred_scores, dim=-1).values
            for box, det_conf, det_label in zip(boxes_xywh, det_confs, det_labels):
                x, y, w, h = box.tolist()
                lines.append(
                    f"{frame_idx + 1},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{det_conf.item():.3f},{det_label.item()},-1,-1\n"
                )
            return lines

        det_boxes_xyxyxyxy = rotate_norm_boxes_to_boxes(pred_boxes.cpu(), (eff_h, eff_w), version='le135')
        det_boxes_xyxyxyxy = obb2poly(det_boxes_xyxyxyxy)
        det_labels = torch.max(pred_scores, dim=-1).indices
        det_confs = torch.max(pred_scores, dim=-1).values
        for det_box, det_conf, det_label in zip(det_boxes_xyxyxyxy, det_confs, det_labels):
            x1, y1, x2, y2, x3, y3, x4, y4 = det_box.tolist()
            lines.append(
                f"{frame_idx + 1:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},"
                f"{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{det_conf.item():.3f},{det_label:2d},-1\n"
            )
        return lines

    @torch.no_grad()
    def run(self):
        """Entry point: choose the specific run mode based on config flags."""
        if self.only_train_detr:#仍然可能Prior_map = True
            self._run_with_only_train_detr()
        elif self.use_scem_gt:
            self._run_with_GT()
        elif self.use_prior_map:
            self._run_with_prior_map()
        else:
            self._run()

    @torch.no_grad()
    def _run_with_only_train_detr(self):
        """Inference path when only DETR is trained (no motion, no prior map, no SCEM)."""
        txt_lines = []
        det_txt_lines = []
        last_ori_image = None
        last_det_boxes_vis = None
        last_det_confs_vis = None
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # 单帧图像
            tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                    num_classes=get_model(self.model).num_classes,
                                    ).to(self.device)]

            effective_img_shape = ori_image.shape[1:4]  # (H, W, C), pre-pad valid area
            padded_img_shape = image[0].shape
            padded_img_shape = (padded_img_shape[1], padded_img_shape[2], padded_img_shape[0])
            frame = tensor_list_to_nested_tensor_with_shared_shapes(
                [image[0]],
                effective_img_shape=effective_img_shape,
                padded_img_shape=padded_img_shape
            ).to(self.device)

            res = self.model(frame=frame, tracks=tracks)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks = new_tracks
            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            eff_h, eff_w = self._effective_hw(ori_image)
            txt_lines.extend(self._tracks_to_txt_lines(tracks_result, i, eff_h, eff_w))

            det_scores = res["scores"][0].cpu()
            det_txt_lines.extend(
                self._pred_to_det_txt_lines(res["pred_bboxes"][0].cpu(), det_scores, i, eff_h, eff_w)
            )
            if self.draw_pic_dir is not None:
                last_ori_image = ori_image
                last_det_confs_vis = torch.max(det_scores, dim=-1).values
                if self.rect_bbox:
                    boxes = denormalize_cxcywh(res["pred_bboxes"][0].cpu(), (eff_h, eff_w))
                    last_det_boxes_vis = box_cxcywh_to_xyxy(boxes)
                else:
                    last_det_boxes_vis = obb2poly(
                        rotate_norm_boxes_to_boxes(res["pred_bboxes"][0].cpu(), (eff_h, eff_w), version='le135')
                    )

        # 保存跟踪结果
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)

        # 保存检测结果
        with open(os.path.join(self.predict_det_dir, f"{self.seq_name}_det.txt"), "w") as file:
            file.writelines(det_txt_lines)

        # 保存画图
        if self.draw_pic_dir is not None and last_ori_image is not None:
            save_pic = os.path.join(self.draw_pic_dir, f"ep{self.epoch}_{self.seq_name}_det.jpg")
            if self.rect_bbox:
                img = np.ascontiguousarray(last_ori_image[0].cpu().numpy()[:, :, :3][:, :, ::-1])
                for box, conf in zip(last_det_boxes_vis, last_det_confs_vis):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    color = (0, 0, 255) if conf >= 0.5 else (0, 255, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            else:
                img = np.ascontiguousarray(last_ori_image[0].cpu().numpy()[:, :, [4, 2, 1]])
                for boxes, confs in zip(last_det_boxes_vis, last_det_confs_vis):
                    if confs < 0.1:
                        color = (128, 0, 0)
                    elif confs < 0.5:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    draw_rotated_bbox(
                        img, None,
                        boxes[0], boxes[1], boxes[2], boxes[3],
                        boxes[4], boxes[5], boxes[6], boxes[7],
                        confs,
                        thickness=1,
                        font_scale=0,
                        color=color
                    )
            cv2.imwrite(save_pic, img)

    @torch.no_grad()
    def _run_with_prior_map(self):
        """Inference path using learned prior map (SCEM prior_mode != None)."""
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 ).to(self.device)]

        txt_lines = []
        prev_frame = None
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            effective_img_shape = ori_image.shape[1:4]  # (H, W, C), pre-pad valid area
            padded_img_shape = image[0].shape
            padded_img_shape = (padded_img_shape[1], padded_img_shape[2], padded_img_shape[0])
            frame = tensor_list_to_nested_tensor_with_shared_shapes(
                [image[0]],
                effective_img_shape=effective_img_shape,
                padded_img_shape=padded_img_shape
            ).to(self.device)

            if prev_frame is not None:
                gmc = compute_gmc_sequence(images=[prev_frame[0], frame.tensors[0]], method='sparseOptFlow', downscale=1)[-1] # [2, 3]
            else:
                gmc = np.eye(2, 3, dtype=np.float32)
            prev_frame = frame.tensors.detach().clone()
            gmc = torch.tensor(gmc, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, 2, 3]

            res = self.model(frame=frame, tracks=tracks, gmc=gmc)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)

            if self.use_motion:
                for _ in range(len(tracks[0])):
                    if tracks[0].disappear_time[_].item() > 0:
                        if len(self.tracker.motions[tracks[0].ids[_].item()]) >= \
                               self.tracker.motions[tracks[0].ids[_].item()].min_record_length:
                            tracks[0].ref_pts[_] = inverse_sigmoid(
                                tracks[0].last_appear_boxes[_]
                            ) + self.motion_lambda * self.tracker.motions[tracks[0].ids[_].item()].get_box_delta(
                                miss_length=tracks[0].disappear_time[_].item()
                            ).to(tracks[0].last_appear_boxes.device)

            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            eff_h, eff_w = self._effective_hw(ori_image)
            txt_lines.extend(self._tracks_to_txt_lines(tracks_result, i, eff_h, eff_w))

        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)

    @torch.no_grad()
    def _run_with_GT(self):
        """Inference path that consumes GT heatmap as SCEM supervision (debug/eval)."""
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 ).to(self.device)]

        txt_lines = []
        for i, ((image, ori_image), info, heatmap) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            effective_img_shape = ori_image.shape[1:4]  # (H, W, C), pre-pad valid area
            padded_img_shape = image[0].shape
            padded_img_shape = (padded_img_shape[1], padded_img_shape[2], padded_img_shape[0])
            frame = tensor_list_to_nested_tensor_with_shared_shapes(
                [image[0]],
                effective_img_shape=effective_img_shape,
                padded_img_shape=padded_img_shape
            ).to(self.device)
            heatmap = heatmap.to(self.device)
            res = self.model(frame=frame, tracks=tracks, heatmap=heatmap)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)

            if self.use_motion:
                for _ in range(len(tracks[0])):
                    if tracks[0].disappear_time[_].item() > 0:
                        if len(self.tracker.motions[tracks[0].ids[_].item()]) >= \
                               self.tracker.motions[tracks[0].ids[_].item()].min_record_length:
                            tracks[0].ref_pts[_] = inverse_sigmoid(
                                tracks[0].last_appear_boxes[_]
                            ) + self.motion_lambda * self.tracker.motions[tracks[0].ids[_].item()].get_box_delta(
                                miss_length=tracks[0].disappear_time[_].item()
                            ).to(tracks[0].last_appear_boxes.device)

            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            eff_h, eff_w = self._effective_hw(ori_image)
            txt_lines.extend(self._tracks_to_txt_lines(tracks_result, i, eff_h, eff_w))

        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)

    @torch.no_grad()
    def _run(self):
        """Default inference path: no prior map, optional motion compensation."""
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 ).to(self.device)]

        txt_lines = []
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            effective_img_shape = ori_image.shape[1:4]  # (H, W, C), pre-pad valid area
            padded_img_shape = image[0].shape
            padded_img_shape = (padded_img_shape[1], padded_img_shape[2], padded_img_shape[0])
            frame = tensor_list_to_nested_tensor_with_shared_shapes(
                [image[0]],
                effective_img_shape=effective_img_shape,
                padded_img_shape=padded_img_shape
            ).to(self.device)
            res = self.model(frame=frame, tracks=tracks)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)

            if self.use_motion:
                for _ in range(len(tracks[0])):
                    if tracks[0].disappear_time[_].item() > 0:
                        if len(self.tracker.motions[tracks[0].ids[_].item()]) >= \
                               self.tracker.motions[tracks[0].ids[_].item()].min_record_length:
                            tracks[0].ref_pts[_] = inverse_sigmoid(
                                tracks[0].last_appear_boxes[_]
                            ) + self.motion_lambda * self.tracker.motions[tracks[0].ids[_].item()].get_box_delta(
                                miss_length=tracks[0].disappear_time[_].item()
                            ).to(tracks[0].last_appear_boxes.device)

            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            eff_h, eff_w = self._effective_hw(ori_image)
            txt_lines.extend(self._tracks_to_txt_lines(tracks_result, i, eff_h, eff_w))

        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)

    @staticmethod
    def filter_by_score(tracks: TrackInstances, thresh: float = 0.7):
        keep = torch.max(tracks.scores, dim=-1).values > thresh
        return tracks[keep]

    @staticmethod
    def filter_by_area(tracks: TrackInstances, thresh: int = 10):
        assert len(tracks.area) == len(tracks.ids), f"Tracks' 'area' should have the same dim with 'ids'"
        keep = tracks.area > thresh
        return tracks[keep]

    def update_results(self, tracks_result: TrackInstances, frame_idx: int, results: list, img_path: str):
        """Helper to convert tracks into BDD100K json-style result for a single frame."""
        bdd_cls2label = {
            1: "pedestrian",
            2: "rider",
            3: "car",
            4: "truck",
            5: "bus",
            6: "train",
            7: "motorcycle",
            8: "bicycle"
        }
        frame_result = {
            "name": img_path.split("/")[-1],
            "videoName": img_path.split("/")[-1][:-12],
            # "frameIndex": int(img_path.split("/")[-1][:-4].split("-")[-1]) - 1
            "frameIndex": frame_idx,
            "labels": []
        }
        for i in range(len(tracks_result)):
            x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
            ID = str(tracks_result.ids[i].item())
            label = bdd_cls2label[tracks_result.labels[i].item() + 1]
            frame_result["labels"].append(
                {
                    "id": ID,
                    "category": label,
                    "box2d": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }
            )
        results.append(frame_result)
        return

    def write_results(self, tracks_result: TrackInstances, frame_idx: int):
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "a") as file:
            for i in range(len(tracks_result)):
                if self.dataset_name == "DanceTrack" or self.dataset_name == "SportsMOT" \
                        or self.dataset_name == "MOT17" or self.dataset_name == "MOT17_SPLIT":
                    x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
                    w, h = x2 - x1, y2 - y1
                    result_line = f"{frame_idx+1}," \
                                  f"{tracks_result.ids[i].item()}," \
                                  f"{x1},{y1},{w},{h},1,-1,-1,-1\n"
                else:
                    raise ValueError(f"{self.dataset_name} dataset is not supported for submit process.")
                file.write(result_line)
        return


@dataclass
class SubmitRunContext:
    dataset_name: str
    data_split_dir: str
    outputs_dir: str
    train_config: dict
    config: dict
    det_score_thresh: float
    track_score_thresh: float
    result_score_thresh: float
    use_motion: bool
    motion_min_length: int
    motion_max_length: int
    motion_lambda: float
    miss_tolerance: int
    use_scem_gt: bool
    dataset_type: str | None
    only_train_detr: bool = False
    epoch: int | None = None
    draw_pic_dir: str | None = None
    dataloader_num_workers: int = 4


def resolve_submit_dataloader_workers(config: dict, num_workers: int) -> int:
    """多进程 submit 时降低每个进程 DataLoader 的 worker 数，避免进程数爆炸。"""
    val = config.get("SUBMIT_DATALOADER_WORKERS")
    if val is not None and str(val).strip() != "":
        return max(0, int(val))
    base = int(config.get("NUM_WORKERS", 2))
    if num_workers <= 1:
        return base
    return max(0, base // num_workers)


def _parse_gpu_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def resolve_submit_gpus(config: dict) -> list[int]:
    """Resolve logical cuda device indices for submit workers (0 .. device_count-1).

    ``AVAILABLE_GPUS`` in main.py sets ``CUDA_VISIBLE_DEVICES`` (physical GPU ids),
    not ``torch.cuda`` indices. After that env is applied, workers must use 0,1,...
    """
    for key in ("SUBMIT_GPUS", "SUBMIT_GPU"):
        gpu_ids = _parse_gpu_list(config.get(key))
        if gpu_ids:
            return gpu_ids
    if is_distributed():
        return [distributed_rank()]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 0:
            return list(range(n))
    return [0]


def _split_sequences_for_current_rank(seq_names: list[str]) -> list[str]:
    if not is_distributed():
        return seq_names
    world_size = distributed_world_size()
    rank = distributed_rank()
    return [seq_name for idx, seq_name in enumerate(seq_names) if idx % world_size == rank]


def _resolve_gpu_ids_for_submit(config: dict) -> list[int]:
    if is_distributed():
        return [distributed_rank()]
    return resolve_submit_gpus(config)


def _format_submit_mode(gpu_ids: list[int], num_workers: int) -> str:
    if num_workers <= 1:
        return "single-process"
    gpu_text = ",".join(str(g) for g in gpu_ids)
    return f"multi-process x{num_workers} on gpu(s) [{gpu_text}]"


def _make_submit_run_context(
    *,
    dataset_name: str,
    data_split_dir: str,
    outputs_dir: str,
    train_config: dict,
    config: dict,
    use_scem_gt: bool,
    dataset_type: str | None,
    dataloader_num_workers: int,
    only_train_detr: bool = False,
    epoch: int | None = None,
    draw_pic_dir: str | None = None,
) -> SubmitRunContext:
    return SubmitRunContext(
        dataset_name=dataset_name,
        data_split_dir=data_split_dir,
        outputs_dir=outputs_dir,
        train_config=train_config,
        config=config,
        det_score_thresh=config["DET_SCORE_THRESH"],
        track_score_thresh=config["TRACK_SCORE_THRESH"],
        result_score_thresh=config["RESULT_SCORE_THRESH"],
        use_motion=config["USE_MOTION"],
        motion_min_length=config["MOTION_MIN_LENGTH"],
        motion_max_length=config["MOTION_MAX_LENGTH"],
        motion_lambda=config["MOTION_LAMBDA"],
        miss_tolerance=config["MISS_TOLERANCE"],
        use_scem_gt=use_scem_gt,
        dataset_type=dataset_type,
        only_train_detr=only_train_detr,
        epoch=epoch,
        draw_pic_dir=draw_pic_dir,
        dataloader_num_workers=dataloader_num_workers,
    )


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _init_track_instances_static(train_config: dict, model: nn.Module) -> None:
    rect_bbox = is_rect_memotr_version(train_config.get("MEMOTR_VERSION", ""))
    TrackInstances.set_static_properties(
        use_spectral_decoder=train_config.get("DECODER_SPECTRAL", True),
        use_dab=train_config["USE_DAB"],
        use_q_spec=bool(getattr(get_model(model), "use_q_spec", False)),
        bbox_dim=4 if rect_bbox else 5,
    )


def _load_submit_model(train_config: dict, checkpoint_path: str, gpu_id: int) -> nn.Module:
    model = build_model(config=train_config)
    load_checkpoint(model=model, path=checkpoint_path)
    model = model.cuda(gpu_id)
    model.eval()
    return model


def _load_submit_model_from_state_dict(
    train_config: dict,
    state_dict: dict,
    gpu_id: int,
) -> nn.Module:
    model = build_model(config=train_config)
    model.load_state_dict(state_dict)
    model = model.cuda(gpu_id)
    model.eval()
    return model


def _create_submit_worker_model(
    train_config: dict,
    gpu_id: int,
    checkpoint_path: str | None,
    state_dict: dict | None,
) -> nn.Module:
    if checkpoint_path is not None:
        return _load_submit_model(train_config, checkpoint_path, gpu_id)
    if state_dict is not None:
        return _load_submit_model_from_state_dict(train_config, state_dict, gpu_id)
    raise ValueError("Either checkpoint_path or state_dict must be provided for submit worker.")


def _collect_error_queue(error_queue: mp.Queue) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except queue.Empty:
            break
    return errors


def _shutdown_submit_processes(
    processes: list[mp.Process],
    task_queue: mp.JoinableQueue | None,
    num_workers: int,
    *,
    terminate: bool = False,
    join_timeout: float = 10.0,
) -> None:
    """关闭 submit 子进程；terminate=True 时强制终止。"""
    if task_queue is not None:
        try:
            for _ in range(num_workers):
                task_queue.put(None)
        except (BrokenPipeError, OSError, ValueError):
            pass

    deadline = time.time() + join_timeout
    for process in processes:
        if not process.is_alive():
            continue
        remaining = max(0.0, deadline - time.time())
        process.join(timeout=remaining)

    if terminate:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)


def _submit_worker_process(
    worker_idx: int,
    gpu_id: int,
    task_queue: mp.JoinableQueue,
    error_queue: mp.Queue,
    ctx: SubmitRunContext,
    checkpoint_path: str | None,
    state_dict: dict | None,
) -> None:
    """子进程入口：加载 1 个 model，动态从队列取序列推理。"""
    try:
        torch.cuda.set_device(gpu_id)
        model = _create_submit_worker_model(
            train_config=ctx.train_config,
            gpu_id=gpu_id,
            checkpoint_path=checkpoint_path,
            state_dict=state_dict,
        )
        _init_track_instances_static(ctx.train_config, model)
        while True:
            seq_name = task_queue.get()
            if seq_name is None:
                task_queue.task_done()
                break
            try:
                _process_one_sequence(seq_name, model, ctx)
            except Exception as exc:
                error_queue.put((str(seq_name), repr(exc)))
            finally:
                task_queue.task_done()
    except Exception as exc:
        error_queue.put(("__worker__", f"worker {worker_idx} on gpu {gpu_id} failed: {exc!r}"))


def _run_submit_in_main_process(
    seq_names: list[str],
    ctx: SubmitRunContext,
    gpu_id: int,
    checkpoint_path: str | None,
    state_dict: dict | None,
) -> None:
    model = _create_submit_worker_model(
        train_config=ctx.train_config,
        gpu_id=gpu_id,
        checkpoint_path=checkpoint_path,
        state_dict=state_dict,
    )
    _init_track_instances_static(ctx.train_config, model)
    for seq_name in seq_names:
        _process_one_sequence(seq_name, model, ctx)


def _build_submitter_from_ctx(
    ctx: SubmitRunContext,
    seq_name: str,
    dataset,
    model: nn.Module,
) -> Submitter:
    train_config = ctx.train_config
    rect_bbox = is_rect_memotr_version(train_config.get("MEMOTR_VERSION", ""))
    return Submitter(
        dataset_name=ctx.dataset_name,
        split_dir=ctx.data_split_dir,
        seq_name=str(seq_name),
        outputs_dir=ctx.outputs_dir,
        model=model,
        dataset=dataset,
        use_dab=train_config["USE_DAB"],
        det_score_thresh=ctx.det_score_thresh,
        track_score_thresh=ctx.track_score_thresh,
        result_score_thresh=ctx.result_score_thresh,
        use_motion=ctx.use_motion,
        motion_min_length=ctx.motion_min_length,
        motion_max_length=ctx.motion_max_length,
        motion_lambda=ctx.motion_lambda,
        miss_tolerance=ctx.miss_tolerance,
        decoder_spectral=train_config.get("DECODER_SPECTRAL", True),
        use_scem_gt=ctx.use_scem_gt,
        only_train_detr=ctx.only_train_detr,
        epoch=ctx.epoch,
        draw_pic_dir=ctx.draw_pic_dir,
        rect_bbox=rect_bbox,
        dataloader_num_workers=ctx.dataloader_num_workers,
    )


def _process_one_sequence(seq_name: str, model: nn.Module, ctx: SubmitRunContext) -> None:
    dataset = build_seq_dataset(
        seq_dir=path.join(ctx.data_split_dir, str(seq_name)),
        split_dir=ctx.data_split_dir,
        seq_name=str(seq_name),
        npy2rgb=ctx.config["NPY2RGB"],
        use_scem_gt=ctx.use_scem_gt,
        dataset_type=ctx.dataset_type,
        dataset_name=ctx.dataset_name,
    )
    _build_submitter_from_ctx(ctx, seq_name, dataset, model).run()


def run_submit_sequences_parallel(
    seq_names: list[str],
    ctx: SubmitRunContext,
    num_workers: int,
    gpu_ids: list[int],
    checkpoint_path: str | None = None,
    state_dict: dict | None = None,
    logger: Logger | None = None,
) -> None:
    """多进程 submit：每进程 1 个 model，共享任务队列，可多个进程复用同一张 GPU。"""
    if not seq_names:
        return

    num_workers = max(1, min(int(num_workers), len(seq_names)))
    if logger is not None:
        logger.show(
            head=(
                f"[submit] {_format_submit_mode(gpu_ids, num_workers)}: "
                f"dataloader_workers={ctx.dataloader_num_workers}, sequences={len(seq_names)}"
            )
        )

    if num_workers == 1:
        _run_submit_in_main_process(
            seq_names=seq_names,
            ctx=ctx,
            gpu_id=gpu_ids[0],
            checkpoint_path=checkpoint_path,
            state_dict=state_dict,
        )
        return

    mp_ctx = mp.get_context("spawn")
    task_queue: mp.JoinableQueue = mp_ctx.JoinableQueue()
    error_queue: mp.Queue = mp_ctx.Queue()
    processes: list[mp.Process] = []

    for seq_name in seq_names:
        task_queue.put(str(seq_name))

    try:
        for worker_idx in range(num_workers):
            gpu_id = gpu_ids[worker_idx % len(gpu_ids)]
            process = mp_ctx.Process(
                target=_submit_worker_process,
                args=(
                    worker_idx,
                    gpu_id,
                    task_queue,
                    error_queue,
                    ctx,
                    checkpoint_path,
                    state_dict,
                ),
                name=f"submit-worker-{worker_idx}",
                daemon=False,
            )
            process.start()
            processes.append(process)

        if logger is not None:
            worker_desc = ", ".join(
                f"w{i}->gpu{gpu_ids[i % len(gpu_ids)]}" for i in range(num_workers)
            )
            logger.show(head=f"[submit] started workers: {worker_desc}")

        for _ in range(num_workers):
            task_queue.put(None)

        task_queue.join()
        _shutdown_submit_processes(processes, task_queue=None, num_workers=0, terminate=False)
    except KeyboardInterrupt:
        if logger is not None:
            logger.show(head="[submit] interrupted, terminating worker processes...")
        _shutdown_submit_processes(processes, task_queue, num_workers, terminate=True)
        raise
    except Exception:
        _shutdown_submit_processes(processes, task_queue, num_workers, terminate=True)
        raise

    errors = _collect_error_queue(error_queue)
    if errors:
        details = "\n".join(f"  - {seq}: {err}" for seq, err in errors)
        raise RuntimeError(f"Submit failed for {len(errors)} sequence(s):\n{details}")


def _run_submit_pipeline(
    *,
    config: dict,
    train_config: dict,
    outputs_dir: str,
    data_split_dir: str,
    dataset_name: str,
    seq_names: list[str],
    logger: Logger,
    use_scem_gt: bool,
    dataset_type: str | None,
    checkpoint_path: str | None = None,
    source_model: nn.Module | None = None,
    only_train_detr: bool = False,
    epoch: int | None = None,
    draw_pic_dir: str | None = None,
) -> None:
    submit_workers = int(config.get("SUBMIT_THREADS", 1))
    dataloader_num_workers = resolve_submit_dataloader_workers(config, submit_workers)
    gpu_ids = _resolve_gpu_ids_for_submit(config)
    seq_names = _split_sequences_for_current_rank(seq_names)

    state_dict = None
    if source_model is not None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in _unwrap_model(source_model).state_dict().items()
        }

    ctx = _make_submit_run_context(
        dataset_name=dataset_name,
        data_split_dir=data_split_dir,
        outputs_dir=outputs_dir,
        train_config=train_config,
        config=config,
        use_scem_gt=use_scem_gt,
        dataset_type=dataset_type,
        dataloader_num_workers=dataloader_num_workers,
        only_train_detr=only_train_detr,
        epoch=epoch,
        draw_pic_dir=draw_pic_dir,
    )
    run_submit_sequences_parallel(
        seq_names=seq_names,
        ctx=ctx,
        num_workers=submit_workers,
        gpu_ids=gpu_ids,
        checkpoint_path=checkpoint_path,
        state_dict=state_dict,
        logger=logger,
    )


_SUBMIT_TRAIN_CONFIG_OVERLAY_KEYS = ("MEMOTR_VERSION",)


def resolve_submit_checkpoint_and_output(config: dict) -> tuple[str, str]:
    """
    Resolve checkpoint root (weights + train/config.yaml) and output root (tracker/eval).

    When SUBMIT_CHECKPOINT_DIR is set, checkpoints are read there and results are written
    under SUBMIT_OUTPUT_DIR (or SUBMIT_DIR if output dir is omitted).
    """
    prefer_stage = str(config.get("SUBMIT_STAGE_PREFER", "stage2")).lower()
    checkpoint_base = config.get("SUBMIT_CHECKPOINT_DIR") or config["SUBMIT_DIR"]
    checkpoint_root = resolve_two_stage_dir(checkpoint_base, prefer=prefer_stage)

    output_base = config.get("SUBMIT_OUTPUT_DIR") or config["SUBMIT_DIR"]
    if output_base is None:
        output_root = checkpoint_root
    else:
        output_root = path.abspath(output_base)
        os.makedirs(output_root, exist_ok=True)
    return checkpoint_root, output_root


def _overlay_submit_train_config(train_config: dict, config: dict) -> None:
    """Allow standalone submit yaml to override model fields saved in train/config.yaml."""
    for key in _SUBMIT_TRAIN_CONFIG_OVERLAY_KEYS:
        if key in config and config[key] is not None:
            train_config[key] = config[key]


def submit(config: dict):
    assert config["SUBMIT_DIR"] is not None, f"'--submit-dir' must not be None for submit process."
    assert config["SUBMIT_MODEL"] is not None, f"'--submit-model' must not be None for submit process."
    assert config["SUBMIT_DATA_SPLIT"] is not None, f"'--submit-data-split' must not be None for submit process."
    # 两阶段训练 fallback：若顶层 SUBMIT_DIR 无 train/config.yaml，则自动指向 stage2_mot/stage1_detr。
    # 用 yaml 顶层可选字段 SUBMIT_STAGE_PREFER 控制（默认 "stage2"），命令行也可直接 --submit-dir 指明。
    checkpoint_root, output_root = resolve_submit_checkpoint_and_output(config)
    config["SUBMIT_CHECKPOINT_DIR"] = checkpoint_root
    config["SUBMIT_OUTPUT_DIR"] = output_root

    submit_logger = Logger(logdir=os.path.join(output_root, config["SUBMIT_DATA_SPLIT"]), only_main=True)
    submit_logger.show(head="Configs:", log=config)
    submit_logger.write(log=config, filename="config.yaml", mode="w")

    train_config = load_train_config(path=path.join(checkpoint_root, "train/config.yaml"))
    _overlay_submit_train_config(train_config=train_config, config=config)

    dataset_name = train_config["DATASET"]
    config["DATASET"] = dataset_name
    dataset_split = config["SUBMIT_DATA_SPLIT"]
    outputs_dir = path.join(output_root, dataset_split)
    dataset_type = config.get("DATASET_TYPE", train_config.get("DATASET_TYPE", None))
    use_scem_gt = config.get("SCEM", {}).get("USE_GT", False)
    dataset_dir = train_config.get("DATASET_DIR", "VT-Tiny-MOT")
    dataset_version = config.get("DATASET_VERSION", train_config.get("DATASET_VERSION"))
    checkpoint_path = path.join(checkpoint_root, config["SUBMIT_MODEL"])

    data_split_dir = resolve_submit_split_dir(
        data_root=config["DATA_ROOT"],
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_version=dataset_version,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
    )
    seq_names = list_submit_sequences(data_split_dir, dataset_name)

    _run_submit_pipeline(
        config=config,
        train_config=train_config,
        outputs_dir=outputs_dir,
        data_split_dir=data_split_dir,
        dataset_name=dataset_name,
        seq_names=seq_names,
        logger=submit_logger,
        use_scem_gt=use_scem_gt,
        dataset_type=dataset_type,
        checkpoint_path=checkpoint_path,
        only_train_detr=train_config.get("ONLY_TRAIN_DETR", False),
    )


def resolve_submit_split_dir(
    data_root: str,
    dataset_name: str,
    dataset_split: str,
    dataset_version: str | None = None,
    dataset_type: str | None = None,
    dataset_dir: str | None = None,
):
    """
    Resolve sequence folder for submit/infer.
    hsmot:
      - 3JPG -> <root>/<split>/npy2jpg
      - else -> <root>/<split>/npy
    vt_tiny_mot:
      - <root>/<DATASET_DIR>/<version?>/{split}2017/
    """
    if is_vt_tiny_dataset(dataset_name):
        return resolve_vt_tiny_submit_split_dir(
            data_root=data_root,
            dataset_split=dataset_split,
            dataset_dir=dataset_dir or "VT-Tiny-MOT",
            dataset_version=dataset_version,
        )

    if "hsmot" not in dataset_name:
        raise ValueError(f"Unsupported dataset for submit process: {dataset_name}")

    dataset_root = path.join(data_root, dataset_name.replace("_8ch", ""))
    if dataset_version is not None:
        dataset_root = path.join(dataset_root, dataset_version)

    split_root = path.join(dataset_root, dataset_split)
    dataset_type_upper = str(dataset_type).upper() if dataset_type is not None else "NPY"
    target_subdir = "npy2jpg" if dataset_type_upper == "3JPG" else "npy"
    target_dir = path.join(split_root, target_subdir)
    if path.isdir(target_dir):
        return target_dir
    return split_root


def build_seq_dataset(
    seq_dir: str,
    split_dir: str,
    seq_name: str,
    npy2rgb: bool,
    use_scem_gt: bool,
    dataset_type: str | None = None,
    dataset_name: str | None = None,
):
    """Centralized SeqDataset builder used by both submit() and submit_during_train()."""
    if dataset_name is not None and is_vt_tiny_dataset(dataset_name):
        if use_scem_gt:
            raise ValueError("VT-Tiny-MOT submit does not support SCEM USE_GT mode.")
        return VtTinySeqDataset(seq_dir=seq_dir)

    if use_scem_gt:
        label_file = os.path.join(split_dir, "..", "mot", f"{seq_name}.txt")
        return SeqDataset_HeatmapGT(seq_dir=seq_dir, label_file=label_file, npy2rgb=npy2rgb, dataset_type=dataset_type)
    return SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb, dataset_type=dataset_type)


def _trackeval_script_path(script_name: str) -> str:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    return path.join(current_file_dir, "..", "TrackEval", "scripts", script_name)


def _run_vt_tiny_trackeval(
    *,
    config: dict,
    dataset_root: str,
    dataset_split: str,
    data_split_dir: str,
    tracker_dir: str,
    trackers_name: str,
    trackers_subfolder: str,
    submit_logger: Logger | None = None,
) -> None:
    """VT-Tiny MOT 评测：00 / 01 各跑一次，结果分别写入 eval_00 / eval_01。"""
    iou_thresh = config.get("EVAL_IOU_THRESHOLD", config.get("TRACK_IOU_THRESH", 0.5))
    script = _trackeval_script_path("run_vt_tiny_mot.py")
    eval_runs = list_vt_tiny_channel_eval_runs(
        dataset_root=dataset_root,
        dataset_split=dataset_split,
        mot_stage=True,
    )
    for run in eval_runs:
        ch = run["channel"]
        gt_coco_ann = run["gt_coco_ann"]
        output_sub = run["output_sub_folder"]
        cmd = (
            f"{sys.executable} {script} "
            f"--USE_PARALLEL False "
            f"--METRICS HOTA CLEAR Identity "
            f"--GT_COCO_ANN {gt_coco_ann} "
            f"--IMG_FOLDER {data_split_dir} "
            f"--TRACKERS_FOLDER {tracker_dir} "
            f"--TRACKERS_TO_EVAL {trackers_name} "
            f"--TRACKER_SUB_FOLDER {trackers_subfolder} "
            f"--IOU_THRESHOLD {iou_thresh} "
            f"--OUTPUT_SUB_FOLDER {output_sub} "
        )
        head = f"VT-Tiny MOT TrackEval ({ch} -> {output_sub})"
        if submit_logger is not None:
            submit_logger.show(head=head, log=cmd)
        os_flag = os.system(cmd)
        assert os_flag == 0, f"TrackEval for VT-Tiny channel {ch} failed to run."


def _run_hsmot_trackeval(
    *,
    dataset_root: str,
    dataset_split: str,
    dataset_type: str | None,
    tracker_dir: str,
    trackers_name: str,
    trackers_subfolder: str,
) -> None:
    gt_dir = path.join(dataset_root, dataset_split, "mot")
    if str(dataset_type).upper() == "3JPG":
        img_dir = path.join(dataset_root, dataset_split, "npy2jpg")
    else:
        img_dir = path.join(dataset_root, dataset_split, "npy")
    script = _trackeval_script_path("run_hsmot_8ch.py")
    cmd = (
        f"{sys.executable} {script} "
        f"--USE_PARALLEL False "
        f"--METRICS HOTA CLEAR Identity "
        f"--GT_FOLDER {gt_dir} "
        f"--TRACKERS_FOLDER {tracker_dir} "
        f"--TRACKERS_TO_EVAL {trackers_name} "
        f"--TRACKER_SUB_FOLDER {trackers_subfolder} "
        f"--IMG_FOLDER {img_dir} "
    )
    os_flag = os.system(cmd)
    assert os_flag == 0, "TrackEval failed to run."


def _run_post_submit_eval(
    *,
    config: dict,
    dataset_name: str,
    dataset_split: str,
    dataset_type: str | None,
    dataset_root: str,
    data_split_dir: str,
    tracker_dir: str,
    trackers_name: str,
    trackers_subfolder: str,
    only_train_detr: bool,
    submit_logger: Logger,
    train_logger: Logger | None = None,
) -> None:
    """训练时 submit 后的评测：先按阶段（DETR / MOT），再按数据集分流。"""
    if only_train_detr:
        # 阶段 1（STAGE1 / ONLY_TRAIN_DETR）：检测预训练验证
        if is_vt_tiny_dataset(dataset_name):
            pred_det_folder = path.join(tracker_dir, trackers_name, "det")
            eval_runs = list_vt_tiny_channel_eval_runs(
                dataset_root=dataset_root,
                dataset_split=dataset_split,
                mot_stage=False,
            )
            for run in eval_runs:
                ch = run["channel"]
                val_lines = val_vt_tiny_coco_det(
                    gt_coco_ann=run["gt_coco_ann"],
                    pred_det_folder=pred_det_folder,
                    data_split_dir=data_split_dir,
                    ann_mode=run["ann_mode"],
                    ir_ann_path=None,
                )
                head = f"Stage1 DETR Validation Results (VT-Tiny {ch}):"
                log_text = "\n".join(val_lines)
                submit_logger.show(head=head, log=log_text)
                submit_logger.write(head=head, log=log_text, filename="log.txt", mode="a")
                if train_logger is not None:
                    train_logger.write(head=head, log=log_text, filename="log.txt", mode="a")
        else:
            gt_dir = path.join(dataset_root, dataset_split, "mot")
            val_lines = val_folder(
                gt_folder=gt_dir,
                pred_folder=path.join(tracker_dir, trackers_name, trackers_subfolder),
            )
            submit_logger.show(head="Stage1 DETR Validation Results:", log="\n".join(val_lines))
            submit_logger.write(
                head="Stage1 DETR Validation Results:",
                log="\n".join(val_lines),
                filename="log.txt",
                mode="a",
            )
            if train_logger is not None:
                train_logger.write(
                    head="Stage1 DETR Validation Results:",
                    log="\n".join(val_lines),
                    filename="log.txt",
                    mode="a",
                )
    else:
        # 阶段 2（MOT finetune）：完整跟踪评测
        if is_vt_tiny_dataset(dataset_name):
            _run_vt_tiny_trackeval(
                config=config,
                dataset_root=dataset_root,
                dataset_split=dataset_split,
                data_split_dir=data_split_dir,
                tracker_dir=tracker_dir,
                trackers_name=trackers_name,
                trackers_subfolder=trackers_subfolder,
                submit_logger=submit_logger,
            )
        else:
            _run_hsmot_trackeval(
                dataset_root=dataset_root,
                dataset_split=dataset_split,
                dataset_type=dataset_type,
                tracker_dir=tracker_dir,
                trackers_name=trackers_name,
                trackers_subfolder=trackers_subfolder,
            )


def submit_during_train(config: dict, epoch: int, model: nn.Module, only_train_detr: bool = False, train_logger: Logger = None):

    model.eval()

    assert config["SUBMIT_DIR"] is not None, f"'--submit-dir' must not be None for submit process."
    assert config["SUBMIT_DATA_SPLIT"] is not None, f"'--submit-data-split' must not be None for submit process."

    draw_pic_dir = os.path.join(config["SUBMIT_DIR"], "draw_pic")
    os.makedirs(draw_pic_dir, exist_ok=True)

    submit_dir_epoch = os.path.join(config["SUBMIT_DIR"], f"epoch_{epoch}")
    submit_logger = Logger(logdir=os.path.join(submit_dir_epoch, config["SUBMIT_DATA_SPLIT"]), only_main=True)
    submit_logger.show(head="Configs:", log=config)
    submit_logger.write(log=config, filename="config.yaml", mode="w")

    dataset_name = config["DATASET"]
    dataset_split = config["SUBMIT_DATA_SPLIT"]
    outputs_dir = path.join(submit_dir_epoch, dataset_split)
    dataset_version = config.get("DATASET_VERSION")
    dataset_type = config.get("DATASET_TYPE", "NPY")
    use_scem_gt = config["SCEM"]["USE_GT"]
    dataset_dir = config.get("DATASET_DIR", "VT-Tiny-MOT")

    data_split_dir = resolve_submit_split_dir(
        data_root=config["DATA_ROOT"],
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_version=dataset_version,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
    )
    dataset_root = resolve_submit_dataset_root(
        data_root=config["DATA_ROOT"],
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_version=dataset_version,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
    )
    seq_names = list_submit_sequences(data_split_dir, dataset_name)

    _run_submit_pipeline(
        config=config,
        train_config=config,
        outputs_dir=outputs_dir,
        data_split_dir=data_split_dir,
        dataset_name=dataset_name,
        seq_names=seq_names,
        logger=submit_logger,
        use_scem_gt=use_scem_gt,
        dataset_type=dataset_type,
        source_model=model,
        only_train_detr=only_train_detr,
        epoch=epoch,
        draw_pic_dir=draw_pic_dir,
    )

    if is_distributed():
        torch.distributed.barrier()

    if distributed_rank() == 0:
        _run_post_submit_eval(
            config=config,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dataset_type=dataset_type,
            dataset_root=dataset_root,
            data_split_dir=data_split_dir,
            tracker_dir=submit_dir_epoch,
            trackers_name=outputs_dir.split("/")[-1],
            trackers_subfolder="tracker",
            only_train_detr=only_train_detr,
            submit_logger=submit_logger,
            train_logger=train_logger,
        )

    if is_distributed():
        torch.distributed.barrier()