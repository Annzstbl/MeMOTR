# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import json
import torch
import torch.nn as nn
import sys

from tqdm import tqdm
from os import path
from typing import List
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from models import build_model
from models.utils import load_checkpoint, get_model
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict, is_distributed, distributed_world_size, distributed_rank, inverse_sigmoid
from utils.nested_tensor import tensor_list_to_nested_tensor_already_padded_shape
from utils.box_ops import box_cxcywh_to_xyxy
from log.logger import Logger
from data.seq_dataset import SeqDataset, SeqDataset_HeatmapGT
from structures.track_instances import TrackInstances
from hsmot.datasets.pipelines.channel import rotate_norm_boxes_to_boxes
from hsmot.mmlab.hs_mmrotate import obb2poly
from utils.GMC import compute_gmc_sequence
import numpy as np 
from hsmot.eval.validator import PredictValidator, val_folder
from utils.batch_vis_result import draw_rotated_bbox
import cv2


class Submitter:
    def __init__(self, dataset_name: str, split_dir: str, seq_name: str, outputs_dir: str, model: nn.Module,
                 det_score_thresh: float = 0.7, track_score_thresh: float = 0.6, result_score_thresh: float = 0.7,
                 miss_tolerance: int = 5,
                 use_motion: bool = False, motion_lambda: float = 0.5,
                 motion_min_length: int = 3, motion_max_length: int = 5,
                 use_dab: bool = False,
                 visualize: bool = False,
                 npy2rgb: bool = False, 
                 decoder_spectral: bool = True,
                 use_scem_gt: bool = False,
                 only_train_detr: bool = False,
                 epoch: int = None,
                 draw_pic_dir: str = None):
        self.dataset_name = dataset_name
        self.seq_name = seq_name
        self.seq_dir = path.join(split_dir, seq_name)
        self.outputs_dir = outputs_dir
        self.predict_dir = path.join(self.outputs_dir, "tracker")
        self.model = model
        self.tracker = RuntimeTracker(det_score_thresh=det_score_thresh, track_score_thresh=track_score_thresh,
                                      miss_tolerance=miss_tolerance,
                                      use_motion=use_motion,
                                      motion_min_length=motion_min_length, motion_max_length=motion_max_length,
                                      visualize=visualize, use_dab=use_dab, decoder_spectral=decoder_spectral)
        self.result_score_thresh = result_score_thresh
        self.motion_lambda = motion_lambda
        self.use_scem_gt = use_scem_gt
        if self.use_scem_gt:
            self.label_file = os.path.join(split_dir, '..', 'mot', '{seq_name}.txt'.format(seq_name=seq_name))
            self.dataset = SeqDataset_HeatmapGT(seq_dir=self.seq_dir, label_file=self.label_file, npy2rgb=npy2rgb)
        else:
            self.dataset = SeqDataset(seq_dir=self.seq_dir, npy2rgb=npy2rgb)
        self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=4, shuffle=False)
        self.device = next(self.model.parameters()).device
        self.use_dab = use_dab
        self.use_motion = use_motion
        self.visualize = visualize
        self.decoder_spectral = decoder_spectral
        self.only_train_detr = only_train_detr
        self.draw_pic_dir = draw_pic_dir
        self.epoch = epoch

        # 对路径进行一些操作
        os.makedirs(self.predict_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_dir, f'{self.seq_name}.txt')):
            os.remove(os.path.join(self.predict_dir, f'{self.seq_name}.txt'))
        self.model.eval()

        self.use_prior_map = False

        #如果有scem_module在model中
        if hasattr(get_model(self.model), 'scem_module') and get_model(self.model).scem_module is not None and get_model(self.model).scem_module.prior_mode is not None:
            self.use_prior_map = True
        return

    @torch.no_grad()
    def run(self):

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


        txt_lines = []
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # 单帧图像
            tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                    num_classes=get_model(self.model).num_classes,
                                    use_dab=self.use_dab,
                                    ).to(self.device)]

            # image: (1, C, H, W); ori_image: (1, H, W, C)
            ori_img_shape = ori_image.shape[1:4]#(H, W, C)
            pad_shape = image[0].shape
            pad_shape = (pad_shape[1], pad_shape[2], pad_shape[0])
            frame = tensor_list_to_nested_tensor_already_padded_shape([image[0]], ori_img_shape, pad_shape).to(self.device)
            # frame = tensor_list_to_nested_tensor([image[0]]).to(self.device)

            res = self.model(frame=frame, tracks=tracks)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            # tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)
            tracks = new_tracks
            tracks_result = tracks[0].to(torch.device("cpu"))
            ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
            # box = [x, y, w, h]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            # to xyxy:
            # tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            # tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))

            # if self.dataset_name == "BDD100K":
            #     self.update_results(tracks_result=tracks_result, frame_idx=i, results=bdd100k_results, img_path=info[0])
            # else:
            boxes_xyxyxyxy = rotate_norm_boxes_to_boxes(tracks_result.boxes.cpu(), (image.shape[2], image.shape[3]), version='le135')
            boxes_xyxyxyxy = obb2poly(boxes_xyxyxyxy)

            for _tracks, xyxyxyxy in zip(tracks_result, boxes_xyxyxyxy):
                save_format = '{frame:6d},{id:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n'
                x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy.tolist()
                obj_id = _tracks.ids.item()
                conf = torch.max(_tracks.scores, dim=-1).values.item()
                label = _tracks.labels.item()
                line = save_format.format(frame=i + 1, id=obj_id, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, conf=conf, label=label)
                txt_lines.append(line)
            
            # 整理检测结果，包括所有得分的检测框
            # scores = model_outputs["scores"]#经过了logits_to_scores处理
            # boxes = model_outputs["pred_bboxes"][0]
            # frame_id = i+1
            # label = torch.max(new_tracks.scores, dim=-1).indices
            # save_format = '{frame:6d},{id:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n'
            det_txt_lines = []
            det_boxes_xyxyxyxy = rotate_norm_boxes_to_boxes(res["pred_bboxes"][0].cpu(), (image.shape[2], image.shape[3]), version='le135')
            det_boxes_xyxyxyxy = obb2poly(det_boxes_xyxyxyxy)
            det_scores = res["scores"][0].cpu()
            det_labels = torch.max(det_scores, dim=-1).indices
            det_confs = torch.max(det_scores, dim=-1).values
            for i, (det_box, det_conf, det_label) in enumerate(zip(det_boxes_xyxyxyxy, det_confs, det_labels)):
                save_format = '{frame:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n'
                x1, y1, x2, y2, x3, y3, x4, y4 = det_box.tolist()
                conf = det_conf.item()
                label = det_label.item()
                line = save_format.format(frame=i + 1, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, conf=conf, label=label)
                det_txt_lines.append(line)

        # 保存跟踪结果
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)

        # 保存检测结果
        with open(os.path.join(self.predict_dir, f"{self.seq_name}_det.txt"), "w") as file:
            file.writelines(det_txt_lines)

        # 保存画图
        if self.draw_pic_dir is not None:
            save_pic = os.path.join(self.draw_pic_dir, f"ep{self.epoch}_{self.seq_name}_det.jpg")
            img = np.ascontiguousarray(ori_image[0].cpu().numpy()[:,:,[4,2,1]])
            for boxes, confs in zip (det_boxes_xyxyxyxy, det_confs):
                if confs < 0.1:
                    draw_rotated_bbox(img, None, boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6], boxes[7], confs, thickness=1, font_scale=0, color=(128,0,0))
                else:
                    draw_rotated_bbox(img, None, boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6], boxes[7], confs,color=(0,0,255))
            cv2.imwrite(save_pic, img)

    @torch.no_grad()
    def _run_with_prior_map(self):
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 use_dab=self.use_dab,
                                 ).to(self.device)]

        txt_lines = []
        prev_frame = None
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # image: (1, C, H, W); ori_image: (1, H, W, C)
            frame = tensor_list_to_nested_tensor([image[0]]).to(self.device)

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

            # We do not use this...
            # but I do not want to remove this part.
            # WHAT IF it breaks down!!!
            # of course not :)
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
            # box = [x, y, w, h]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            # to xyxy:
            # tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            # tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))

            # if self.dataset_name == "BDD100K":
            #     self.update_results(tracks_result=tracks_result, frame_idx=i, results=bdd100k_results, img_path=info[0])
            # else:
            boxes_xyxyxyxy = rotate_norm_boxes_to_boxes(tracks_result.boxes.cpu(), (image.shape[2], image.shape[3]), version='le135')
            boxes_xyxyxyxy = obb2poly(boxes_xyxyxyxy)

            for _tracks, xyxyxyxy in zip(tracks_result, boxes_xyxyxyxy):
                save_format = '{frame:6d},{id:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n'
                x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy.tolist()
                obj_id = _tracks.ids.item()
                conf = torch.max(_tracks.scores, dim=-1).values.item()
                label = _tracks.labels.item()
                line = save_format.format(frame=i + 1, id=obj_id, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, conf=conf, label=label)
                txt_lines.append(line)

            # save_path = os.path.join("/data3/litianhao/hsmot/paper/memotr3ch", self.seq_name)
            # os.makedirs(save_path, exist_ok=True)
            # #tracks[0].query_embed和tracks[0].ids保存下来
            # torch.save(tracks[0].query_embed.cpu(), os.path.join(save_path, f"{i}_query_embed.pt"))
            # torch.save(tracks[0].ids.cpu(), os.path.join(save_path, f"{i}_ids.pt"))
            # print(f'save {i} query_embed and ids to {save_path}')
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)

    @torch.no_grad()
    def _run_with_GT(self):
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 use_dab=self.use_dab,
                                 ).to(self.device)]

        txt_lines = []
        for i, ((image, ori_image), info, heatmap) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # image: (1, C, H, W); ori_image: (1, H, W, C)
            frame = tensor_list_to_nested_tensor([image[0]]).to(self.device)
            heatmap = heatmap.to(self.device)
            res = self.model(frame=frame, tracks=tracks, heatmap=heatmap)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)

            # We do not use this...
            # but I do not want to remove this part.
            # WHAT IF it breaks down!!!
            # of course not :)
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
            # box = [x, y, w, h]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            boxes_xyxyxyxy = rotate_norm_boxes_to_boxes(tracks_result.boxes.cpu(), (image.shape[2], image.shape[3]), version='le135')
            boxes_xyxyxyxy = obb2poly(boxes_xyxyxyxy)

            for _tracks, xyxyxyxy in zip(tracks_result, boxes_xyxyxyxy):
                save_format = '{frame:6d},{id:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n'
                x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy.tolist()
                obj_id = _tracks.ids.item()
                conf = torch.max(_tracks.scores, dim=-1).values.item()
                label = _tracks.labels.item()
                line = save_format.format(frame=i + 1, id=obj_id, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, conf=conf, label=label)
                txt_lines.append(line)

        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)
        return

    @torch.no_grad()
    def _run(self):
        tracks = [TrackInstances(hidden_dim=get_model(self.model).hidden_dim,
                                 num_classes=get_model(self.model).num_classes,
                                 use_dab=self.use_dab,
                                 ).to(self.device)]
        # bdd100k_results = []    # for bdd100k, will be converted into json file, different from other datasets.

        txt_lines = []
        for i, ((image, ori_image), info) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # image: (1, C, H, W); ori_image: (1, H, W, C)
            frame = tensor_list_to_nested_tensor([image[0]]).to(self.device)
            res = self.model(frame=frame, tracks=tracks)
            previous_tracks, new_tracks = self.tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = get_model(self.model).postprocess_single_frame(previous_tracks, new_tracks, None)

            # We do not use this...
            # but I do not want to remove this part.
            # WHAT IF it breaks down!!!
            # of course not :)
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
            # box = [x, y, w, h]
            tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                 tracks_result.boxes[:, 3] * ori_h
            tracks_result = self.filter_by_score(tracks_result, thresh=self.result_score_thresh)
            tracks_result = self.filter_by_area(tracks_result)
            # to xyxy:
            # tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
            # tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))

            # if self.dataset_name == "BDD100K":
            #     self.update_results(tracks_result=tracks_result, frame_idx=i, results=bdd100k_results, img_path=info[0])
            # else:
            boxes_xyxyxyxy = rotate_norm_boxes_to_boxes(tracks_result.boxes.cpu(), (image.shape[2], image.shape[3]), version='le135')
            boxes_xyxyxyxy = obb2poly(boxes_xyxyxyxy)

            for _tracks, xyxyxyxy in zip(tracks_result, boxes_xyxyxyxy):
                save_format = '{frame:6d},{id:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n'
                x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy.tolist()
                obj_id = _tracks.ids.item()
                conf = torch.max(_tracks.scores, dim=-1).values.item()
                label = _tracks.labels.item()
                line = save_format.format(frame=i + 1, id=obj_id, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, conf=conf, label=label)
                txt_lines.append(line)

            # save_path = os.path.join("/data3/litianhao/hsmot/paper/memotr3ch", self.seq_name)
            # os.makedirs(save_path, exist_ok=True)
            # #tracks[0].query_embed和tracks[0].ids保存下来
            # torch.save(tracks[0].query_embed.cpu(), os.path.join(save_path, f"{i}_query_embed.pt"))
            # torch.save(tracks[0].ids.cpu(), os.path.join(save_path, f"{i}_ids.pt"))
            # print(f'save {i} query_embed and ids to {save_path}')
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "w") as file:
            file.writelines(txt_lines)
            # if self.visualize:
            #     os.makedirs(f"./outputs/visualize_tmp/frame_{i+1}/", exist_ok=False)
            #     os.system(f"mv ./outputs/visualize_tmp/query_updater/ ./outputs/visualize_tmp/frame_{i+1}/")
            #     os.system(f"mv ./outputs/visualize_tmp/decoder/ ./outputs/visualize_tmp/frame_{i+1}/")
            #     os.system(f"mv ./outputs/visualize_tmp/memotr/ ./outputs/visualize_tmp/frame_{i+1}/")
            #     os.system(f"mv ./outputs/visualize_tmp/runtime_tracker/ ./outputs/visualize_tmp/frame_{i+1}/")

        # if self.visualize:
        #     visualize_save_dir = os.path.join("./outputs/visualize/", self.seq_name)
        #     os.makedirs(visualize_save_dir, exist_ok=True)
        #     os.system(f"mv ./outputs/visualize_tmp/* {visualize_save_dir}")

        # if self.dataset_name == "BDD100K":
        #     with open(os.path.join(self.predict_dir, '{}.json'.format(self.seq_name)), 'w', encoding='utf-8') as f:
        #         json.dump(bdd100k_results, f)

        return

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
        # Only be used for BDD100K:
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
                if "hsmot" in self.dataset_name:
                    x1
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


def submit(config: dict):
    submit_logger = Logger(logdir=os.path.join(config["SUBMIT_DIR"], config["SUBMIT_DATA_SPLIT"]), only_main=True)
    submit_logger.show(head="Configs:", log=config)
    submit_logger.write(log=config, filename="config.yaml", mode="w")

    assert config["SUBMIT_DIR"] is not None, f"'--submit-dir' must not be None for submit process."
    assert config["SUBMIT_MODEL"] is not None, f"'--submit-model' must not be None for submit process."
    assert config["SUBMIT_DATA_SPLIT"] is not None, f"'--submit-data-split' must not be None for submit process."
    train_config = yaml_to_dict(path=path.join(config["SUBMIT_DIR"], "train/config.yaml"))

    data_root = config["DATA_ROOT"]
    dataset_name = train_config["DATASET"]
    config["DATASET"] = dataset_name
    dataset_split = config["SUBMIT_DATA_SPLIT"]
    outputs_dir = path.join(config["SUBMIT_DIR"], dataset_split)
    use_dab = train_config["USE_DAB"]
    det_score_thresh = config["DET_SCORE_THRESH"]
    track_score_thresh = config["TRACK_SCORE_THRESH"]
    result_score_thresh = config["RESULT_SCORE_THRESH"]
    use_motion = config["USE_MOTION"]
    motion_min_length = config["MOTION_MIN_LENGTH"]
    motion_max_length = config["MOTION_MAX_LENGTH"]
    motion_lambda = config["MOTION_LAMBDA"]
    miss_tolerance = config["MISS_TOLERANCE"]

    model = build_model(config=train_config)
    load_checkpoint(
        model=model,
        path=path.join(config["SUBMIT_DIR"], config["SUBMIT_MODEL"])
    )
    if "hsmot" in dataset_name:
        data_split_dir = path.join(data_root, dataset_name.replace("_8ch",""), dataset_split, 'npy')
    # if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
    #     data_split_dir = path.join(data_root, dataset_name, dataset_split)
    # elif dataset_name == "BDD100K":
    #     data_split_dir = path.join(data_root, dataset_name, "images/track/", dataset_split)
    # else:
    #     data_split_dir = path.join(data_root, dataset_name, "images", dataset_split)
    seq_names = os.listdir(data_split_dir)

    if is_distributed():
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=False)
        total_seq_names = seq_names
        seq_names = []
        for i in range(len(total_seq_names)):
            if i % distributed_world_size() == distributed_rank():
                seq_names.append(total_seq_names[i])

    for seq_name in seq_names:
        seq_name = str(seq_name)
        submitter = Submitter(
            dataset_name=dataset_name,
            split_dir=data_split_dir,
            seq_name=seq_name,
            outputs_dir=outputs_dir,
            model=model,
            use_dab=use_dab,
            det_score_thresh=det_score_thresh,
            track_score_thresh=track_score_thresh,
            result_score_thresh=result_score_thresh,
            use_motion=use_motion,
            motion_min_length=motion_min_length,
            motion_max_length=motion_max_length,
            motion_lambda=motion_lambda,
            miss_tolerance=miss_tolerance,
            npy2rgb = config["NPY2RGB"]
        )
        submitter.run()
    return

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

    data_root = config["DATA_ROOT"]
    dataset_name = config["DATASET"]
    dataset_split = config["SUBMIT_DATA_SPLIT"]
    outputs_dir = path.join(submit_dir_epoch, dataset_split)
    use_dab = config["USE_DAB"]
    det_score_thresh = config["DET_SCORE_THRESH"]
    track_score_thresh = config["TRACK_SCORE_THRESH"]
    result_score_thresh = config["RESULT_SCORE_THRESH"]
    use_motion = config["USE_MOTION"]
    motion_min_length = config["MOTION_MIN_LENGTH"]
    motion_max_length = config["MOTION_MAX_LENGTH"]
    motion_lambda = config["MOTION_LAMBDA"]
    miss_tolerance = config["MISS_TOLERANCE"]
    dataset_version = config["DATASET_VERSION"]
    
    # 构造带 version 的数据集根目录
    if "hsmot" in dataset_name:
        dataset_root = path.join(data_root, dataset_name.replace("_8ch", ""))
        if dataset_version is not None:
            dataset_root = path.join(dataset_root, dataset_version)
        data_split_dir = path.join(dataset_root, dataset_split, 'npy')
    seq_names = os.listdir(data_split_dir)

    if is_distributed():
        total_seq_names = seq_names
        seq_names = []
        for i in range(len(total_seq_names)):
            if i % distributed_world_size() == distributed_rank():
                seq_names.append(total_seq_names[i])

    for seq_name in seq_names:
        seq_name = str(seq_name)
        submitter = Submitter(
            dataset_name=dataset_name,
            split_dir=data_split_dir,
            seq_name=seq_name,
            outputs_dir=outputs_dir,
            model=model,
            use_dab=use_dab,
            det_score_thresh=det_score_thresh,
            track_score_thresh=track_score_thresh,
            result_score_thresh=result_score_thresh,
            use_motion=use_motion,
            motion_min_length=motion_min_length,
            motion_max_length=motion_max_length,
            motion_lambda=motion_lambda,
            miss_tolerance=miss_tolerance,
            npy2rgb = config["NPY2RGB"],
            decoder_spectral= config["DECODER_SPECTRAL"],
            use_scem_gt=config["SCEM"]["USE_GT"],
            only_train_detr=only_train_detr,
            epoch=epoch,
            draw_pic_dir=draw_pic_dir,
        )
        submitter.run()


    if is_distributed():
        torch.distributed.barrier()

    if distributed_rank() == 0:
        # 评估阶段同样复用 dataset_root

        gt_dir = os.path.join(dataset_root, dataset_split, 'mot')
        img_dir = os.path.join(dataset_root, dataset_split, 'npy')

        tracker_dir = submit_dir_epoch
        trackers_name = outputs_dir.split('/')[-1]
        trackers_subfolder = 'tracker'
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        if only_train_detr:
            val_lines = val_folder(gt_folder=gt_dir, pred_folder=os.path.join(tracker_dir, trackers_name, trackers_subfolder))
            submit_logger.show(head="Validation Results:", log='\n'.join(val_lines))
            submit_logger.write(head="Validation Results:", log='\n'.join(val_lines), filename="log.txt", mode="a")
            if train_logger is not None:
                train_logger.write(head="Validation Results:", log='\n'.join(val_lines), filename="log.txt", mode="a")
        else:
            os_flag = os.system(
                f"{sys.executable} {current_file_dir}/../TrackEval/scripts/run_hsmot_8ch.py " 
                f"--USE_PARALLEL False "
                f"--METRICS HOTA CLEAR Identity " 
                f"--GT_FOLDER {gt_dir} "
                f"--TRACKERS_FOLDER {tracker_dir} "
                f"--TRACKERS_TO_EVAL {trackers_name} "
                f"--TRACKER_SUB_FOLDER {trackers_subfolder} "
                f"--IMG_FOLDER {img_dir} "
            )
            assert os_flag == 0, "TrackEval failed to run."

    if is_distributed():
        torch.distributed.barrier()
    return