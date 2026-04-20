# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import copy
import math
import re

import torch.nn.functional as F
import torch.distributed

from typing import List, Tuple, Dict

from .matcher import build as build_matcher, HungarianMatcher, pairwise_min_permuted_segment_loss
from .loss.eql_lossV2_nobg import EQLv2NoBg
from .loss.efl_loss import EqualizedFocalLoss
from .model_output_accessors import (
    get_last_layer_input_query,
    get_last_layer_input_ref,
    get_last_layer_output_query,
)
from structures.track_instances import TrackInstances
from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy, box_iou_union
from utils.utils import is_distributed, distributed_world_size

from hsmot.loss.loss import l1_loss_rotate, loss_rotated_iou_norm_bboxes1
from hsmot.util.dist import box_iou_rotated_norm_bboxes1
from utils.edge_swap import EdgeSwap

class ClipCriterion:
    def __init__(self, num_classes, matcher: HungarianMatcher, n_det_queries, aux_loss: bool, weight: dict,
                 max_frame_length: int, n_aux: int, merge_det_track_layer: int = 0, aux_weights: List = None,
                 hidden_dim: int = 256, use_dab: bool = True, kl_cos_scheduler_epoch: int = 10, kl_weight_eta = 0, decoder_spectral :bool = False, scem: bool = False, loss_nll_config:dict = None, edge_swap: bool = False, label_loss_type: str = "sigmoid_focal_loss", eql_loss_config: dict | None = None, efl_loss_config: dict | None = None, num_decoder_layers: int = 6):
        """
        Init a criterion function.

        Args:
            num_classes: class num.
            matcher: matcher from DETR.
            n_det_queries: how many detection queries.
            aux_loss: whether use aux loss.
            weight: include "box_l1_loss", "box_giou_loss", "label_focal_loss"
        """
        self.device: None | torch.device = None
        self.aux_loss = aux_loss
        self.weight = weight
        self.num_classes = num_classes
        self.matcher = matcher
        self.n_det_queries = n_det_queries
        self.max_frame_length = max_frame_length
        self.n_aux = n_aux
        self.use_dab = use_dab
        self.frame_weights = [1.0] * self.max_frame_length  # if you want to set different weights for different frames
        self.aux_weights = aux_weights                      # different weights for different DETR layers
        self.hidden_dim = hidden_dim
        self.merge_det_track_layer = merge_det_track_layer
        self.edge_swap = edge_swap
        self.gt_trackinstances_list: None | List[List[TrackInstances]] = None     # (clip_size, B)
        self.target_list: None | List[List[Dict]] = None
        self.loss = {}
        self.log = {}
        self.n_gts = []
        self.epoch = 0
        self.kl_cos_scheduler_epoch = kl_cos_scheduler_epoch
        self.kl_weight_eta = kl_weight_eta
        self.decoder_spectral_mse = decoder_spectral
        self.scem = scem
        self.loss_nll_config = loss_nll_config
        self.label_loss_type = label_loss_type.lower()
        self.eql_loss_config = eql_loss_config or {}
        self.efl_loss_config = efl_loss_config or {}
        self.num_decoder_layers = num_decoder_layers
        self.eqlv2_nobg_loss: EQLv2NoBg | None = None
        self.efl_loss: EqualizedFocalLoss | None = None

        if self.label_loss_type == "eql_lossv2_nobg":
            self.eqlv2_nobg_loss = EQLv2NoBg(
                num_classes=self.num_classes,
                loss_weight=1.0,
                gamma=self.eql_loss_config.get("GAMMA", 12),
                mu=self.eql_loss_config.get("MU", 0.8),
                alpha=self.eql_loss_config.get("ALPHA", 4.0),
            )
        elif self.label_loss_type == "efl_loss":
            if (not self.aux_loss) and self.num_decoder_layers > 1:
                raise ValueError(
                    "LOSS_LABEL_TYPE='efl_loss' requires AUX_LOSS=True when NUM_DEC_LAYERS>1, "
                    "otherwise EFL gradient hooks and label loss calls are inconsistent."
                )
            self.efl_loss = EqualizedFocalLoss(
                reduction='mean',
                loss_weight=1.0,
                ignore_index=self.efl_loss_config.get("IGNORE_INDEX", -2),
                num_classes=self.num_classes,
                focal_gamma=self.efl_loss_config.get("FOCAL_GAMMA", 2.0),
                focal_alpha=self.efl_loss_config.get("FOCAL_ALPHA", 0.25),
                scale_factor=self.efl_loss_config.get("SCALE_FACTOR", 8.0),
                num_decoder_layers=self.num_decoder_layers,
                eps=self.efl_loss_config.get("EPS", 1e-8),
            )
        elif self.label_loss_type != "sigmoid_focal_loss":
            raise ValueError(
                f"Unsupported LOSS_LABEL_TYPE '{label_loss_type}', only support "
                f"'sigmoid_focal_loss', 'eql_lossV2_nobg' and 'efl_loss'."
            )

    def set_epoch(self, epoch: int):
        '''
            设置一些随epoch变化的损失权重
        '''
        self.epoch = epoch
        if self.epoch >= self.kl_cos_scheduler_epoch:
            self.weight["spectral_kl_loss"] = self.kl_weight_eta
        else:
            coeff = math.cos(math.pi * self.epoch / self.kl_cos_scheduler_epoch)
            self.weight["spectral_kl_loss"] = self.weight["spectral_kl_loss"] * coeff
        print(f"spectral_kl_loss weight: {self.weight['spectral_kl_loss']}")

    def set_device(self, device: torch.device):
        self.device = device
        if self.eqlv2_nobg_loss is not None:
            self.eqlv2_nobg_loss = self.eqlv2_nobg_loss.to(self.device)
        if self.efl_loss is not None:
            self.efl_loss = self.efl_loss.to(self.device)

    def init_a_clip(self, batch: Dict, hidden_dim: int, num_classes: int, device: torch.device):
        """
        Init this function for a specific clip.
        Args:
            batch: a batch data.
            hidden_dim:
            num_classes:
            device:
        Returns:
        """
        self.device = device
        clip_size = len(batch["imgs"][0])
        batch_size = len(batch["imgs"])
        self.gt_trackinstances_list = []
        self.target_list = []
        for c in range(clip_size):
            gt_trackinstances = TrackInstances.init_tracks(batch, hidden_dim=hidden_dim,
                                                           num_classes=num_classes, device=self.device)
            target_list = []
            for b in range(batch_size):
                gt_trackinstances[b].ids = batch["infos"][b][c]["obj_ids"]
                gt_trackinstances[b].labels = batch["infos"][b][c]["labels"]
                gt_trackinstances[b].boxes = batch["infos"][b][c]["boxes"]
                gt_trackinstances[b].norm_boxes = batch["infos"][b][c]["norm_boxes"]
                gt_trackinstances[b].pred_spectral_weights = batch["infos"][b][c]["spectral_weights"]
                # gt_trackinstances[b].heatmap = batch["infos"][b][c]["heatmap"]
                gt_trackinstances[b] = gt_trackinstances[b].to(self.device)
                
                if 'heatmap' in batch["infos"][b][c]:
                    _target = {
                        "heatmap" : batch["infos"][b][c]["heatmap"].to(self.device)
                    }
                    target_list.append(_target)

            self.target_list.append(target_list)
            self.gt_trackinstances_list.append(gt_trackinstances)

        self.n_gts = []
        if self.aux_loss:
            self.loss = {
                "box_l1_loss": torch.zeros(()).to(self.device),
                "box_giou_loss": torch.zeros(()).to(self.device),
                "label_focal_loss": torch.zeros(()).to(self.device),
                "aux_box_l1_loss": torch.zeros(()).to(self.device),
                "aux_box_giou_loss": torch.zeros(()).to(self.device),
                "aux_label_focal_loss": torch.zeros(()).to(self.device),
                "spectral_kl_loss": torch.zeros(()).to(self.device),
            }
            if self.decoder_spectral_mse:
                # "spectral_decoder_mse_loss": torch.zeros(()).to(self.device),
                # "aux_spectral_decoder_mse_loss": torch.zeros(()).to(self.device)
                self.loss["spectral_decoder_mse_loss"] = torch.zeros(()).to(self.device)
                self.loss["aux_spectral_decoder_mse_loss"] = torch.zeros(()).to(self.device)
        else:
            self.loss = {
                "box_l1_loss": torch.zeros(()).to(self.device),
                "box_giou_loss": torch.zeros(()).to(self.device),
                "label_focal_loss": torch.zeros(()).to(self.device),
                "spectral_kl_loss": torch.zeros(()).to(self.device),
            }
            if self.decoder_spectral_mse:
                self.loss["spectral_decoder_mse_loss"] = torch.zeros(()).to(self.device)

        if self.scem:
            self.loss["scem_nll_loss"] = torch.zeros(()).to(self.device)
            self.loss["scem_bce_loss"] = torch.zeros(()).to(self.device)
            self.loss["scem_dice_loss"] = torch.zeros(()).to(self.device)
            self.loss["scem_pool_div_loss"] = torch.zeros(()).to(self.device)
            self.loss["scem_gamma_cover_loss"] = torch.zeros(()).to(self.device)

        return

    def get_sum_loss_dict(self, loss_dict: dict, log_dict: dict):
        
        '''
            把每个损失乘上权重
        '''
        def get_weight(loss_name):
            if "box_l1_loss" in loss_name:
                return self.weight["box_l1_loss"]
            elif "box_giou_loss" in loss_name:
                return self.weight["box_giou_loss"]
            elif "label_focal_loss" in loss_name:
                return self.weight["label_focal_loss"]
            elif "spectral_kl_loss" in loss_name:
                return self.weight["spectral_kl_loss"]
            elif "spectral_decoder_mse_loss" in loss_name:
                return self.weight["spectral_decoder_mse_loss"]
            elif "scem_nll_loss" in loss_name:
                return self.weight["scem_nll_loss"]
            elif "scem_bce_loss" in loss_name:
                return self.weight["scem_bce_loss"]
            elif "scem_dice_loss" in loss_name:
                return self.weight["scem_dice_loss"]
            elif "scem_pool_div_loss" in loss_name:
                return self.weight["scem_pool_div_loss"]
            elif "scem_gamma_cover_loss" in loss_name:
                return self.weight["scem_gamma_cover_loss"]
            return 0.0

        def _frame_log_pattern_for_loss_key(loss_key: str):
            """与 process_single_frame 里 self.log 的命名规则一致。"""
            if "spectral_kl" in loss_key:
                return re.compile(r"^frame\d+_spectral_kl_loss$")
            if "scem_" in loss_key:
                return re.compile(rf"^frame\d+_{re.escape(loss_key)}$")
            if "spectral_decoder_mse" in loss_key:
                return re.compile(r"^frame\d+_(?:aux_layer\d+_)?spectral_decoder_mse_loss$")
            if "box_l1" in loss_key:
                return re.compile(r"^frame\d+_(?:aux_layer\d+_)?box_l1_loss(?:_class_\d+)?$")
            if "box_giou" in loss_key:
                return re.compile(r"^frame\d+_(?:aux_layer\d+_)?box_giou_loss(?:_class_\d+)?$")
            if "label_focal" in loss_key:
                return re.compile(r"^frame\d+_(?:aux_layer\d+_)?label_focal_loss$")
            return None

        # 任一损失权重为 0 时，从 loss_dict / log_dict 中去掉该项（不参与加权求和、日志也不显示）
        for lk in list(loss_dict.keys()):
            if get_weight(lk) != 0:
                continue
            loss_dict.pop(lk, None)
            pat = _frame_log_pattern_for_loss_key(lk)
            if pat is None:
                continue
            for gk in list(log_dict.keys()):
                if pat.match(gk):
                    log_dict.pop(gk, None)

        loss = sum([
            get_weight(k) * v for k, v in loss_dict.items()
        ])

        return loss, log_dict

    def get_mean_by_n_gts(self) -> Tuple[Dict, Dict]:
        '''
            把所有帧的损失加到一起，除以总的gt数，得到平均损失
            scem损失不需要除以总的gt数
        '''
        total_n_gts = sum(self.n_gts)
        total_n_gts = torch.as_tensor(total_n_gts, dtype=torch.float, device=self.device)
        n_gts = torch.as_tensor(self.n_gts, dtype=torch.float, device=self.device)
        if is_distributed():
            torch.distributed.all_reduce(total_n_gts)
            torch.distributed.all_reduce(n_gts)
        total_n_gts = torch.clamp(total_n_gts / distributed_world_size(), min=1).item()
        n_gts = torch.clamp(n_gts / distributed_world_size(), min=1).tolist()
        loss = {}
        for k in self.loss:
            # scem损失不需要除以总的gt数
            if "scem" not in k:
                loss[k] = self.loss[k] / total_n_gts
            else:
                loss[k] = self.loss[k]
        log = {}
        for k in self.log:
            for i in range(len(n_gts)):
                if f"frame{i}" in k:
                    if "scem" not in k:
                        # 对于按类统计的损失（*_class_*），self.log[k] 已经是 per-class mean，
                        # 不再按总 GT 数做归一化；其它损失仍按该帧总 GT 数归一化。
                        if "_class_" in k:
                            log[k] = (self.log[k], 1)
                        else:
                            log[k] = (self.log[k] / n_gts[i], 1)
                    else:
                        log[k] = (self.log[k], 1)
                    break
        return loss, log

    def process_single_frame(self, model_outputs: dict, tracked_instances: List[TrackInstances], frame_idx: int, img_metas):
        """
        Process this criterion for a single frame.

        I know this part is really complex and hard to understand (T.T),
        I will modify these in a possible extension version of this work in the future,
        but it works, doesn't it? :)
        Args:
            model_outputs: outputs from DETR.
            tracked_instances: already tracked instances.
            frame_idx: frame_idx t.
        """

        batch_size = len(tracked_instances)
        last_layer_input_query = get_last_layer_input_query(model_outputs=model_outputs)
        last_layer_output_query = get_last_layer_output_query(model_outputs=model_outputs)
        last_layer_input_ref = get_last_layer_input_ref(model_outputs=model_outputs)

        # 1. Get the GTs in current t frame.
        gt_trackinstances = self.gt_trackinstances_list[frame_idx]

        # 2. Update the already tracked instances. 
        # 更新上一帧的boxes logits output_embed pred_spectral_weights
        # 不更新query_embed和ref_pts和query_spectral_weights
        tracked_instances = self.update_tracked_instances(model_outputs=model_outputs,
                                                          tracked_instances=tracked_instances)

        # 3. Get the detection results in current frame.
        if self.decoder_spectral_mse:
            detection_res = {
                "pred_logits": model_outputs["pred_logits"][:, :self.n_det_queries, :].detach(),    # (B, Nd, n_classes)
                "pred_boxes": model_outputs["pred_bboxes"][:, :self.n_det_queries, :].detach(),      # (B, Nd, 4)
                    "pred_spectral_weights": model_outputs["pred_spectral_weights"][:, :self.n_det_queries, :].detach()      # (B, Nd, 8)
            }
        else:
            detection_res = {
                "pred_logits": model_outputs["pred_logits"][:, :self.n_det_queries, :].detach(),    # (B, Nd, n_classes)
                "pred_boxes": model_outputs["pred_bboxes"][:, :self.n_det_queries, :].detach(),      # (B, Nd, 4)
            }

        # 4. Find some gts that do not include in the tracked instances mentioned in (2.),
        #    this gts need to be detected in current frame.
        gt_ids_to_idx = []
        for b in range(batch_size):
            gt_ids_to_idx.append({
                gt_id.item(): gt_idx for gt_idx, gt_id in enumerate(gt_trackinstances[b].ids)
            })
        num_disappeared_tracked_gts = 0
        for b in range(batch_size):
            tracked_to_cur_gt_idx = []
            if len(tracked_instances[b]) > 0:
                for gt_id in tracked_instances[b].ids.tolist():
                    if gt_id in gt_ids_to_idx[b]:
                        tracked_to_cur_gt_idx.append(gt_ids_to_idx[b][gt_id])
                    else:
                        tracked_to_cur_gt_idx.append(-1)
                        num_disappeared_tracked_gts += 1
            #根据上一帧的id和这一帧的gt匹配情况，更新matched_idx
            tracked_instances[b].matched_idx = torch.as_tensor(data=tracked_to_cur_gt_idx,
                                                               dtype=tracked_instances[b].matched_idx.dtype)
        # 4.+ Filter the gts that not in the tracked instances:
        gt_full_idx = []
        untracked_gt_trackinstances = []
        for b in range(batch_size):
            gt_full_idx.append(
                torch.arange(start=0, end=len(gt_trackinstances[b]))
            )
        for b in range(batch_size):
            idx_bool = torch.ones(size=gt_full_idx[b].shape, dtype=torch.bool)
            for i in tracked_instances[b].matched_idx:
                if i.item() >= 0:
                    idx_bool[i.item()] = False
            untracked_gt_trackinstances.append(gt_trackinstances[b][idx_bool])

        # 5. Use Hungarian algorithm to matching.
        matcher_res = self.matcher(outputs=detection_res, targets=untracked_gt_trackinstances, use_focal=True, img_metas=img_metas)
        matcher_res = [list(mr) for mr in matcher_res]

        def matcher_res_for_gt_idx(res):
            for bi in range(len(res)):
                ids = untracked_gt_trackinstances[bi].ids[res[bi][1]]
                idx = []
                for _ in ids:   # 遍历 ID
                    idx.append(gt_ids_to_idx[bi][_.item()])
                res[bi][1] = torch.as_tensor(idx, dtype=torch.long)
            return res

        # 6. Use the matched results to generate the tracked instances.
        # 根据新匹配到的目标构建trackinstances
        new_trackinstances = []     # len is B
        for b in range(batch_size):
            trackinstances = TrackInstances(frame_height=tracked_instances[b].frame_height,
                                            frame_width=tracked_instances[b].frame_width,
                                            hidden_dim=tracked_instances[b].hidden_dim,
                                            num_classes=self.num_classes)
            output_idx, gt_idx = matcher_res[b]
            gt_ids = untracked_gt_trackinstances[b].ids[gt_idx]
            gt_idx = torch.as_tensor([gt_ids_to_idx[b][gt_id.item()] for gt_id in gt_ids], dtype=torch.long)
            trackinstances.ids = gt_ids
            trackinstances.matched_idx = gt_idx
            if self.use_dab:
                trackinstances.query_embed = last_layer_input_query[b][output_idx]
            else:
                raise NotImplementedError("Not Support for no DAB.")
            trackinstances.ref_pts = last_layer_input_ref[b][output_idx]
            trackinstances.output_embed = last_layer_output_query[b][output_idx]
            trackinstances.boxes = model_outputs["pred_bboxes"][b][output_idx]
            trackinstances.logits = model_outputs["pred_logits"][b][output_idx]
            trackinstances.iou = torch.zeros((len(gt_idx),), dtype=torch.float)
            if self.decoder_spectral_mse:
                trackinstances.pred_spectral_weights = model_outputs["pred_spectral_weights"][b][output_idx]
                trackinstances.query_spectral_weights = model_outputs["last_query_spectral_weights"][b][output_idx]# 最后一层的输入
            trackinstances = trackinstances.to(self.device)
            new_trackinstances.append(trackinstances)

        # 7. Add tracked instances to the matcher res, for loss computing.
        matcher_res = matcher_res_for_gt_idx(matcher_res)
        tracked_idx_to_gts_idx = []
        for b in range(batch_size):
            tracked_outputs_idx = torch.arange(start=self.n_det_queries,
                                               end=self.n_det_queries + len(tracked_instances[b]))
            tracked_gts_idx = tracked_instances[b].matched_idx
            tracked_idx_to_gts_idx.append([
                tracked_outputs_idx, tracked_gts_idx
            ])
            assert len(tracked_outputs_idx) == len(tracked_gts_idx)
        outputs_idx_to_gts_idx = copy.deepcopy(matcher_res)
        for b in range(batch_size):
            outputs_idx_to_gts_idx[b][0] = torch.cat((outputs_idx_to_gts_idx[b][0], tracked_idx_to_gts_idx[b][0]))
            outputs_idx_to_gts_idx[b][1] = torch.cat((outputs_idx_to_gts_idx[b][1], tracked_idx_to_gts_idx[b][1]))

        # 到此位置所有匹配工作结束
        # 8. Compute the classification loss.
        loss_label = self.get_loss_label(outputs=model_outputs,
                                         gt_trackinstances=gt_trackinstances,
                                         idx_to_gts_idx=outputs_idx_to_gts_idx)

        # 9. Compute the bounding box loss.
        loss_l1, loss_giou, loss_by_class = self.get_loss_box(outputs=model_outputs,
                                               gt_trackinstances=gt_trackinstances,
                                               idx_to_gts_idx=outputs_idx_to_gts_idx, img_metas=img_metas, edge_swap=self.edge_swap)

        if self.decoder_spectral_mse:
            # compute spectral decoder mse loss
            loss_spectral_decoder_mse = self.get_loss_spectral_decoder_mse(outputs=model_outputs,
                                                                        gt_trackinstances=gt_trackinstances,
                                                                        idx_to_gts_idx=outputs_idx_to_gts_idx)

        # 10. Count how many GTs.
        n_gts = sum([len(gts) for gts in gt_trackinstances])
        self.loss["box_l1_loss"] += loss_l1 * self.frame_weights[frame_idx]
        self.loss["box_giou_loss"] += loss_giou * self.frame_weights[frame_idx]
        self.loss["label_focal_loss"] += loss_label * self.frame_weights[frame_idx]
        if self.decoder_spectral_mse:
            self.loss["spectral_decoder_mse_loss"] += loss_spectral_decoder_mse * self.frame_weights[frame_idx]
        # Update logs.
        self.log[f"frame{frame_idx}_box_l1_loss"] = loss_l1.item()
        self.log[f"frame{frame_idx}_box_giou_loss"] = loss_giou.item()
        self.log[f"frame{frame_idx}_label_focal_loss"] = loss_label.item()
        if self.decoder_spectral_mse:
            self.log[f"frame{frame_idx}_spectral_decoder_mse_loss"] = loss_spectral_decoder_mse.item()
        
        # 记录按类别统计的损失
        for label, l1_loss_val in loss_by_class['loss_l1_by_class'].items():
            self.log[f"frame{frame_idx}_box_l1_loss_class_{label}"] = l1_loss_val
        for label, giou_loss_val in loss_by_class['loss_giou_by_class'].items():
            self.log[f"frame{frame_idx}_box_giou_loss_class_{label}"] = giou_loss_val
        
        self.n_gts.append(n_gts)

        # 11. Compute aux loss.
        if self.aux_loss:
            for i, aux_outputs in enumerate(model_outputs["aux_outputs"]):
                # Same to 3.
                if self.decoder_spectral_mse:
                    aux_det_res = {
                        "pred_logits": aux_outputs["pred_logits"][:, :self.n_det_queries, :].detach(),
                        "pred_boxes": aux_outputs["pred_bboxes"][:, :self.n_det_queries, :].detach(),
                        "pred_spectral_weights": aux_outputs["pred_spectral_weights"][:, :self.n_det_queries, :].detach()
                    }
                else:
                    aux_det_res = {
                        "pred_logits": aux_outputs["pred_logits"][:, :self.n_det_queries, :].detach(),
                        "pred_boxes": aux_outputs["pred_bboxes"][:, :self.n_det_queries, :].detach(),
                    }
                # Same to 5.
                if i < self.merge_det_track_layer:
                    aux_matcher_res = self.matcher(outputs=aux_det_res, targets=gt_trackinstances,
                                                   use_focal=True, img_metas=img_metas)
                    aux_matcher_res = [list(mr) for mr in aux_matcher_res]
                else:
                    aux_matcher_res = self.matcher(outputs=aux_det_res, targets=untracked_gt_trackinstances,
                                                   use_focal=True, img_metas=img_metas)
                    aux_matcher_res = [list(mr) for mr in aux_matcher_res]
                    aux_matcher_res = matcher_res_for_gt_idx(aux_matcher_res)
                # Same to some part in 7.
                aux_idx_to_gts_idx = copy.deepcopy(aux_matcher_res)
                for b in range(batch_size):
                    if i < self.merge_det_track_layer:
                        aux_idx_to_gts_idx[b][0] = aux_idx_to_gts_idx[b][0]
                        aux_idx_to_gts_idx[b][1] = aux_idx_to_gts_idx[b][1]
                    else:
                        aux_idx_to_gts_idx[b][0] = torch.cat((aux_idx_to_gts_idx[b][0], tracked_idx_to_gts_idx[b][0]))
                        aux_idx_to_gts_idx[b][1] = torch.cat((aux_idx_to_gts_idx[b][1], tracked_idx_to_gts_idx[b][1]))

                # Compute the aux loss.
                aux_loss_label = self.get_loss_label(outputs=model_outputs["aux_outputs"][i],
                                                     gt_trackinstances=gt_trackinstances,
                                                     idx_to_gts_idx=aux_idx_to_gts_idx)
                aux_loss_l1, aux_loss_giou, aux_loss_by_class = self.get_loss_box(outputs=model_outputs["aux_outputs"][i],
                                                               gt_trackinstances=gt_trackinstances,
                                                               idx_to_gts_idx=aux_idx_to_gts_idx, img_metas=img_metas, edge_swap=self.edge_swap)

                if self.decoder_spectral_mse:
                    aux_loss_spectral_decoder_mse = self.get_loss_spectral_decoder_mse(outputs=model_outputs["aux_outputs"][i], gt_trackinstances=gt_trackinstances, idx_to_gts_idx=aux_idx_to_gts_idx)

                self.loss["aux_box_l1_loss"] += aux_loss_l1 * self.frame_weights[frame_idx] * self.aux_weights[i]
                self.loss["aux_box_giou_loss"] += aux_loss_giou * self.frame_weights[frame_idx] * self.aux_weights[i]
                self.loss["aux_label_focal_loss"] += aux_loss_label * self.frame_weights[frame_idx] * self.aux_weights[i]
                if self.decoder_spectral_mse:
                    self.loss["aux_spectral_decoder_mse_loss"] += aux_loss_spectral_decoder_mse * self.frame_weights[frame_idx] * self.aux_weights[i]
                
                # 记录辅助损失的日志
                self.log[f"frame{frame_idx}_aux_layer{i}_box_l1_loss"] = aux_loss_l1.item()
                self.log[f"frame{frame_idx}_aux_layer{i}_box_giou_loss"] = aux_loss_giou.item()
                self.log[f"frame{frame_idx}_aux_layer{i}_label_focal_loss"] = aux_loss_label.item()
                if self.decoder_spectral_mse:
                    self.log[f"frame{frame_idx}_aux_layer{i}_spectral_decoder_mse_loss"] = aux_loss_spectral_decoder_mse.item()
                
                # 记录辅助损失按类别统计的损失
                for label, l1_loss_val in aux_loss_by_class['loss_l1_by_class'].items():
                    self.log[f"frame{frame_idx}_aux_layer{i}_box_l1_loss_class_{label}"] = l1_loss_val
                for label, giou_loss_val in aux_loss_by_class['loss_giou_by_class'].items():
                    self.log[f"frame{frame_idx}_aux_layer{i}_box_giou_loss_class_{label}"] = giou_loss_val

        # Prepare the unmatched detection results.
        unmatched_detections = []
        for b in range(batch_size):
            matched_indexes = set(outputs_idx_to_gts_idx[b][0].tolist())
            indexes = set([_ for _ in range(len(model_outputs["det_query_embed"]))])
            unmatched_indexes = list(indexes - matched_indexes)
            unmatched_indexes = torch.as_tensor(unmatched_indexes, dtype=torch.long)
            detections = TrackInstances(
                hidden_dim=last_layer_output_query.shape[-1],
                num_classes=model_outputs["pred_logits"].shape[-1]
            ).to(last_layer_output_query.device)
            detections.ref_pts = model_outputs["init_ref_pts"][b][unmatched_indexes]
            detections.output_embed = last_layer_output_query[b][unmatched_indexes]
            detections.logits = model_outputs["pred_logits"][b][unmatched_indexes]
            detections.boxes = model_outputs["pred_bboxes"][b][unmatched_indexes]
            if self.decoder_spectral_mse:
                detections.pred_spectral_weights = model_outputs["pred_spectral_weights"][b][unmatched_indexes]
                detections.query_spectral_weights = model_outputs["init_query_spectral_weights"][b][unmatched_indexes]
            if self.use_dab:
                detections.query_embed = last_layer_input_query[b][unmatched_indexes]
            else:
                raise NotImplementedError("Not Support for no DAB.")

            detections.ids = -torch.ones((len(detections.query_embed),), dtype=torch.long, device=self.device)
            detections.matched_idx = -torch.ones((len(detections.query_embed),), dtype=torch.long, device=self.device)
            detections.iou = torch.zeros((len(detections.ids,)), dtype=torch.float, device=self.device)
            unmatched_detections.append(detections)
            pass

        # Move to device.
        for b in range(batch_size):
            tracked_instances[b] = tracked_instances[b].to(self.device)
            new_trackinstances[b] = new_trackinstances[b].to(self.device)

        # Compute IoU.
        for b in range(batch_size):
            new_trackinstances[b].iou[new_trackinstances[b].matched_idx >= 0] =  box_iou_rotated_norm_bboxes1(
                new_trackinstances[b][new_trackinstances[b].matched_idx >= 0].boxes,
                gt_trackinstances[b][new_trackinstances[b][new_trackinstances[b].matched_idx >= 0].matched_idx].boxes,
                img_shape=img_metas['img_shape'], version=img_metas['version'], aligned=True
            )

            tracked_instances[b].iou[tracked_instances[b].matched_idx >= 0] = box_iou_rotated_norm_bboxes1(
                tracked_instances[b][tracked_instances[b].matched_idx >= 0].boxes,
                gt_trackinstances[b][tracked_instances[b][tracked_instances[b].matched_idx >= 0].matched_idx].boxes,
                img_shape=img_metas['img_shape'], version=img_metas['version'], aligned=True
            )

        # 12 calculate spectral kl loss
        spectral_weights_list = model_outputs["spectral_weights"]
        target_weights = spectral_weights_list[0]
        kl_loss = torch.zeros(()).to(self.device)
        for b in range(1, len(spectral_weights_list)):
            source_weights = spectral_weights_list[b]
            _target_weights = F.adaptive_avg_pool2d(target_weights, (source_weights.shape[2], source_weights.shape[3]))
            kl_loss += F.kl_div(F.log_softmax(_target_weights, dim=1), F.softmax(source_weights, dim=1), reduction="batchmean")

        self.loss["spectral_kl_loss"] += kl_loss * self.frame_weights[frame_idx]
        self.log[f"frame{frame_idx}_spectral_kl_loss"] = kl_loss.item()

        # 13 calculate scem loss
        if self.scem:
            gamma = model_outputs["scem_gamma"]
            log_mix = model_outputs["scem_log_mix"]

            heatmap = self.target_list[frame_idx][0]['heatmap'].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            #降尺度
            heatmap = F.adaptive_avg_pool2d(heatmap, (gamma.shape[2], gamma.shape[3]))

            scem_bce_loss = focal_bce_loss(gamma, heatmap)
            scem_dice_loss = dice_loss(gamma, heatmap)
            nll_type = self.loss_nll_config.get("TYPE", "mean")
            if nll_type == "mean":
                scem_nll_loss = -log_mix.mean()
            elif nll_type == "focal":
                alpha_pos = self.loss_nll_config.get("ALPHA_POS")
                alpha_neg = self.loss_nll_config.get("ALPHA_NEG")
                gamma_pos = self.loss_nll_config.get("GAMMA_POS")
                gamma_neg = self.loss_nll_config.get("GAMMA_NEG")
                scem_nll_loss = supervised_focal_nll_from_log_mix(log_mix, heatmap, alpha_pos, alpha_neg, gamma_pos, gamma_neg)

            self.loss["scem_bce_loss"] += scem_bce_loss * self.frame_weights[frame_idx]
            self.loss["scem_nll_loss"] += scem_nll_loss * self.frame_weights[frame_idx]
            self.loss["scem_dice_loss"] += scem_dice_loss * self.frame_weights[frame_idx]
            
            # log
            self.log[f"frame{frame_idx}_scem_bce_loss"] = scem_bce_loss.item()
            self.log[f"frame{frame_idx}_scem_nll_loss"] = scem_nll_loss.item()
            self.log[f"frame{frame_idx}_scem_dice_loss"] = scem_dice_loss.item()

            aux = model_outputs.get("scem_aux_losses")
            if isinstance(aux, dict):
                lp = aux.get("loss_pool_div")
                lg = aux.get("loss_gamma_cover")
                if torch.is_tensor(lp):
                    scem_pool_div_loss = lp
                    self.loss["scem_pool_div_loss"] += scem_pool_div_loss * self.frame_weights[frame_idx]
                    self.log[f"frame{frame_idx}_scem_pool_div_loss"] = scem_pool_div_loss.item()
                if torch.is_tensor(lg):
                    scem_gamma_cover_loss = lg
                    self.loss["scem_gamma_cover_loss"] += scem_gamma_cover_loss * self.frame_weights[frame_idx]
                    self.log[f"frame{frame_idx}_scem_gamma_cover_loss"] = scem_gamma_cover_loss.item()


        return tracked_instances, new_trackinstances, unmatched_detections

    def update_tracked_instances(self, model_outputs: dict, tracked_instances: List[TrackInstances])\
            -> List[TrackInstances]:
        """
        Update tracked instances.
        """
        last_layer_output_query = get_last_layer_output_query(model_outputs=model_outputs)
        for b in range(len(tracked_instances)):
            if len(tracked_instances[b]) > 0:
                track_mask = model_outputs["query_mask"][b][self.n_det_queries:]
                tracked_instances[b].boxes = model_outputs["pred_bboxes"][b][self.n_det_queries:][~track_mask]
                tracked_instances[b].logits = model_outputs["pred_logits"][b][self.n_det_queries:][~track_mask]
                # Query embed and ref_pts will be updated in the query_updater module.
                tracked_instances[b].output_embed = last_layer_output_query[b][self.n_det_queries:][~track_mask]
                tracked_instances[b].matched_idx = torch.zeros((0, ), dtype=tracked_instances[b].matched_idx.dtype)
                tracked_instances[b].labels = torch.zeros((0, ), dtype=tracked_instances[b].matched_idx.dtype)
                # query_spectral_weights update in query_updater
                if self.decoder_spectral_mse:
                    tracked_instances[b].pred_spectral_weights = model_outputs["pred_spectral_weights"][b][self.n_det_queries:][~track_mask]
        return tracked_instances

    def get_loss_label(self, outputs, gt_trackinstances: List[TrackInstances], idx_to_gts_idx):
        """
        Compute the classification loss.
        """
        pred_logits_per_batch = [
            preds[~mask] for preds, mask in zip(outputs["pred_logits"], outputs["query_mask"])
        ]
        gt_labels_per_batch = [
            torch.full((pred_logits_per_batch[b].shape[:1]),
                       self.num_classes,
                       dtype=torch.int64,
                       device=self.device) for b in range(len(gt_trackinstances))
        ]
        for b in range(len(pred_logits_per_batch)):
            gt_labels_per_batch[b][idx_to_gts_idx[b][0][idx_to_gts_idx[b][1] >= 0]] \
                = gt_trackinstances[b].labels[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]

        if self.label_loss_type == "efl_loss":
            if self.efl_loss is None:
                raise RuntimeError("EqualizedFocalLoss is not initialized.")
            pred_logits = outputs["pred_logits"]
            query_mask = outputs["query_mask"]
            B, Q, _ = pred_logits.shape
            gt_labels = torch.full(
                (B, Q),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            gt_labels[query_mask] = self.efl_loss.ignore_index

            for b in range(B):
                valid_pos = torch.nonzero(~query_mask[b], as_tuple=False).squeeze(1)
                matched = idx_to_gts_idx[b][1] >= 0
                if matched.any():
                    matched_outputs_filtered = idx_to_gts_idx[b][0][matched]
                    matched_outputs_raw = valid_pos[matched_outputs_filtered]
                    gt_labels[b][matched_outputs_raw] = gt_trackinstances[b].labels[idx_to_gts_idx[b][1][matched]]

            loss = self.efl_loss(pred_logits=pred_logits, target_classes=gt_labels, normalizer=1)#不要进行平均，会在外边统一除以GT进行
        elif self.label_loss_type == "eql_lossv2_nobg":
            if self.eqlv2_nobg_loss is None:
                raise RuntimeError("EQLv2NoBg loss is not initialized.")
            pred_logits = torch.cat(pred_logits_per_batch)
            gt_labels = torch.cat(gt_labels_per_batch)
            loss = self.eqlv2_nobg_loss(cls_score=pred_logits, label=gt_labels)
        else:
            pred_logits = torch.cat(pred_logits_per_batch)
            gt_labels = torch.cat(gt_labels_per_batch)
            gt_labels_one_hot = F.one_hot(gt_labels, self.num_classes+1)[:, :-1]\
                .to(pred_logits.dtype).to(pred_logits.device)

            loss = sigmoid_focal_loss(inputs=pred_logits,
                                      targets=gt_labels_one_hot,
                                      alpha=0.25,
                                      gamma=2)

        return loss

    @staticmethod
    def get_loss_box(outputs, gt_trackinstances: List[TrackInstances], idx_to_gts_idx, img_metas, edge_swap):
        """
        Computer the bounding box loss, l1 and giou.
        按类别统计损失。
        """
        matched_pred_boxes = [
            boxes[outputs_idx[0][outputs_idx[1] >= 0]]
            for boxes, outputs_idx in zip(outputs["pred_bboxes"], idx_to_gts_idx)
        ]
        gt_boxes = [
            gt_trackinstances[b].boxes[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]
            for b in range(len(gt_trackinstances))
        ]
        norm_gt_boxes = [
            gt_trackinstances[b].norm_boxes[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]
            for b in range(len(gt_trackinstances))
        ]
        # 获取对应的 gt_labels
        gt_labels = [
            gt_trackinstances[b].labels[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]
            for b in range(len(gt_trackinstances))
        ]
        
        matched_pred_boxes = torch.cat(matched_pred_boxes)
        if edge_swap:
            matched_pred_boxes = EdgeSwap.edge_swap(matched_pred_boxes, img_metas['version'], img_metas['img_shape'])
        gt_boxes = torch.cat(gt_boxes).to(matched_pred_boxes.device)
        norm_gt_boxes = torch.cat(norm_gt_boxes).to(matched_pred_boxes.device)
        gt_labels = torch.cat(gt_labels).to(matched_pred_boxes.device)


        # 
        h_img, w_img = img_metas['img_shape']
        min_img_shape = min(h_img, w_img)
        l1_weight = torch.as_tensor([w_img / min_img_shape, h_img / min_img_shape, w_img / min_img_shape, h_img / min_img_shape, 1.0], dtype=matched_pred_boxes.dtype, device=matched_pred_boxes.device)#[5,]

        loss_l1 = l1_loss_rotate(matched_pred_boxes, norm_gt_boxes, weight=l1_weight).sum()
        if(matched_pred_boxes.size(0) == 0):
            loss_giou = torch.zeros_like(loss_l1)
            loss_by_class = {
                'loss_l1_by_class': {},
                'loss_giou_by_class': {}
            }
        else:
            loss_giou = (1-loss_rotated_iou_norm_bboxes1(matched_pred_boxes,  gt_boxes, img_metas['img_shape'], img_metas['version'])).sum()
            
            # 按类别统计损失
            loss_by_class = {
                'loss_l1_by_class': {},
                'loss_giou_by_class': {}
            }
            
            # 计算每个样本的损失（不求和）
            loss_l1_per_sample = l1_loss_rotate(matched_pred_boxes, norm_gt_boxes, weight=l1_weight)  # [N, 5] or [N]
            if loss_l1_per_sample.dim() == 2:
                loss_l1_per_sample = loss_l1_per_sample.sum(dim=1)  # [N]
            
            ious = loss_rotated_iou_norm_bboxes1(matched_pred_boxes, gt_boxes, img_metas['img_shape'], img_metas['version'])
            loss_giou_per_sample = 1 - ious  # [N]
            
            # 获取所有类别
            unique_labels = torch.unique(gt_labels)
            
            # 按类别统计
            for label in unique_labels:
                label_mask = (gt_labels == label)
                if label_mask.sum() > 0:
                    # 计算每个类别内部的平均损失（per-class mean），而不是总和
                    loss_by_class['loss_l1_by_class'][label.item()] = loss_l1_per_sample[label_mask].mean().item()
                    loss_by_class['loss_giou_by_class'][label.item()] = loss_giou_per_sample[label_mask].mean().item()

        return loss_l1, loss_giou, loss_by_class


    @staticmethod
    def get_loss_spectral_decoder_mse(outputs, gt_trackinstances, idx_to_gts_idx):
        """
        Compute the spectral decoder mse loss.
        """
        matched_pred_spectral_weights = [
            spectral_weights[outputs_idx[0][outputs_idx[1] >= 0]]
            for spectral_weights, outputs_idx in zip(outputs["pred_spectral_weights"], idx_to_gts_idx)
        ]
        gt_spectral_weights = [
            gt_trackinstances[b].pred_spectral_weights[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]
            for b in range(len(gt_trackinstances))
        ]
        matched_pred_spectral_weights = torch.cat(matched_pred_spectral_weights)
        gt_spectral_weights = torch.cat(gt_spectral_weights).sigmoid()# 光谱强度归一化之后，sigmoid到[0,1]
        if(len(matched_pred_spectral_weights) == 0):
            loss_spectral_decoder_mse = outputs["pred_spectral_weights"].sum()*0.0
        else:
            loss_spectral_decoder_mse = pairwise_min_permuted_segment_loss(matched_pred_spectral_weights, gt_spectral_weights, reduction='mse', aggregate_segment='mean', aggregate_loss="sum") / matched_pred_spectral_weights.size(1)

        return loss_spectral_decoder_mse

    
def supervised_focal_nll_from_log_mix(
    log_mix: torch.Tensor,
    y_soft: torch.Tensor,
    alpha_pos: float = 1.0,
    alpha_neg: float = 0.1,
    gamma_pos: float = 1.0,
    gamma_neg: float = 0.0,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
):
    """
    log_mix : [B,1,H,W] = log(P_mix)
    y_soft  : [B,1,H,W] in [0,1]  (Gaussian soft label)
    """

    # 1) 计算 supervised 权重 w(y)
    #    w = alpha_pos * y^gamma_pos + alpha_neg * (1-y)^gamma_neg
    y = y_soft.clamp(0.0, 1.0)
    w_pos = (y + eps).pow(gamma_pos)
    w_neg = (1.0 - y + eps).pow(gamma_neg)
    w = alpha_pos * w_pos + alpha_neg * w_neg  # [B,1,H,W]

    # 2) NLL = - log P_mix
    #    加权后成为监督式 NLL
    loss_map = - w * log_mix  # log_mix <= 0 -> loss >= 0

    # 3) 可选：只在 valid 区域取平均
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        loss_valid = loss_map[~valid_mask]
        if loss_valid.numel() == 0:
            return log_mix.new_zeros(())
        return loss_valid.mean()
    else:
        return loss_map.mean()

def focal_bce_loss(pred, target, alpha=0.75, gamma=2.0):
    eps = 1e-6
    pred = pred.clamp(eps, 1-eps)
    pos_loss = -alpha * (1 - pred) ** gamma * target * torch.log(pred)
    neg_loss = -(1 - alpha) * pred ** gamma * (1 - target) * torch.log(1 - pred)
    return (pos_loss + neg_loss).mean()

def dice_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = (pred * pred).sum() + (target * target).sum()
    return 1 - (2 * inter + eps) / (union + eps)

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()   # 在类别上计算平均

def build(config: dict):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
        "hsmot_8ch": 8,
    }
    return ClipCriterion(
        num_classes=dataset_num_classes[config["DATASET"]],
        matcher=build_matcher(config=config),
        n_det_queries=config["NUM_DET_QUERIES"],
        aux_loss=config["AUX_LOSS"],
        weight={
            "box_l1_loss": config["LOSS_WEIGHT_L1"],
            "box_giou_loss": config["LOSS_WEIGHT_GIOU"],
            "label_focal_loss": config["LOSS_WEIGHT_FOCAL"],
            "spectral_kl_loss": config["LOSS_SPECTRAL_KL"],
            "spectral_decoder_mse_loss": config["LOSS_SPECTRAL_DECODER_MSE"],
            "scem_nll_loss": config["LOSS_SCEM_NLL"],
            "scem_bce_loss": config["LOSS_SCEM_BCE"],
            "scem_dice_loss": config["LOSS_SCEM_DICE"],
            "scem_pool_div_loss": config.get("LOSS_SCEM_POOL_DIV"),
            "scem_gamma_cover_loss": config.get("LOSS_SCEM_GAMMA_COVER"),
        },
        max_frame_length=max(config["SAMPLE_LENGTHS"]),
        n_aux=config["NUM_DEC_LAYERS"]-1,
        merge_det_track_layer=(0 if "MERGE_DET_TRACK_LAYER" not in config else config["MERGE_DET_TRACK_LAYER"]),
        aux_weights=config["AUX_LOSS_WEIGHT"],
        hidden_dim=config["HIDDEN_DIM"],
        use_dab=config["USE_DAB"],
        decoder_spectral=config["DECODER_SPECTRAL"],
        scem = config["SCEM"]["ENABLE"],
        loss_nll_config=config["LOSS_NLL_CONFIG"],
        label_loss_type=config.get("LOSS_LABEL_TYPE", "sigmoid_focal_loss"),
        eql_loss_config=config.get("LOSS_LABEL_EQLV2_NOBG", {}),
        efl_loss_config=config.get("LOSS_LABEL_EFL", {}),
        num_decoder_layers=config["NUM_DEC_LAYERS"],
    )
