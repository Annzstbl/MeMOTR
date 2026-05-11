# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import math
import torch
import torch.nn as nn

from typing import List
from ..utils import pos_to_pos_embed_rotated, logits_to_scores
from torch.utils.checkpoint import checkpoint

from ..ffn import FFN
from ..mlp import MLP
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union


class QueryUpdater(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int,
                 tp_drop_ratio: float, fp_insert_ratio: float,
                 dropout: float,
                 use_checkpoint: bool, use_dab: bool,
                 update_threshold: float, long_memory_lambda: float,
                 track_iou_threshold: float = 0.5,
                 track_iou_adaptive: bool = True,
                 track_iou_threshold_max: float = 0.5,
                 track_iou_area_min: float = 800.0,
                 track_iou_area_max: float = 4000.0,
                 q_spec_lambda: float = 0,
                 visualize: bool = False,
                 ):
        super(QueryUpdater, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize

        self.update_threshold = update_threshold
        self.long_memory_lambda = long_memory_lambda
        self.track_iou_threshold = track_iou_threshold
        self.track_iou_adaptive = track_iou_adaptive
        self.track_iou_threshold_max = track_iou_threshold_max
        self.track_iou_area_min = track_iou_area_min
        self.track_iou_area_max = track_iou_area_max
        self.q_spec_lambda = q_spec_lambda

        self.confidence_weight_net = nn.Sequential(
            MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, num_layers=2),
            nn.Sigmoid()
        )
        self.short_memory_fusion = MLP(input_dim=2*self.hidden_dim, hidden_dim=2*self.hidden_dim,
                                       output_dim=self.hidden_dim, num_layers=2)
        self.memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_pos_head = MLP(
            input_dim=self.hidden_dim*2 + 2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )
        self.q_spec_residual_mlp = MLP(
            input_dim=4 * self.hidden_dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2,
        )
        self.q_spec_gate_mlp = MLP(
            input_dim=4 * self.hidden_dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2,
        )
        self.q_spec_update_norm = nn.LayerNorm(self.hidden_dim)
        # self.query_spectral_head = MLP(
        #     input_dim=8,
        #     hidden_dim=self.hidden_dim,
        #     output_dim=self.hidden_dim,
        #     num_layers=2
        # )

        if self.use_dab is False:   # D-DETR, use this module to update the
            self.linear_pos1 = nn.Linear(256, 256)
            self.linear_pos2 = nn.Linear(256, 256)
            self.norm_pos = nn.LayerNorm(256)
            self.activation = nn.ReLU(inplace=True)
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self._init_q_spec_updater_parameters()

    def _init_q_spec_updater_parameters(self):
        residual_last = self.q_spec_residual_mlp.layers[-1]
        nn.init.zeros_(residual_last.weight)
        nn.init.zeros_(residual_last.bias)

        gate_last = self.q_spec_gate_mlp.layers[-1]
        if self.q_spec_lambda > 0.0:
            init_p = min(max(float(self.q_spec_lambda), 1e-4), 1.0 - 1e-4)
            gate_bias = math.log(init_p / (1.0 - init_p))
        else:
            gate_bias = -3.0
        nn.init.zeros_(gate_last.weight)
        nn.init.constant_(gate_last.bias, gate_bias)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                no_augment: bool = False,
                img_metas=None):
        # scores> update_threshold 或者 active_tracks.ids>=0的会列为active_tracks
        # 不active的会直接消失
        # IoU低于尺度自适应阈值的active_tracks会将track_id=-1
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment, img_metas=img_metas)
        tracks = self.update_tracks_embedding(tracks=tracks)

        return tracks

    def update_tracks_embedding(self, tracks: List[TrackInstances]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold #之前在选择active的时候按照 scores>update_threshold 或者 id>0来筛选，这里选择scores足够高的部分进行更新
            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())

            if TrackInstances.use_spectral_decoder:
                tracks[b].query_spectral_weights[is_pos] = tracks[b][is_pos].pred_spectral_weights.detach().clone()

            if TrackInstances.use_q_spec:
                obs_q_spec = getattr(tracks[b], "obs_q_spec", None)
                prev_q_spec = getattr(tracks[b], "query_q_spec", None)
                if (
                    obs_q_spec is not None
                    and prev_q_spec is not None
                    and obs_q_spec.shape == prev_q_spec.shape
                    and obs_q_spec.numel() > 0
                ):
                    obs_q_spec = obs_q_spec.detach()
                    pair = torch.cat(
                        [
                            prev_q_spec,
                            obs_q_spec,
                            obs_q_spec - prev_q_spec,
                            obs_q_spec * prev_q_spec,
                            scores.unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    residual = self.q_spec_residual_mlp(pair)
                    candidate = obs_q_spec + residual
                    gate = torch.sigmoid(self.q_spec_gate_mlp(pair))
                    updated_q_spec = (1.0 - gate) * prev_q_spec + gate * candidate
                    updated_q_spec = self.q_spec_update_norm(updated_q_spec)
                    tracks[b].query_q_spec = torch.where(
                        is_pos.unsqueeze(-1),
                        updated_q_spec,
                        prev_q_spec,
                    )


            output_embed = tracks[b].output_embed
            last_output_embed = tracks[b].last_output
            long_memory = tracks[b].long_memory.detach()
            query_pos = pos_to_pos_embed_rotated(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
            # query_spectral = self.query_spectral_head(tracks[b].spectral_weights.sigmoid())
            
            # Confidence Weight
            confidence_weight = self.confidence_weight_net(output_embed)

            # Adaptive Aggregation
            short_memory = self.short_memory_fusion(
                torch.cat((
                    confidence_weight * output_embed,
                    last_output_embed
                ), dim=-1)
            )

            # Query Feature Generate
            query_pos = self.query_pos_head(query_pos)
            # q = short_memory + query_pos + query_spectral
            # k = long_memory + query_pos + query_spectral
            q = short_memory + query_pos
            k = long_memory + query_pos
            tgt = output_embed
            
            # Attention
            tgt2 = self.memory_attn(q[None, :], k[None, :], tgt[None, :])[0][0, :]
            tgt = tgt + self.memory_dropout(tgt2)
            tgt = self.memory_norm(tgt)
            tgt = self.memory_ffn(tgt)

            # Long Memory ResNet
            query_feat = long_memory + self.query_feat_dropout(tgt)
            query_feat = self.query_feat_norm(query_feat)
            query_feat = self.query_feat_ffn(query_feat)

            # Update Long Memory
            long_memory = (1 - self.long_memory_lambda) * long_memory + \
                          self.long_memory_lambda * tracks[b].output_embed
            tracks[b].long_memory = tracks[b].long_memory * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    long_memory * is_pos.reshape((is_pos.shape[0], 1))
            # Update Last Outputs Embedding
            tracks[b].last_output = tracks[b].last_output * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    output_embed * is_pos.reshape((is_pos.shape[0], 1))

            if self.use_dab:
                tracks[b].query_embed[is_pos] = query_feat[is_pos]
            else:
                raise NotImplementedError("Not Support for no DAB.")
                tracks[b].query_embed[:, self.hidden_dim:][is_pos] = query_feat[is_pos]
                # Update query pos, which is not appeared in DAB-D-DETR framework:
                new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(output_embed)))
                query_pos = tracks[b].query_embed[:, :self.hidden_dim]
                query_pos = query_pos + new_query_pos
                query_pos = self.norm_pos(query_pos)
                tracks[b].query_embed[:, :self.hidden_dim][is_pos] = query_pos[is_pos]
        return tracks

    def _init_track_memory_fields(self, track_instances: TrackInstances):
        track_instances.last_output = track_instances.output_embed
        if self.use_dab:
            track_instances.long_memory = track_instances.query_embed
        else:
            track_instances.long_memory = track_instances.query_embed[:, self.hidden_dim:]
        if TrackInstances.use_q_spec:
            # obs_q_spec在criterion.py中会统一更新
            track_instances.query_q_spec = track_instances.obs_q_spec.detach().clone()

    @staticmethod
    def _get_img_area(img_metas, batch_idx: int = 0) -> float:
        if isinstance(img_metas, (list, tuple)):
            img_metas = img_metas[batch_idx]
        if isinstance(img_metas, dict) and "transform_metas" in img_metas:
            img_metas = img_metas["transform_metas"]
        if hasattr(img_metas, "data"):
            img_metas = img_metas.data
        if not isinstance(img_metas, dict) or "img_shape" not in img_metas:
            raise ValueError("img_metas must provide 'img_shape' to compute scale-adaptive IoU threshold.")

        h_img, w_img = img_metas["img_shape"][:2]
        if torch.is_tensor(h_img):
            h_img = h_img.item()
        if torch.is_tensor(w_img):
            w_img = w_img.item()
        return float(h_img) * float(w_img)

    def _get_scale_adaptive_iou_threshold(self, track_instances: TrackInstances, img_metas, batch_idx: int = 0) -> torch.Tensor:
        """Use the current image size to relax IoU gates for small objects."""
        boxes = track_instances.boxes
        if boxes.numel() == 0:
            return boxes.new_zeros((0,))

        img_area = self._get_img_area(img_metas=img_metas, batch_idx=batch_idx)
        box_area = boxes[:, 2].clamp(min=0) * boxes[:, 3].clamp(min=0) * img_area
        area_span = max(self.track_iou_area_max - self.track_iou_area_min, 1e-6)
        scale_ratio = ((box_area - self.track_iou_area_min) / area_span).clamp(0.0, 1.0)
        return self.track_iou_threshold + (
            self.track_iou_threshold_max - self.track_iou_threshold
        ) * scale_ratio

    def _resolve_iou_threshold(
            self,
            track_instances: TrackInstances,
            img_metas=None,
            batch_idx: int = 0,
    ) -> torch.Tensor | float:
        if not self.track_iou_adaptive:
            return self.track_iou_threshold
        return self._get_scale_adaptive_iou_threshold(track_instances, img_metas=img_metas, batch_idx=batch_idx)


    def _select_active_tracks_no_aug(self, previous_tracks: TrackInstances,
                                     new_tracks: TrackInstances,
                                     unmatched_dets: TrackInstances,
                                     img_metas,
                                     batch_idx: int = 0) -> TrackInstances:
        """Select active tracks for training without augmentation.

        Pipeline:
        1) Merge `previous_tracks` and `new_tracks`.
        2) Append `unmatched_dets` (their ids are -1 by design, used as hard negatives).
        3) Keep tracks that are confident (`score > update_threshold`) OR already have valid ids (`id >= 0`).
        4) Apply a scale-adaptive IoU gate: tracks below the dynamic threshold are marked as invalid.

        Why set `id = -1`:
        - It marks a track as inactive/unreliable identity in current frame.
        - Downstream logic treats negative ids as non-confirmed tracks, so they are excluded
          from the persistent active set in later filtering/postprocess steps.
        """
        active_tracks = TrackInstances.cat_tracked_instances(previous_tracks, new_tracks)
        # ids of unmatched_dets are always -1, but their features may be used as hard negatives.
        active_tracks = TrackInstances.cat_tracked_instances(active_tracks, unmatched_dets)
        scores = torch.max(logits_to_scores(logits=active_tracks.logits), dim=1).values
        keep_idxes = (scores > self.update_threshold) | (active_tracks.ids >= 0)
        active_tracks = active_tracks[keep_idxes]
        iou_threshold = self._resolve_iou_threshold(active_tracks, img_metas=img_metas, batch_idx=batch_idx)
        active_tracks.ids[active_tracks.iou < iou_threshold] = -1
        return active_tracks

    def _select_active_tracks_with_aug(self, previous_tracks: TrackInstances,
                                       new_tracks: TrackInstances,
                                       unmatched_dets: TrackInstances,
                                       no_augment: bool,
                                       img_metas,
                                       batch_idx: int = 0) -> TrackInstances:
        active_tracks = TrackInstances.cat_tracked_instances(previous_tracks, new_tracks)
        iou_threshold = self._resolve_iou_threshold(active_tracks, img_metas=img_metas, batch_idx=batch_idx)
        active_tracks = active_tracks[(active_tracks.iou > iou_threshold) & (active_tracks.ids >= 0)]

        if self.tp_drop_ratio > 0.0 and not no_augment and len(active_tracks) > 0:
            tp_keep_idx = torch.rand((len(active_tracks), )) > self.tp_drop_ratio
            active_tracks = active_tracks[tp_keep_idx]

        if self.fp_insert_ratio > 0.0 and not no_augment:
            selected_active_tracks = active_tracks[
                torch.bernoulli(torch.ones((len(active_tracks), )) * self.fp_insert_ratio).bool()
            ]
            if len(unmatched_dets) > 0 and len(selected_active_tracks) > 0:
                fp_num = len(selected_active_tracks)
                if fp_num >= len(unmatched_dets):
                    insert_fp = unmatched_dets
                else:
                    selected_active_boxes = box_cxcywh_to_xyxy(selected_active_tracks.boxes)
                    unmatched_boxes = box_cxcywh_to_xyxy(unmatched_dets.boxes)
                    iou, _ = box_iou_union(unmatched_boxes, selected_active_boxes)
                    fp_idx = torch.max(iou, dim=0).indices
                    fp_idx = torch.unique(fp_idx)
                    insert_fp = unmatched_dets[fp_idx]
                active_tracks = TrackInstances.cat_tracked_instances(active_tracks, insert_fp)
        return active_tracks

    def _build_fake_tracks(self, n_classes: int) -> TrackInstances:
        device = next(self.query_feat_ffn.parameters()).device
        fake_tracks = TrackInstances(frame_height=1.0, frame_width=1.0, hidden_dim=self.hidden_dim).to(device=device)
        if self.use_dab:
            fake_tracks.query_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
            fake_tracks.ref_pts = torch.randn((1, 5), dtype=torch.float, device=device)
        else:
            fake_tracks.query_embed = torch.randn((1, 2 * self.hidden_dim), dtype=torch.float, device=device)
            # fake_tracks.ref_pts = torch.randn((1, 2), dtype=torch.float, device=device)
            fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
        fake_tracks.output_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
        fake_tracks.ids = torch.as_tensor([-2], dtype=torch.long, device=device)  # fake tracks sentinel id
        fake_tracks.matched_idx = torch.as_tensor([-2], dtype=torch.long, device=device)
        fake_tracks.boxes = torch.randn((1, 5), dtype=torch.float, device=device)
        fake_tracks.logits = torch.randn((1, n_classes), dtype=torch.float, device=device)
        fake_tracks.iou = torch.zeros((1,), dtype=torch.float, device=device)
        fake_tracks.last_output = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
        fake_tracks.long_memory = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
        if TrackInstances.use_q_spec:
            fake_tracks.obs_q_spec = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
            fake_tracks.query_q_spec = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
        if TrackInstances.use_spectral_decoder:
            fake_tracks.pred_spectral_weights = torch.randn((1, TrackInstances.decoder_spectral_weights_dim), dtype=torch.float, device=device)
            fake_tracks.query_spectral_weights = torch.randn((1, TrackInstances.decoder_spectral_weights_dim), dtype=torch.float, device=device)
        return fake_tracks

    def select_active_tracks(self, previous_tracks: List[TrackInstances],
                             new_tracks: List[TrackInstances],
                             unmatched_dets: List[TrackInstances],
                             no_augment: bool = False,
                             img_metas=None):
        tracks = []
        if self.training:
            if self.track_iou_adaptive and img_metas is None:
                raise ValueError("img_metas is required for scale-adaptive IoU threshold during training.")
            for b in range(len(new_tracks)):
                self._init_track_memory_fields(new_tracks[b])
                self._init_track_memory_fields(unmatched_dets[b])

                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = self._select_active_tracks_no_aug(
                        previous_tracks=previous_tracks[b],
                        new_tracks=new_tracks[b],
                        unmatched_dets=unmatched_dets[b],
                        img_metas=img_metas,
                        batch_idx=b,
                    )
                else:
                    active_tracks = self._select_active_tracks_with_aug(
                        previous_tracks=previous_tracks[b],
                        new_tracks=new_tracks[b],
                        unmatched_dets=unmatched_dets[b],
                        no_augment=no_augment,
                        img_metas=img_metas,
                        batch_idx=b,
                    )

                if len(active_tracks) == 0:
                    active_tracks = self._build_fake_tracks(n_classes=active_tracks.logits.shape[1])
                tracks.append(active_tracks)
        else:
            # Eval only has B=1.
            assert len(previous_tracks) == 1 and len(new_tracks) == 1
            self._init_track_memory_fields(new_tracks[0])
            active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[0], new_tracks[0])
            active_tracks = active_tracks[active_tracks.ids >= 0]
            tracks.append(active_tracks)
        return tracks


def build(config: dict):
    if config["ONLY_TRAIN_DETR"] is True:
        return None
    return QueryUpdater(
            hidden_dim=config["HIDDEN_DIM"],
            ffn_dim=config["FFN_DIM"],
            dropout=config["DROPOUT"],
            tp_drop_ratio=config["TP_DROP_RATE"] if "TP_DROP_RATE" in config else 0.0,
            fp_insert_ratio=config["FP_INSERT_RATE"] if "FP_INSERT_RATE" in config else 0.0,
            use_checkpoint=config["USE_CHECKPOINT"],
            use_dab=config["USE_DAB"],
            update_threshold=config["UPDATE_THRESH"],
            long_memory_lambda=config["LONG_MEMORY_LAMBDA"],
            track_iou_threshold=config.get("TRACK_IOU_THRESH", 0.3),
            track_iou_adaptive=config.get("TRACK_IOU_ADAPTIVE", True),
            track_iou_threshold_max=config.get("TRACK_IOU_THRESH_MAX", 0.5),
            track_iou_area_min=config.get("TRACK_IOU_AREA_MIN", 800.0),
            track_iou_area_max=config.get("TRACK_IOU_AREA_MAX", 4000.0),
            visualize=config["VISUALIZE"],
            q_spec_lambda=config["Q_SPEC_LAMBDA"] if "Q_SPEC_LAMBDA" in config else 0.0,
        )

