# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import math
import torch
import torch.nn as nn

from typing import List
from .utils import pos_to_pos_embed_rotated, logits_to_scores
from torch.utils.checkpoint import checkpoint

from .ffn import FFN
from .mlp import MLP
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union


class QueryUpdater(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int,
                 tp_drop_ratio: float, fp_insert_ratio: float,
                 dropout: float,
                 use_checkpoint: bool, use_dab: bool,
                 update_threshold: float, long_memory_lambda: float,
                 visualize: bool = False,
                 query_spectral_weights_dim: int = 8, 
                 decoder_spectral: bool = True):
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
        
        self.query_spectral_weights_dim = query_spectral_weights_dim
        self.decoder_spectral = decoder_spectral

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                no_augment: bool = False):
        # scores> update_threshold 或者 active_tracks.ids>=0的会列为active_tracks
        # 不active的会直接消失
        # iou<0.5的 active_tracks会将track_id=-1
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
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

            if self.decoder_spectral:
                tracks[b].query_spectral_weights[is_pos] = tracks[b][is_pos].pred_spectral_weights.detach().clone()


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

    def _select_active_tracks_no_aug(self, previous_tracks: TrackInstances,
                                     new_tracks: TrackInstances,
                                     unmatched_dets: TrackInstances) -> TrackInstances:
        """Select active tracks for training without augmentation.

        Pipeline:
        1) Merge `previous_tracks` and `new_tracks`.
        2) Append `unmatched_dets` (their ids are -1 by design, used as hard negatives).
        3) Keep tracks that are confident (`score > update_threshold`) OR already have valid ids (`id >= 0`).
        4) Apply an IoU gate: tracks with `iou < 0.5` are marked as invalid by setting `id = -1`.

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
        active_tracks.ids[active_tracks.iou < 0.5] = -1 #TODO 0.5这个阈值太大了
        return active_tracks

    def _select_active_tracks_with_aug(self, previous_tracks: TrackInstances,
                                       new_tracks: TrackInstances,
                                       unmatched_dets: TrackInstances,
                                       no_augment: bool) -> TrackInstances:
        active_tracks = TrackInstances.cat_tracked_instances(previous_tracks, new_tracks)
        active_tracks = active_tracks[(active_tracks.iou > 0.5) & (active_tracks.ids >= 0)]

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
        if self.decoder_spectral:
            fake_tracks.pred_spectral_weights = torch.randn((1, self.query_spectral_weights_dim), dtype=torch.float, device=device)
            fake_tracks.query_spectral_weights = torch.randn((1, self.query_spectral_weights_dim), dtype=torch.float, device=device)
        return fake_tracks

    def select_active_tracks(self, previous_tracks: List[TrackInstances],
                             new_tracks: List[TrackInstances],
                             unmatched_dets: List[TrackInstances],
                             no_augment: bool = False):
        tracks = []
        if self.training:
            for b in range(len(new_tracks)):
                self._init_track_memory_fields(new_tracks[b])
                self._init_track_memory_fields(unmatched_dets[b])

                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = self._select_active_tracks_no_aug(
                        previous_tracks=previous_tracks[b],
                        new_tracks=new_tracks[b],
                        unmatched_dets=unmatched_dets[b],
                    )
                else:
                    active_tracks = self._select_active_tracks_with_aug(
                        previous_tracks=previous_tracks[b],
                        new_tracks=new_tracks[b],
                        unmatched_dets=unmatched_dets[b],
                        no_augment=no_augment,
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
            visualize=config["VISUALIZE"],
            query_spectral_weights_dim=config["DECODER_SPECTRAL_CLUSTERS"] * 8,
            decoder_spectral=config["DECODER_SPECTRAL"]
        )

