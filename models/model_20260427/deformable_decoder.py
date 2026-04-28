from typing import Callable, List, Optional

import torch
import torch.nn as nn

from ..mlp import MLP
from ..ops.modules import MSDeformAttn_Rotate_20260427 as MSDeformAttn
from ..utils import get_activation_layer, get_clones, pos_to_pos_embed_rotated
from utils.utils import inverse_sigmoid


class DeformableDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, merge_det_track_layer: int = 0,
                 n_det_queries: int = 300, d_model: int = 256, use_checkpoint: bool = False,
                 use_dab: bool = False, visualize: bool = False, use_q_spec: bool = True):
        super().__init__()
        self.layers = get_clones(module=decoder_layer, n=num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.merge_det_track_layer = merge_det_track_layer
        self.n_det_queries = n_det_queries
        self.d_model = d_model
        self.bbox_embed = None
        self.angle_embed = None
        self.class_embed = None
        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize
        self.use_q_spec = use_q_spec
        if self.use_dab:
            self.query_scale = MLP(self.d_model, self.d_model, self.d_model, 2)
            self.ref_point_head = MLP(self.d_model * 2 + 2, self.d_model, self.d_model, 2)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos, query_mask, src_padding_mask, q_spec=None,
                spectral_weights: Optional[List[torch.Tensor]] = None,
                q_spec_builder: Optional[Callable[[List[torch.Tensor], torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                track_q_spec: Optional[torch.Tensor] = None):
        output = tgt
        # 统一语义:
        # - layer_input_queries[l]:  第 l 层 decoder 的输入 query（进入 layer 之前）
        # - layer_output_queries[l]: 第 l 层 decoder 的输出 query（经过 layer 之后）
        # - layer_output_refs[l]:    第 l 层 decoder 更新后的 reference points（层后）
        layer_output_queries, layer_output_refs, layer_input_queries = [], [], []
        layer_cls_q_specs, layer_obs_q_specs, layer_track_obs_q_specs = [], [], []
        num_total_queries = reference_points.shape[1]
        num_tracks = max(0, num_total_queries - self.n_det_queries)
        static_track_q_spec = None
        if self.use_q_spec and num_tracks > 0:
            if track_q_spec is not None:
                static_track_q_spec = track_q_spec
                assert static_track_q_spec.shape[1] == num_tracks, "static_track_q_spec的长度和num_tracks要保持一致"
            elif q_spec is not None:
                static_track_q_spec = q_spec[:, self.n_det_queries:, :]
        for lid, layer in enumerate(self.layers):
            if (lid == 0) and (self.use_dab is False):
                ref_pts_backup = reference_points.clone()
                reference_points = reference_points[:, :, :2]
            if reference_points.shape[-1] == 5:
                reference_points_input = reference_points[:, :, None] * torch.cat(
                    [src_valid_ratios, src_valid_ratios, torch.ones((*src_valid_ratios.shape[:-1], 1), device=src_valid_ratios.device)],
                    -1,
                )[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            if self.use_dab:
                anchor_embed = pos_to_pos_embed_rotated(reference_points_input[:, :, 0, :], num_pos_feats=self.d_model // 2)
                raw_query_pos = self.ref_point_head(anchor_embed)
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            layer_input_queries.append(output)
            layer_q_spec = q_spec
            if self.use_q_spec and q_spec_builder is not None and spectral_weights is not None:
                det_obs_q_spec = q_spec_builder(
                    spectral_weights=spectral_weights,
                    reference_points=reference_points[:, :self.n_det_queries],
                    valid_ratios=src_valid_ratios,
                )
                if num_tracks > 0:
                    if static_track_q_spec is None:
                        raise ValueError(
                            "track_q_spec is required when track queries exist and use_q_spec=True."
                        )
                    layer_q_spec = torch.cat((det_obs_q_spec, static_track_q_spec), dim=1)
                else:
                    layer_q_spec = det_obs_q_spec

            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(
                    layer, output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                    query_mask, src_padding_mask, (lid >= self.merge_det_track_layer), layer_q_spec, use_reentrant=False
                )
            else:
                output = layer(
                    tgt=output, query_pos=query_pos, reference_points=reference_points_input, src=src,
                    src_spatial_shapes=src_spatial_shapes, level_start_index=src_level_start_index,
                    query_mask=query_mask, src_padding_mask=src_padding_mask, merge_det_track=(lid >= self.merge_det_track_layer),
                    q_spec=layer_q_spec,
                )

            if self.bbox_embed is not None:
                tmp = torch.cat((self.bbox_embed[lid](output), self.angle_embed[lid](output)), dim=-1)
                if reference_points.shape[-1] == 5:
                    new_reference_points = (tmp + inverse_sigmoid(reference_points)).sigmoid()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                if lid < self.merge_det_track_layer:
                    if self.use_dab:
                        reference_points = torch.cat((new_reference_points[:, :self.n_det_queries, :].detach(),
                                                      reference_points[:, self.n_det_queries:, :]), dim=1)
                    else:
                        reference_points = torch.cat((new_reference_points[:, :self.n_det_queries, :].detach(),
                                                      ref_pts_backup[:, self.n_det_queries:, :]), dim=1)
                else:
                    reference_points = new_reference_points.detach()

            layer_cls_q_spec = None
            layer_obs_q_spec = None
            layer_track_obs_q_spec = None
            if self.use_q_spec and q_spec_builder is not None and spectral_weights is not None:
                det_obs_q_spec = q_spec_builder(
                    spectral_weights=spectral_weights,
                    reference_points=reference_points[:, :self.n_det_queries],
                    valid_ratios=src_valid_ratios,
                )
                if num_tracks > 0:
                    if static_track_q_spec is None:
                        raise ValueError(
                            "track_q_spec is required when track queries exist and use_q_spec=True."
                        )
                    layer_track_obs_q_spec = q_spec_builder(
                        spectral_weights=spectral_weights,
                        reference_points=reference_points[:, self.n_det_queries:],
                        valid_ratios=src_valid_ratios,
                    )
                    layer_cls_q_spec = torch.cat((det_obs_q_spec, static_track_q_spec), dim=1)
                    layer_obs_q_spec = torch.cat((det_obs_q_spec, layer_track_obs_q_spec), dim=1)
                else:
                    layer_cls_q_spec = det_obs_q_spec
                    layer_obs_q_spec = det_obs_q_spec

            if self.return_intermediate:
                layer_output_queries.append(output)
                layer_output_refs.append(reference_points)
                layer_cls_q_specs.append(layer_cls_q_spec)
                layer_obs_q_specs.append(layer_obs_q_spec)
                layer_track_obs_q_specs.append(layer_track_obs_q_spec)

        if self.return_intermediate:
            stacked_cls_q_specs = None
            stacked_obs_q_specs = None
            stacked_track_obs_q_specs = None
            if self.use_q_spec and len(layer_cls_q_specs) > 0 and layer_cls_q_specs[0] is not None:
                stacked_cls_q_specs = torch.stack(layer_cls_q_specs)
                stacked_obs_q_specs = torch.stack(layer_obs_q_specs)
                if layer_track_obs_q_specs[0] is not None:
                    stacked_track_obs_q_specs = torch.stack(layer_track_obs_q_specs)
            return (
                torch.stack(layer_output_queries),
                torch.stack(layer_output_refs),
                torch.stack(layer_input_queries),
                stacked_cls_q_specs,
                stacked_obs_q_specs,
                stacked_track_obs_q_specs,
            )
        raise NotImplementedError("Not Support for no Inter Outputs.")


class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="ReLU",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False, extra_track_attn=False,
                 n_det_queries=300, visualize: bool = False, use_q_spec: bool = True):
        super().__init__()
        self.visualize = visualize
        self.n_det_queries = n_det_queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points, sigmoid_attn=sigmoid_attn, use_q_spec=use_q_spec)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_layer(activation=activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            self.track_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_self_attn(self, tgt, query_pos, query_mask):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, tgt, key_padding_mask=query_mask)
        return self.norm2(tgt + self.dropout2(tgt2))

    def forward_track_attn(self, tgt, query_pos, query_mask):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > self.n_det_queries:
            tgt2, _ = self.track_attn(q[:, self.n_det_queries:], k[:, self.n_det_queries:], tgt[:, self.n_det_queries:],
                                      key_padding_mask=query_mask[:, self.n_det_queries:])
            tgt2 = self.norm4(tgt[:, self.n_det_queries:] + self.dropout5(tgt2))
            tgt = torch.cat([tgt[:, :self.n_det_queries], tgt2], dim=1)
        return tgt

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(tgt2))

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, query_mask,
                src_padding_mask=None, merge_det_track=False, q_spec=None):
        if merge_det_track is False:
            track_tgt = tgt[:, self.n_det_queries:, :]
            tgt = tgt[:, :self.n_det_queries, :]
            query_pos = query_pos[:, :self.n_det_queries, :]
            reference_points = reference_points[:, :self.n_det_queries, :, :]
            query_mask = query_mask[:, :self.n_det_queries]
            if q_spec is not None:
                q_spec = q_spec[:, :self.n_det_queries, :]

        if self.extra_track_attn:
            assert False, "Not Support for extra track attn."
            tgt = self.forward_track_attn(tgt, query_pos, query_mask)

        tgt = self.forward_self_attn(tgt, query_pos, query_mask)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos), reference_points=reference_points,
            input_flatten=src, input_spatial_shapes=src_spatial_shapes, input_level_start_index=level_start_index,
            input_padding_mask=src_padding_mask,
            q_spec=q_spec
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt = self.forward_ffn(tgt)
        if merge_det_track is False:
            tgt = torch.cat((tgt, track_tgt), dim=1)
        return tgt
