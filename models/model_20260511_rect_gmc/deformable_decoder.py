import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from ..mlp import MLP
from ..ops.functions import MSDeformAttnFunction
from ..ops.modules.ms_deform_attn import MSDeformAttn
from ..utils import get_activation_layer, get_clones, pos_to_pos_embed
from utils.utils import inverse_sigmoid


class MSDeformAttnQSpec(MSDeformAttn):
    """标准 MSDeformAttn（2D/4D ref）+ q_spec 调制 attention logits。

    20260511 旋转版使用 MSDeformAttn_Rotate_20260420（5D ref + q_spec）；
    正框 rect 版 ref 为 4D cxcywh，因此在标准 MSDeformAttn 上补 q_spec 路径。
    """

    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        sigmoid_attn=False,
        use_q_spec=True,
        visualize=False,
    ):
        super().__init__(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            sigmoid_attn=sigmoid_attn,
            visualize=visualize,
        )
        self.use_q_spec = use_q_spec
        if self.use_q_spec:
            self.spec_attn_norm = nn.LayerNorm(d_model)
            self.spec_attn_proj = nn.Linear(d_model, n_heads * n_levels * n_points)
            self.spec_attn_alpha = nn.Parameter(torch.tensor(0.1))
            constant_(self.spec_attn_proj.weight.data, 0.0)
            constant_(self.spec_attn_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        q_spec=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value.masked_fill_(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2,
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points,
        )

        if self.use_q_spec and q_spec is not None:
            spec_logits = self.spec_attn_proj(self.spec_attn_norm(q_spec)).view(
                N, Len_q, self.n_heads, self.n_levels * self.n_points,
            )
            attention_weights = attention_weights + spec_logits * self.spec_attn_alpha

        if self.sigmoid_attn:
            attention_weights = attention_weights.sigmoid().view(
                N, Len_q, self.n_heads, self.n_levels, self.n_points,
            )
        else:
            attention_weights = F.softmax(attention_weights, -1).view(
                N, Len_q, self.n_heads, self.n_levels, self.n_points,
            )

        if reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / input_spatial_shapes[None, None, None, :, None, (1, 0)]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}."
            )

        if self.visualize:
            torch.save(
                sampling_locations[0].cpu(),
                "./outputs/visualize_tmp/decoder/sampling_locations.tensor",
            )
        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        return self.output_proj(output)


class DeformableDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, merge_det_track_layer: int = 0,
                 n_det_queries: int = 300, d_model: int = 256, use_checkpoint: bool = False,
                 use_dab: bool = False, visualize: bool = False):
        super().__init__()
        self.layers = get_clones(module=decoder_layer, n=num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.merge_det_track_layer = merge_det_track_layer
        self.n_det_queries = n_det_queries
        self.d_model = d_model
        self.bbox_embed = None
        self.class_embed = None
        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize
        if self.use_dab:
            self.query_scale = MLP(self.d_model, self.d_model, self.d_model, 2)
            self.ref_point_head = MLP(self.d_model * 2, self.d_model, self.d_model, 2)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos, query_mask, src_padding_mask, q_spec=None):
        output = tgt
        layer_output_queries, layer_output_refs, layer_input_queries = [], [], []
        for lid, layer in enumerate(self.layers):
            if (lid == 0) and (self.use_dab is False):
                ref_pts_backup = reference_points.clone()
                reference_points = reference_points[:, :, :2]

            reference_points_input = reference_points[:, :, None] * torch.cat(
                [src_valid_ratios, src_valid_ratios], -1
            )[:, None]

            if self.use_dab:
                anchor_embed = pos_to_pos_embed(
                    reference_points_input[:, :, 0, :],
                    num_pos_feats=self.d_model // 2,
                )
                raw_query_pos = self.ref_point_head(anchor_embed)
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            layer_input_queries.append(output)

            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(
                    layer, output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                    query_mask, src_padding_mask, (lid >= self.merge_det_track_layer), q_spec, use_reentrant=False
                )
            else:
                output = layer(
                    tgt=output, query_pos=query_pos, reference_points=reference_points_input, src=src,
                    src_spatial_shapes=src_spatial_shapes, level_start_index=src_level_start_index,
                    query_mask=query_mask, src_padding_mask=src_padding_mask, merge_det_track=(lid >= self.merge_det_track_layer),
                    q_spec=q_spec,
                )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                new_reference_points = (tmp + inverse_sigmoid(reference_points)).sigmoid()
                if lid < self.merge_det_track_layer:
                    if self.use_dab:
                        reference_points = torch.cat((new_reference_points[:, :self.n_det_queries, :].detach(),
                                                      reference_points[:, self.n_det_queries:, :]), dim=1)
                    else:
                        reference_points = torch.cat((new_reference_points[:, :self.n_det_queries, :].detach(),
                                                      ref_pts_backup[:, self.n_det_queries:, :]), dim=1)
                else:
                    reference_points = new_reference_points.detach()

            if self.return_intermediate:
                layer_output_queries.append(output)
                layer_output_refs.append(reference_points)

        if self.return_intermediate:
            return (
                torch.stack(layer_output_queries),
                torch.stack(layer_output_refs),
                torch.stack(layer_input_queries),
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
        self.cross_attn = MSDeformAttnQSpec(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            sigmoid_attn=sigmoid_attn,
            use_q_spec=use_q_spec,
            visualize=visualize,
        )
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
