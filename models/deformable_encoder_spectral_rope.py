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
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .ops.modules import MSDeformAttn, MSDeformAttnSpectral
from .utils import get_activation_layer, get_clones


class DeformableEncoderSpectralRope(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint: bool, rope_pos_module):
        super(DeformableEncoderSpectralRope, self).__init__()
        self.layers = get_clones(module=encoder_layer, n=num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.rope_pos_module = rope_pos_module
        
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, spectral=None):
        assert pos is None, "pos should be None"

        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            if self.use_checkpoint:
                assert len(self.layers) % 2 == 0, f"Encoder Layers must be 3x"

                def fn(x, i):
                    x = self.layers[i](x, self.rope_pos_module, spectral, reference_points, spatial_shapes, level_start_index, padding_mask, valid_ratios)
                    x = self.layers[i + 1](x, self.rope_pos_module, spectral, reference_points, spatial_shapes, level_start_index, padding_mask, valid_ratios)
                    x = self.layers[i + 2](x, self.rope_pos_module, spectral, reference_points, spatial_shapes, level_start_index, padding_mask, valid_ratios)
                    return x
                if _ % 3 == 0:
                    output = checkpoint(fn, output, _, use_reentrant=False)
                else:
                    pass
            else:
                output = layer(output, self.rope_pos_module, spectral, reference_points, spatial_shapes, level_start_index, padding_mask, valid_ratios)
        return output


class DeformableEncoderLayerSpectralRope(nn.Module):
    """
    input:
    output: (B, Nq, C)
    """
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="ReLU",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False, spectral_attention=False):
        super(DeformableEncoderLayerSpectralRope, self).__init__()

        # Self Attention
        if spectral_attention:
            self.self_attn = MSDeformAttnSpectral(
                d_model=d_model,
                n_levels=n_levels,
                n_heads=n_heads,
                n_points=n_points,
                sigmoid_attn=sigmoid_attn
            )
        else:
            self.self_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            sigmoid_attn=sigmoid_attn
        )   # output shape: (B, Nq, C)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ffn)
        self.activation = get_activation_layer(activation=activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_ffn, out_features=d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    # @staticmethod
    # def with_pos_embed(tensor, pos):
    #     return tensor if pos is None else tensor + pos

    @staticmethod
    def with_spectral_embed(tensor, spectral_embed):
        return tensor if spectral_embed is None else tensor + spectral_embed

    @staticmethod
    def get_pos_rope(spatial_shapes, valid_ratios, device):
        
        num_lvls = len(spatial_shapes)
        lvl_rope = torch.linspace(-1, 1, num_lvls, device=device) #(num_lvls, )

        pos_rope_list = []
        assert valid_ratios.shape[0] == 1, "只支持bs = 1"
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 有效边界是 H_ * valid_ratios
            # 从0到有效边界，值从-1到1构建linspace
            # 有效边界外的，保持间隔继续构建
            end_y = 2 / (H_ * valid_ratios[0, lvl, 1]) * H_ - 1
            end_x = 2 / (W_ * valid_ratios[0, lvl, 0]) * W_ - 1
            pos_y, pos_x = torch.meshgrid(torch.linspace(-1, end_y, H_, device=device), torch.linspace(-1, end_x, W_, device=device)) #
            pos_y = pos_y.reshape(-1)
            pos_x = pos_x.reshape(-1)
            pos_lvl = torch.ones_like(pos_y) * lvl_rope[lvl]
            pos_rope = torch.stack([pos_x, pos_y, pos_lvl], -1) #(H_ * W_, 3)
            pos_rope_list.append(pos_rope)
        pos_rope = torch.cat(pos_rope_list, 0) #(sum(H_ * W_), 3)
        return pos_rope[None]

    def with_pos_embed_rope(self, tensor, pos_module, pos_rope):
        return pos_module.forward_integrate_head(tensor, pos_rope)
        # return pos_module(tensor, pos_rope)

    def forward_ffn(self, src):
        src2 = self.linear2(
            self.dropout2(
                self.activation(
                    self.linear1(src)
                )
            )
        )
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, rope_pos_module, spectral, reference_points, spatial_shapes, level_start_index, padding_mask=None, valid_ratios=None):
        """
        Args:
            src:                    (B, Nq, C)
            pos:                    (B, Nq, C)
            spectral:               (B, Nq, C)
            reference_points:       (B, Nq, n_levels, 2) for point, (B, Nq, n_levels, 4) for box
            spatial_shapes:         (n_levels, 2), as H,W
            level_start_index:      (n_levels, )
            padding_mask:           (B, \\sum_{l=0}^{L-1} H_l \\cdot W_l),
                                    True for padding elements, False for non-padding elements

        Returns:

        """

        pos_rope = self.get_pos_rope(spatial_shapes, valid_ratios, device=src.device)
        src = self.with_spectral_embed(self.with_pos_embed_rope(src, rope_pos_module, pos_rope), spectral)

        # Self Attention
        src2 = self.self_attn(src,
                              reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src
