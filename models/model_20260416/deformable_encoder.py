import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..ops.modules import MSDeformAttn20260416
from ..utils import get_activation_layer, get_clones


class DeformableEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint: bool):
        super().__init__()
        self.layers = get_clones(module=encoder_layer, n=num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                spectral=None, add_tokens=None, add_specs=None, add_pos_embeds=None, add_level_start_index=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint:
                assert len(self.layers) % 3 == 0

                def fn(x, i):
                    x, add_tokens_inner = self.layers[i](x, pos, spectral, reference_points, spatial_shapes, level_start_index,
                                                         padding_mask, add_tokens, add_specs, add_pos_embeds, add_level_start_index)
                    x, add_tokens_inner = self.layers[i + 1](x, pos, spectral, reference_points, spatial_shapes, level_start_index,
                                                             padding_mask, add_tokens_inner, add_specs, add_pos_embeds, add_level_start_index)
                    x, add_tokens_inner = self.layers[i + 2](x, pos, spectral, reference_points, spatial_shapes, level_start_index,
                                                             padding_mask, add_tokens_inner, add_specs, add_pos_embeds, add_level_start_index)
                    return x, add_tokens_inner

                if idx % 3 == 0:
                    output = checkpoint(fn, output, idx, use_reentrant=False)
            else:
                output, add_tokens = layer(
                    output, pos, spectral, reference_points, spatial_shapes, level_start_index,
                    padding_mask, add_tokens, add_specs, add_pos_embeds, add_level_start_index
                )
        return output, add_tokens


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="ReLU",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super().__init__()
        self.self_attn = MSDeformAttn20260416(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points, sigmoid_attn=sigmoid_attn
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ffn)
        self.activation = get_activation_layer(activation=activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_ffn, out_features=d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_spectral_embed(tensor, spectral_embed):
        return tensor if spectral_embed is None else tensor + spectral_embed

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        return self.norm3(src)

    def forward(self, src, pos, spectral, reference_points, spatial_shapes, level_start_index, padding_mask,
                add_tokens, add_specs, add_pos_embeds, add_level_start_inedex):
        output, add_output = self.self_attn(
            self.with_spectral_embed(self.with_pos_embed(src, pos), spectral),
            reference_points, src, spatial_shapes, level_start_index, padding_mask,
            self.with_spectral_embed(self.with_pos_embed(add_tokens, add_pos_embeds), add_specs),
            add_tokens,
            alpha=0.7
        )
        src2 = torch.cat([output, add_output], dim=-2)
        src = torch.cat([src, add_tokens], dim=-2)
        src = self.norm1(src + self.dropout1(src2))
        src = self.forward_ffn(src)
        return src[:, :-add_output.shape[1], :], src[:, -add_output.shape[1]:, :]
