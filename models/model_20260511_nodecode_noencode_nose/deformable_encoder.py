import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..ops.modules import MSDeformAttn
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

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint:
                assert len(self.layers) % 3 == 0

                def fn(x, idx=idx):
                    x = self.layers[idx](x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                    x = self.layers[idx + 1](x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                    x = self.layers[idx + 2](x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                    return x

                if idx % 3 == 0:
                    output = checkpoint(fn, output, use_reentrant=False)
            else:
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="ReLU",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super().__init__()
        self.self_attn = MSDeformAttn(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points, sigmoid_attn=sigmoid_attn
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ffn)
        self.activation = get_activation_layer(activation=activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=d_ffn, out_features=d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        return self.norm2(src)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        tgt2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points, src, spatial_shapes, level_start_index, padding_mask,
        )
        src = self.norm1(src + self.dropout1(tgt2))
        src = self.forward_ffn(src)
        return src
