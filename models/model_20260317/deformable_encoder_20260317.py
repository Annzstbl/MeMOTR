from ..model_20260310.deformable_encoder_20260310 import DeformableEncoder20260310
from ..model_20260310.deformable_encoder_20260310 import DeformableEncoderLayer20260310

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from ..ops.modules import MSDeformAttn20260317
from ..utils import get_activation_layer, get_clones

class DeformableEncoder20260317(DeformableEncoder20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
        spectral=None,
        add_tokens=None,
        add_specs=None,
        add_pos_embeds=None,
        add_level_start_index=None,
    ):
        """
        Args 与 DeformableEncoderLayer20260310.forward 对齐，除了 src/pos 等外，
        还需要接收先验 token 相关的张量，并在每一层中使用。
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint:
                assert len(self.layers) % 3 == 0, f"Encoder Layers count must be 3x when using checkpoint."

                def fn(x, i):
                    x,add_tokens = self.layers[i](
                        x,
                        pos,
                        spectral,
                        reference_points,
                        spatial_shapes,
                        level_start_index,
                        padding_mask,
                        add_tokens,
                        add_specs,
                        add_pos_embeds,
                        add_level_start_index,
                    )
                    x,add_tokens = self.layers[i + 1](
                        x,
                        pos,
                        spectral,
                        reference_points,
                        spatial_shapes,
                        level_start_index,
                        padding_mask,
                        add_tokens,
                        add_specs,
                        add_pos_embeds,
                        add_level_start_index,
                    )
                    x,add_tokens = self.layers[i + 2](
                        x,
                        pos,
                        spectral,
                        reference_points,
                        spatial_shapes,
                        level_start_index,
                        padding_mask,
                        add_tokens,
                        add_specs,
                        add_pos_embeds,
                        add_level_start_index,
                    )
                    return x, add_tokens

                if idx % 3 == 0:
                    output = checkpoint(fn, output, idx, use_reentrant=False)
            else:
                output, add_tokens = layer(
                    output,
                    pos,
                    spectral,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                    add_tokens,
                    add_specs,
                    add_pos_embeds,
                    add_level_start_index,
                )
        return output, add_tokens


class DeformableEncoderLayer20260317(nn.Module):
    """
    input:
    output: (B, Nq, C)
    """
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="ReLU",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super(DeformableEncoderLayer20260317, self).__init__()

        self.self_attn = MSDeformAttn20260317(
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
        src2 = self.linear2(
            self.dropout3(
                self.activation(
                    self.linear1(src)
                )
            )
        )
        src = src + self.dropout4(src2)
        src = self.norm3(src)
        return src
    
    def forward(self, 
                src, 
                pos, 
                spectral, 
                reference_points, 
                spatial_shapes, 
                level_start_index, 
                padding_mask,
                add_tokens,
                add_specs,
                add_pos_embeds,
                add_level_start_inedex
                ):
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

            add_tokens:      (B, num_prior_token, C)
            add_specs: (B, num_prior_token, C)
            add_pos_embeds:  (B, num_prior_token, C)
            add_level_start_inedex: (n_levels, )
        Returns:

        """
        # Self Attention
        src2 = self.self_attn(
            self.with_spectral_embed(self.with_pos_embed(src, pos), spectral),
            reference_points, 
            src, 
            spatial_shapes, 
            level_start_index, 
            padding_mask,
            self.with_spectral_embed(self.with_pos_embed(add_tokens, add_pos_embeds), add_specs),
            add_tokens)

        src = torch.cat([src, add_tokens], dim=-2)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src) #[B, len_q + len_a, C]

        return src[:, :-add_tokens.shape[1], :], src[:, -add_tokens.shape[1]:, :]


DeformableEncoder = DeformableEncoder20260317
DeformableEncoderLayer = DeformableEncoderLayer20260317
