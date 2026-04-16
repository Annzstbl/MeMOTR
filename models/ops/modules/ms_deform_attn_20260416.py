# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: 在 MOTR 的对应文件中增加了部分注释，助于理解。
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttnAddTokenSharedLogits(nn.Module):
    """
    结构包含三部分：
    1) 原有 deformable self-attention 分支
    2) query -> add_token 读取分支
    3) add_token <- value 写回更新分支

    其中第2/3部分共享同一个 query-add logits：
        scores[q, a] = <Q(query_q), K(add_a)>

    - 读取 query -> add 时：
        对 add 维 softmax
    - 写回 add <- query/value 时：
        对 query 维 softmax
    """

    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        sigmoid_attn=False,
        visualize=False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads)
            )

        d_per_head = d_model // n_heads
        if not _is_power_of_2(d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each "
                "attention head a power of 2 which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64
        self.sigmoid_attn = sigmoid_attn
        self.visualize = visualize

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_per_head = d_per_head

        # original deformable branch
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # shared logits for add-token interaction
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        # deformable offsets
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)

        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # deformable attention weights
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        # projections
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)

        xavier_uniform_(self.key_proj.weight.data)
        constant_(self.key_proj.bias.data, 0.)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask,
        add_key,
        add_value,
        alpha=0.7,
    ):
        """
        Args:
            query:                  [N, Len_q, C]
                                    一般为 src + pos (+ spectral)
            reference_points:       [N, Len_q, n_levels, 2] or [N, Len_q, n_levels, 4]
            input_flatten:          [N, Len_in, C]
                                    一般为 src
            input_spatial_shapes:   [n_levels, 2]
            input_level_start_index:[n_levels]
            input_padding_mask:     [N, Len_in], True 为 padding
            add_key:                [N, Len_add, C]
                                    add token 的 key-side 表示（可带 pos/spec）
            add_value:              [N, Len_add, C]
                                    add token 的 value-side 表示（一般不加 pos）
            alpha:                  float
                                    deformable 分支权重；add 读取分支权重为 (1 - alpha)

        Returns:
            output:                 [N, Len_q, C]
            add_output:             [N, Len_add, C]
        """
        N, Len_q, _ = query.shape
        N2, Len_in, _ = input_flatten.shape

        assert N == N2
        assert Len_q == Len_in, "This module is designed for self-attention style usage: Len_q must equal Len_in."
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # ------------------------------------------------------------------
        # 0) prepare value
        # ------------------------------------------------------------------
        value = self.value_proj(input_flatten)  # [N, Len_in, C]
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)

        value_heads = value.view(N, Len_in, self.n_heads, self.d_per_head)  # [N, Q, H, Dh]

        # ------------------------------------------------------------------
        # 1) original deformable self-attention branch
        # ------------------------------------------------------------------
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        local_logits = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )

        if self.sigmoid_attn:
            local_weights = local_logits.sigmoid().view(
                N, Len_q, self.n_heads, self.n_levels, self.n_points
            )
        else:
            local_weights = F.softmax(local_logits, dim=-1).view(
                N, Len_q, self.n_heads, self.n_levels, self.n_points
            )


        if reference_points.shape[-1] == 2:
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / input_spatial_shapes[None, None, None, :, None, (1, 0)]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        if self.visualize:
            torch.save(
                sampling_locations[0].detach().cpu(),
                "./outputs/visualize_tmp/decoder/sampling_locations.tensor"
            )

        output_deformable = MSDeformAttnFunction.apply(
            value_heads,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            local_weights,
            self.im2col_step,
        )  # [N, Len_q, C]

        # ------------------------------------------------------------------
        # 2) shared query-add logits
        #    scores[q, a] = <Q(query_q), K(add_a)>
        # ------------------------------------------------------------------
        Len_add = add_key.shape[1]

        q_add = self.query_proj(query).view(N, Len_q, self.n_heads, self.d_per_head)      # [N, Q, H, Dh]
        k_add = self.key_proj(add_key).view(N, Len_add, self.n_heads, self.d_per_head)     # [N, A, H, Dh]
        v_add = self.value_proj(add_value).view(N, Len_add, self.n_heads, self.d_per_head) # [N, A, H, Dh]

        # shared logits: [N, Q, H, A]
        add_logits = torch.einsum("nqhd,nahd->nqha", q_add, k_add)

        if self.sigmoid_attn:
            # sigmoid 版本仍然可以共享 logits，但注意这不是标准概率归一化
            add_read_weights = add_logits.sigmoid()  # [N, Q, H, A]
            add_write_weights = add_logits.permute(0, 3, 2, 1).sigmoid()  # [N, A, H, Q]
        else:
            # 正确缩放：sqrt(d_head)
            scaled_add_logits = add_logits / math.sqrt(self.d_per_head)

            # 2.1 query -> add 读取：在 Len_add 维 softmax
            add_read_weights = F.softmax(scaled_add_logits, dim=-1)  # [N, Q, H, A]

            # 2.2 add <- query/value 写回：在 Len_q 维 softmax
            add_write_logits = scaled_add_logits.permute(0, 3, 2, 1)  # [N, A, H, Q]

            if input_padding_mask is not None:
                add_write_logits = add_write_logits.masked_fill(
                    input_padding_mask[:, None, None, :], float("-inf")
                )

            add_write_weights = F.softmax(add_write_logits, dim=-1)  # [N, A, H, Q]


        # ------------------------------------------------------------------
        # 3) query -> add 读取
        # ------------------------------------------------------------------
        output_add = torch.einsum("nqha,nahd->nqhd", add_read_weights, v_add).reshape(
            N, Len_q, self.d_model
        )

        # ------------------------------------------------------------------
        # 4) add <- query/value 写回更新
        #    softmax 已经在 Len_q 维做过
        # ------------------------------------------------------------------
        add_output = torch.einsum("nahq,nqhd->nahd", add_write_weights, value_heads).reshape(
            N, Len_add, self.d_model
        )

        # ------------------------------------------------------------------
        # 5) merge
        # ------------------------------------------------------------------
        output = output_deformable + output_add * (1-alpha)
        output = self.output_proj(output)
        add_output = self.output_proj(add_output)

        return output, add_output