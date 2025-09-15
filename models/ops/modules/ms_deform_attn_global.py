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
from torch import logit, nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttnGlobal(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False, visualize=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        :param sigmoid_attn 使用 sigmoid 代替 softmax 计算 attention score，在原本的 Deformable DETR 中没有。
        :param scheme       joint
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.sigmoid_attn = sigmoid_attn

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.attention_weights_to_global = nn.Linear(d_model, n_heads * n_levels)
        self.visualize = visualize
 
        self.reset_parameters()

    def reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

        constant_(self.attention_weights_to_global.weight.data, 0.)
        constant_(self.attention_weights_to_global.bias.data, 0.)


    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, global_query=None, global_input=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :param global_query                (N, n_levels, C)
        :param global_input                (N, n_levels, C)

        :return output                     (N, Length_{query}, C)
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        global_value = self.value_proj(global_input).view(N, self.n_levels, self.n_heads, self.d_model // self.n_heads)

        if input_padding_mask is not None:
            value.masked_fill_(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # query到 global feature的注意力权重
        attention_weights_to_global = self.attention_weights_to_global(query).view(N, Len_q, self.n_heads, self.n_levels)


        if self.sigmoid_attn:
            attention_weights = attention_weights.sigmoid().view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
            attention_weights_to_global = attention_weights_to_global.sigmoid().view(N, Len_q, self.n_heads, self.n_levels, 1)
        else:
            all_attention_weights = torch.cat([attention_weights, attention_weights_to_global], dim=-1) #(N, Len_q, n_heads, n_levels * n_points + 1)
            all_attention_weights = F.softmax(all_attention_weights, -1)#(N, Len_q, n_heads, n_levels * n_points + 1)
            attention_weights = all_attention_weights[..., :-self.n_levels].view(N, Len_q, self.n_heads, self.n_levels, self.n_points).contiguous()
            attention_weights_to_global = all_attention_weights[..., -self.n_levels:].view(N, Len_q, self.n_heads, self.n_levels)
        

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / input_spatial_shapes[None, None, None, :, None, (1, 0)]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        if self.visualize:
            torch.save(sampling_locations[0].cpu(),
                       "./outputs/visualize_tmp/decoder/sampling_locations.tensor")

        # deformable attention
        output_defomrable = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)# [N, Len_q, D]
        

        # global attention
        # [B, len_q, n_heads, n_levels, ]  * [B, n_levels, n_heads,  C//n_heads] -> [B, len_q, C]
        output_global = torch.einsum('nqhl,nlhd->nqhd', attention_weights_to_global, global_value).reshape(N, Len_q, self.d_model)
        output = output_defomrable + output_global

        # update global feature
        # [B, len_q, n_heads, n_levels, ] * [B, len_q, n_heads, C//n_heads] -> [B, n_levels, C]
        global_output = torch.einsum('nqhl,nqhd->nlhd', attention_weights_to_global, value).reshape(N, self.n_levels, self.d_model)

        # final projection
        output = torch.cat([output, global_output], dim=-2)# [B, len_q + n_levels, C]
        output = self.output_proj(output) # [B, len_q + n_levels, C]

        return output
