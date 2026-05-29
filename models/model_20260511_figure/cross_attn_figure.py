"""MSDeformAttn 旋转版 + forward 内原生捕获 cross-attn debug。"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..ops.functions import MSDeformAttnFunction
from ..ops.modules.ms_deform_attn_rotate_20260420 import MSDeformAttn_Rotate
from hsmot.datasets.pipelines.channel import rotate_norm_angles_to_angles


class MSDeformAttnFigure(MSDeformAttn_Rotate):
    """在 MSDeformAttn_Rotate 基础上，可选地在 forward 内记录 attn 权重与采样位置。"""

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        q_spec=None,
        capture_debug: bool = False,
    ):
        n, len_q, _ = query.shape
        n_in, len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(n, len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n, len_q, self.n_heads, self.n_levels * self.n_points
        )

        if q_spec is not None and self.use_q_spec:
            spec_logits = self.spec_attn_proj(self.spec_attn_norm(q_spec)).view(
                n, len_q, self.n_heads, self.n_levels * self.n_points
            )
            attention_weights = attention_weights + spec_logits * self.spec_attn_alpha

        if self.sigmoid_attn:
            attention_weights = attention_weights.sigmoid().view(
                n, len_q, self.n_heads, self.n_levels, self.n_points
            )
        else:
            attention_weights = F.softmax(attention_weights, -1).view(
                n, len_q, self.n_heads, self.n_levels, self.n_points
            )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 5:
            ref_xy = reference_points[:, :, None, :, None, :2]
            ref_wh = reference_points[:, :, None, :, None, 2:4]
            ref_theta = rotate_norm_angles_to_angles(
                reference_points[:, :, None, :, None, 4], self.version
            )
            offset = sampling_offsets / self.n_points * ref_wh * 0.5
            cos_theta = torch.cos(ref_theta)
            sin_theta = torch.sin(ref_theta)
            rotation_matrix = torch.stack([
                torch.stack([cos_theta, sin_theta], dim=-1),
                torch.stack([-sin_theta, cos_theta], dim=-1),
            ], dim=-2)
            rotated_offset = torch.einsum("...e,...de->...d", offset, rotation_matrix)
            sampling_locations = (ref_xy + rotated_offset).contiguous()
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 5, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        if capture_debug:
            self.last_cross_attn_debug = {
                "attention_weights": attention_weights.detach(),
                "sampling_locations": sampling_locations.detach(),
                "spatial_shapes": input_spatial_shapes.detach(),
            }
        else:
            self.last_cross_attn_debug = None

        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)
        return output

    def pop_cross_attn_debug(self) -> Optional[Dict[str, Any]]:
        debug = getattr(self, "last_cross_attn_debug", None)
        self.last_cross_attn_debug = None
        return debug
