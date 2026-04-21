from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from ..mlp import MLP
from ..ops.modules import MSDeformAttn, MSDeformAttn_Rotate, MSDeformAttnSpectral
from .deformable_decoder import DeformableDecoder, DeformableDecoderLayer
from .deformable_encoder import DeformableEncoder, DeformableEncoderLayer
from hsmot.datasets.pipelines.channel import rotate_norm_angles_to_angles

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, n_feature_levels=4, n_heads=8, n_enc_points=4, n_dec_points=4,
                 n_enc_layers=6, n_dec_layers=6, merge_det_track_layer=0, dropout=0.1, activation="ReLU",
                 return_intermediate_dec=False, n_det_queries=300, extra_track_attn=False,
                 two_stage=False, two_stage_num_proposals=300, use_checkpoint: bool = False,
                 checkpoint_level: int = 2, use_dab: bool = False, visualize: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize

        encoder_layer = DeformableEncoderLayer(d_model=d_model, d_ffn=d_ffn, dropout=dropout, activation=activation,
                                               n_levels=n_feature_levels, n_heads=n_heads, n_points=n_enc_points, sigmoid_attn=False)
        self.encoder = DeformableEncoder(encoder_layer=encoder_layer, num_layers=n_enc_layers,
                                         use_checkpoint=(self.use_checkpoint and self.checkpoint_level == 1))

        decoder_layer = DeformableDecoderLayer(d_model=d_model, d_ffn=d_ffn, dropout=dropout, activation=activation,
                                               n_levels=n_feature_levels, n_heads=n_heads, n_points=n_dec_points, sigmoid_attn=False,
                                               extra_track_attn=extra_track_attn, n_det_queries=n_det_queries, visualize=self.visualize)
        self.decoder = DeformableDecoder(decoder_layer=decoder_layer, num_layers=n_dec_layers,
                                         return_intermediate=return_intermediate_dec, merge_det_track_layer=merge_det_track_layer,
                                         n_det_queries=n_det_queries, d_model=self.d_model, use_checkpoint=self.use_checkpoint,
                                         use_dab=self.use_dab, visualize=self.visualize)

        self.level_embed = nn.Parameter(torch.Tensor(n_feature_levels, d_model))
        self.spectral_embed = MLP(input_dim=8, hidden_dim=self.d_model, output_dim=self.d_model, num_layers=2)
        self.q_spec_obs_embed = MLP(input_dim=8*n_feature_levels, hidden_dim=self.d_model, output_dim=self.d_model, num_layers=2)
        self.q_spec_fuse = MLP(input_dim=2 * self.d_model, hidden_dim=2 * self.d_model, output_dim=self.d_model, num_layers=2)
        self.q_spec_obs_norm = nn.LayerNorm(self.d_model)
        self.q_spec_fuse_norm = nn.LayerNorm(self.d_model)
        if two_stage:
            assert False, "two stage is not supported"
        else:
            if use_dab:
                pass
            else:
                assert False, "use_dab is not supported"
        self.reset_parameters()

    def enable_checkpoint(self, enable: bool):
        self.use_checkpoint = enable
        self.encoder.use_checkpoint = enable
        self.decoder.use_checkpoint = enable

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for module in self.modules():
            if any([isinstance(module, MSDeformAttn), isinstance(module, MSDeformAttnSpectral), isinstance(module, MSDeformAttn_Rotate)]):
                module.reset_parameters()
        normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / w, valid_h.float() / h], -1)

    # def _build_query_spectral_state(
    #     self,
    #     spectral_weights: List[torch.Tensor],
    #     reference_points: torch.Tensor,
    #     tgt: torch.Tensor,
    #     valid_ratios: torch.Tensor,
    # )
    def _build_rotated_box_sampling_grid(
        self,
        reference_points: torch.Tensor,
        valid_ratio: torch.Tensor,
        pool_size: int = 3,
    ):
        """
        Build rotated-box region sampling grid for grid_sample.

        Args:
            reference_points: [B, Q, 5], normalized rotated boxes [cx, cy, w, h, theta]
            valid_ratio:      [B, 2], current level valid ratio
            pool_size:        int, e.g. 3 or 5

        Returns:
            grid: [B, Q, pool_size, pool_size, 2] in [-1, 1] for grid_sample
        """
        assert reference_points.shape[-1] == 5, \
            f"Rotated-box pooling requires 5-dim reference_points, got {reference_points.shape[-1]}"

        # [B, Q, 1, 1]
        cx = reference_points[..., 0:1].detach()
        cy = reference_points[..., 1:2].detach()
        w = reference_points[..., 2:3].detach()
        h = reference_points[..., 3:4].detach()

        theta = rotate_norm_angles_to_angles(reference_points[..., 4:5].detach(), "le135")

        # align center/size to current valid region
        # center needs valid-ratio scaling; width/height should also scale per axis
        vr_w = valid_ratio[:, None, 0:1]   # [B, 1, 1]
        vr_h = valid_ratio[:, None, 1:2]   # [B, 1, 1]

        cx = cx * vr_w
        cy = cy * vr_h
        w = w * vr_w
        h = h * vr_h

        # local box grid in [-0.5, 0.5]
        ys = torch.linspace(
            -0.5 + 0.5 / pool_size,
            0.5 - 0.5 / pool_size,
            steps=pool_size,
            device=reference_points.device,
            dtype=reference_points.dtype,
        )
        xs = torch.linspace(
            -0.5 + 0.5 / pool_size,
            0.5 - 0.5 / pool_size,
            steps=pool_size,
            device=reference_points.device,
            dtype=reference_points.dtype,
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # [K, K]

        grid_x = grid_x.view(1, 1, pool_size, pool_size)   # [1,1,K,K]
        grid_y = grid_y.view(1, 1, pool_size, pool_size)

        # scale to box coordinates
        local_x = grid_x * w.unsqueeze(-1)   # [B,Q,K,K]
        local_y = grid_y * h.unsqueeze(-1)   # [B,Q,K,K]

        # rotate local coordinates
        cos_theta = torch.cos(theta).unsqueeze(-1)   # [B,Q,1,1]
        sin_theta = torch.sin(theta).unsqueeze(-1)

        rot_x = local_x * cos_theta - local_y * sin_theta
        rot_y = local_x * sin_theta + local_y * cos_theta

        # translate to global normalized coordinates
        abs_x = rot_x + cx.unsqueeze(-1)   # [B,Q,K,K]
        abs_y = rot_y + cy.unsqueeze(-1)

        # to [-1, 1] for grid_sample
        grid = torch.stack([abs_x, abs_y], dim=-1)   # [B,Q,K,K,2]
        grid = (grid * 2.0 - 1.0).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return grid


    def _sample_query_spectral_obs_center(
        self,
        spectral_weights: List[torch.Tensor],
        reference_points: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        Old version: center-point pooling.
        Returns: [B, Q, 8 * L]
        """
        query_centers = reference_points[..., :2].detach()   # [B, Q, 2]

        sampled_specs = []
        for lvl, lvl_spectral_weight in enumerate(spectral_weights):
            lvl_centers = query_centers * valid_ratios[:, None, lvl, :]   # [B,Q,2]
            lvl_grid = (lvl_centers * 2.0 - 1.0).clamp(
                min=-1.0 + 1e-6, max=1.0 - 1e-6
            ).unsqueeze(2)   # [B,Q,1,2]

            lvl_sampled = F.grid_sample(
                input=lvl_spectral_weight,   # [B,8,H,W]
                grid=lvl_grid,               # [B,Q,1,2]
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )   # [B,8,Q,1]

            lvl_sampled = lvl_sampled.squeeze(-1).transpose(1, 2)   # [B,Q,8]
            sampled_specs.append(lvl_sampled)

        pooled_spec = torch.cat(sampled_specs, dim=-1)   # [B,Q,8*L]
        return pooled_spec


    def _sample_query_spectral_obs_rbox(
        self,
        spectral_weights: List[torch.Tensor],
        reference_points: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        New version: rotated-box region pooling.
        Returns: [B, Q, 8 * L]
        """
        assert reference_points.shape[-1] == 5, \
            f"Rotated-box pooling requires 5-dim reference_points, got {reference_points.shape[-1]}"

        sampled_specs = []

        for lvl, lvl_spectral_weight in enumerate(spectral_weights):
            # [B, Q, K, K, 2]
            lvl_grid = self._build_rotated_box_sampling_grid(
                reference_points=reference_points,
                valid_ratio=valid_ratios[:, lvl, :],
            )

            B, Q, K, _, _ = lvl_grid.shape
            _, C, H, W = lvl_spectral_weight.shape

            # grid_sample input: [N, C, H, W], grid: [N, H_out, W_out, 2]
            # Trick: flatten query dimension into batch dimension
            lvl_feat = lvl_spectral_weight[:, None].expand(B, Q, C, H, W).reshape(B * Q, C, H, W)
            lvl_grid = lvl_grid.reshape(B * Q, K, K, 2)

            lvl_sampled = F.grid_sample(
                input=lvl_feat,      # [B*Q, 8, H, W]
                grid=lvl_grid,       # [B*Q, K, K, 2]
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )   # [B*Q, 8, K, K]

            # average pool inside rotated box
            lvl_sampled = lvl_sampled.mean(dim=(-1, -2))   # [B*Q, 8]
            lvl_sampled = lvl_sampled.view(B, Q, C)        # [B, Q, 8]
            sampled_specs.append(lvl_sampled)

        pooled_spec = torch.cat(sampled_specs, dim=-1)   # [B, Q, 8*L]
        return pooled_spec


    def _build_query_spectral_state_spec_only_center(
        self,
        spectral_weights: List[torch.Tensor],
        reference_points: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> torch.Tensor:
        pooled_spec = self._sample_query_spectral_obs_center(
            spectral_weights=spectral_weights,
            reference_points=reference_points,
            valid_ratios=valid_ratios,
        )
        q_spec = self.q_spec_obs_embed(pooled_spec)
        q_spec = self.q_spec_obs_norm(q_spec)
        return q_spec


    def _build_query_spectral_state_spec_only_rbox(
        self,
        spectral_weights: List[torch.Tensor],
        reference_points: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> torch.Tensor:
        pooled_spec = self._sample_query_spectral_obs_rbox(
            spectral_weights=spectral_weights,
            reference_points=reference_points,
            valid_ratios=valid_ratios,
        )
        q_spec = self.q_spec_obs_embed(pooled_spec)
        q_spec = self.q_spec_obs_norm(q_spec)
        return q_spec


    def _build_query_spectral_state_spec_tgt_fusion_rbox(
        self,
        spectral_weights: List[torch.Tensor],
        reference_points: torch.Tensor,
        tgt: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> torch.Tensor:
        pooled_spec = self._sample_query_spectral_obs_rbox(
            spectral_weights=spectral_weights,
            reference_points=reference_points,
            valid_ratios=valid_ratios,
        )
        pooled_spec_embed = self.q_spec_obs_norm(self.q_spec_obs_embed(pooled_spec))
        q_spec = self.q_spec_fuse(torch.cat([pooled_spec_embed, tgt], dim=-1))
        q_spec = self.q_spec_fuse_norm(q_spec)
        return q_spec


    def _build_query_spectral_state(
        self,
        spectral_weights: List[torch.Tensor],
        reference_points: torch.Tensor,
        tgt: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_spec source ablation:
        - spec_only_center
        - spec_only_rbox
        - spec_tgt_fusion_rbox
        """
        return self._build_query_spectral_state_spec_tgt_fusion_rbox(
                spectral_weights=spectral_weights,
                reference_points=reference_points,
                tgt=tgt,
                valid_ratios=valid_ratios,
            )
        # if self.q_spec_source == "spec_only_center":
        #     return self._build_query_spectral_state_spec_only_center(
        #         spectral_weights=spectral_weights,
        #         reference_points=reference_points,
        #         valid_ratios=valid_ratios,
        #     )
        # elif self.q_spec_source == "spec_only_rbox":
        #     return self._build_query_spectral_state_spec_only_rbox(
        #         spectral_weights=spectral_weights,
        #         reference_points=reference_points,
        #         valid_ratios=valid_ratios,
        #     )
        # elif self.q_spec_source == "spec_tgt_fusion_rbox":
        #     return self._build_query_spectral_state_spec_tgt_fusion_rbox(
        #         spectral_weights=spectral_weights,
        #         reference_points=reference_points,
        #         tgt=tgt,
        #         valid_ratios=valid_ratios,
        #     )
        # else:
        #     raise ValueError(f"Unsupported q_spec_source: {self.q_spec_source}")

    def forward(self, srcs: List[torch.Tensor], masks: List[torch.Tensor], pos_embeds: Optional[List[torch.Tensor]],
                query_embed, ref_pts, query_mask, spectral_weights: List[torch.Tensor],
                additional_tokens: List[torch.Tensor], additional_specs: List[torch.Tensor], additional_pos_embeds: List[torch.Tensor]):
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, spectral_embeds_flatten = [], [], [], [], []
        additional_token_nums, additional_tokens_flatten, additional_specs_flatten, additional_pos_embeds_flatten = [], [], [], []

        for lvl, (src, mask, pos_embed, spectral_weight, additional_token, additional_spec, additional_pos_embed) in enumerate(
            zip(srcs, masks, pos_embeds, spectral_weights, additional_tokens, additional_specs, additional_pos_embeds)
        ):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            spectral_embeds_flatten.append(self.spectral_embed(spectral_weight.flatten(2).transpose(1, 2)))
            lvl_pos_embed_flatten.append(pos_embed + self.level_embed[lvl].view(1, 1, -1))

            add_token_num = additional_token.shape[-1]
            additional_token_nums.append(add_token_num)
            additional_tokens_flatten.append(additional_token)
            additional_specs_flatten.append(self.spectral_embed(additional_spec))
            additional_pos_embed = additional_pos_embed.transpose(1, 2)
            additional_pos_embeds_flatten.append(additional_pos_embed + self.level_embed[lvl].view(1, 1, -1))

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spectral_embeds_flatten = torch.cat(spectral_embeds_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        add_tokens_flatten = torch.cat(additional_tokens_flatten, 1)
        add_specs_flatten = torch.cat(additional_specs_flatten, 1)
        add_pos_embeds_flatten = torch.cat(additional_pos_embeds_flatten, 1)
        add_token_nums = torch.as_tensor(additional_token_nums, dtype=torch.long, device=src_flatten.device)
        add_level_start_index = torch.cat((add_token_nums.new_zeros((1,)), add_token_nums.cumsum(0)[:-1]))

        memory, add_tokens = self.encoder(
            src=src_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios,
            pos=lvl_pos_embed_flatten, padding_mask=mask_flatten, spectral=spectral_embeds_flatten,
            add_tokens=add_tokens_flatten, add_specs=add_specs_flatten, add_pos_embeds=add_pos_embeds_flatten,
            add_level_start_index=add_level_start_index
        )
        bs, _, c = memory.shape

        if self.use_dab:
            tgt = query_embed
            query_embed = None
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=2)
        reference_points = ref_pts.sigmoid()
        init_reference_points = reference_points

        q_spec = self._build_query_spectral_state(
            spectral_weights=spectral_weights, reference_points=init_reference_points, tgt=tgt, valid_ratios=valid_ratios
        )


        # decoder 返回:
        # - layer_output_queries: (L, B, Q, C)
        # - layer_output_refs:    (L, B, Q, 2/5)
        # - layer_input_queries:  (L, B, Q, C)
        layer_output_queries, layer_output_refs, layer_input_queries = self.decoder(
            tgt=tgt, reference_points=init_reference_points, src=memory, src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index, src_valid_ratios=valid_ratios, query_pos=query_embed,
            query_mask=query_mask, src_padding_mask=mask_flatten, q_spec=q_spec
        )
        # TODO: expose q_spec for downstream tracking-state propagation in a future patch if needed.
        return layer_output_queries, init_reference_points, layer_output_refs, layer_input_queries

    def get_d_model(self):
        return self.d_model

    def get_n_dec_layers(self):
        return self.decoder.num_layers

    def set_refine_bbox_embed(self, bbox_embed: nn.Module):
        self.decoder.bbox_embed = bbox_embed

    def set_refine_angle_embed(self, angle_embed: nn.Module):
        self.decoder.angle_embed = angle_embed


def build(config: dict, rope_pos_module: Optional[nn.Module] = None):
    return DeformableTransformer(
        d_model=config["HIDDEN_DIM"], d_ffn=config["FFN_DIM"], n_feature_levels=config["NUM_FEATURE_LEVELS"],
        n_heads=config["NUM_HEADS"], n_enc_points=config["NUM_ENC_POINTS"], n_dec_points=config["NUM_DEC_POINTS"],
        n_enc_layers=config["NUM_ENC_LAYERS"], n_dec_layers=config["NUM_DEC_LAYERS"],
        merge_det_track_layer=0 if "MERGE_DET_TRACK_LAYER" not in config else config["MERGE_DET_TRACK_LAYER"],
        dropout=config["DROPOUT"], activation=config["ACTIVATION"], return_intermediate_dec=config["RETURN_INTER_DEC"],
        n_det_queries=config["NUM_DET_QUERIES"], extra_track_attn=config["EXTRA_TRACK_ATTN"], two_stage=False,
        use_checkpoint=config["USE_CHECKPOINT"], checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"], visualize=config["VISUALIZE"],
    )
