import torch.nn as nn
from typing import Optional

from ..model_20260310.deformable_transformer_20260310 import DeformableTransformer20260310
from ..model_20260317.deformable_encoder_20260317 import DeformableEncoder20260317
from ..model_20260317.deformable_encoder_20260317 import DeformableEncoderLayer20260317
from ..model_20260317.deformable_decoder_20260317 import DeformableDecoder20260317
from ..model_20260317.deformable_decoder_20260317 import DeformableDecoderLayer20260317
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from typing import List, Optional
from ..ops.modules import MSDeformAttn, MSDeformAttnSpectral, MSDeformAttn_Rotate
from ..mlp import MLP
from deprecated.sphinx import deprecated
import torch
import math
import torch.nn as nn

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from typing import List, Optional

class DeformableTransformer20260317(DeformableTransformer20260310):
    def __init__(self, d_model=256, d_ffn=1024,
                 n_feature_levels=4, n_heads=8,
                 n_enc_points=4, n_dec_points=4,
                 n_enc_layers=6, n_dec_layers=6,
                 merge_det_track_layer=0,
                 dropout=0.1, activation="ReLU",
                 return_intermediate_dec=False,
                 n_det_queries=300,
                 extra_track_attn=False,
                 two_stage=False, two_stage_num_proposals=300,
                 use_checkpoint: bool = False,
                 checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False,):
        """
        Args:
            d_model:
            d_ffn:
            n_feature_levels:
            n_heads:
            n_enc_points:
            n_dec_points:
            n_enc_layers:
            n_dec_layers:
            dropout:
            activation:
            return_intermediate_dec:
            n_det_queries:
            extra_track_attn:
            two_stage:
            two_stage_num_proposals:
            visualize
        """
        super(DeformableTransformer20260310, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize
        
        encoder_layer = DeformableEncoderLayer20260317(
                            d_model=d_model, d_ffn=d_ffn,
                            dropout=dropout, activation=activation,
                            n_levels=n_feature_levels, n_heads=n_heads,
                            n_points=n_enc_points, sigmoid_attn=False,
                        )
        self.encoder: DeformableEncoder20260317 = DeformableEncoder20260317(
            encoder_layer=encoder_layer,
            num_layers=n_enc_layers,
            use_checkpoint=(self.use_checkpoint and self.checkpoint_level == 1)
        )

        decoder_layer = DeformableDecoderLayer20260317(
            d_model=d_model, d_ffn=d_ffn,
            dropout=dropout, activation=activation,
            n_levels=n_feature_levels, n_heads=n_heads,
            n_points=n_dec_points, sigmoid_attn=False,
            extra_track_attn=extra_track_attn, n_det_queries=n_det_queries,
            visualize=self.visualize
        )
        self.decoder: DeformableDecoder20260317 = DeformableDecoder20260317(decoder_layer=decoder_layer, num_layers=n_dec_layers,
                                                            return_intermediate=return_intermediate_dec,
                                                            merge_det_track_layer=merge_det_track_layer,
                                                            n_det_queries=n_det_queries,
                                                            d_model=self.d_model,
                                                            use_checkpoint=self.use_checkpoint,
                                                            use_dab=self.use_dab,
                                                            visualize=self.visualize)

            # lvl embedding
        self.level_embed = nn.Parameter(torch.Tensor(n_feature_levels, d_model))


        # 把光谱权重映射为d_model的embedding
        self.spectral_embed = MLP(
                input_dim=8,
                hidden_dim=self.d_model,
                output_dim=self.d_model,
                num_layers=2
            )

        if two_stage:
            assert False, "two stage is not supported"
            # 目前不关心 two stage 的情况
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if use_dab:
                pass
            else:
                assert False, "use_dab is not supported"
                self.reference_points = nn.Linear(d_model, 2)

        self.reset_parameters()



    def forward(self, 
                srcs: List[torch.Tensor], 
                masks: List[torch.Tensor],
                pos_embeds: Optional[List[torch.Tensor]], 
                query_embed, 
                ref_pts, 
                query_mask,

                spectral_weights: List[torch.Tensor], 

                additional_tokens: List[torch.Tensor],
                additional_specs: List[torch.Tensor],
                additional_pos_embeds: List[torch.Tensor],
                ):
        '''
            src: feature from backbome
            
            masks: 解决一个batch图像大小不一样的问题
            
            pos_embes: feature的位置编码
            
            query_embed: 输入decoder的query, 包括检测和跟踪两部分
            
            ref_pts: 输入decoder的ref_pts, 包括检测和跟踪两部分
            
            query_mask: 用于解决一个batch跟踪目标数量不一样的mask

            additional_tokens: lvl * [B, C, 2]
            additional_specs: lvl * [B, 8, 2]
            additional_pos_embed: lvl * [B, C, 2]
            
        '''
    
        
        assert self.two_stage or query_embed is not None

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        spectral_embeds_flatten = []

        additional_token_nums = []
        additional_tokens_flatten = []
        additional_specs_flatten = []
        additional_pos_embeds_flatten = []


        # 展平, 位置编码加上层级权重, encoder光谱权重映射为特征
        for lvl, (src, mask, pos_embed, spectral_weight, additional_token, additional_spec, additional_pos_embed) in enumerate(zip(srcs, masks, pos_embeds, spectral_weights, additional_tokens, additional_specs, additional_pos_embeds)):
            # src.shape = (B, C, H, W) in lvl level.
            # mask.shape = (B, H, W) in lvl level.
            # pos_embed.shape = (B, C, H, W) in lvl level.
            bs, c, h, w = src.shape

            spatial_shape = (h, w)
            src = src.flatten(2).transpose(1, 2)                # (B, H*W, C)
            mask = mask.flatten(1)                              # (B, H*W)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # (B, H*W, C), same as src.

            spectral_embed = self.spectral_embed(spectral_weight.flatten(2).transpose(1, 2)) # (B, H*W, C)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    # (B, H*W, C)

            spatial_shapes.append(spatial_shape)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            spectral_embeds_flatten.append(spectral_embed) 

            #* 处理add_token
            add_token_num = additional_token.shape[-1] 
            # additional_token #(B, num_add_token, C)
            # additional_spec #(B, num_add_token, C)
            additional_spec_embed = self.spectral_embed(additional_spec)
            additional_pos_embed = additional_pos_embed.transpose(1, 2) #(B, num_add_token, C)
            additional_lvl_pos_embed = additional_pos_embed + self.level_embed[lvl].view(1, 1, -1) #(B, num_add_token, C)

            additional_token_nums.append(add_token_num)
            additional_tokens_flatten.append(additional_token)
            additional_specs_flatten.append(additional_spec_embed)
            additional_pos_embeds_flatten.append(additional_lvl_pos_embed)
            

        src_flatten = torch.cat(src_flatten, 1) #(B, sum(W_l * H_l), C)
        mask_flatten = torch.cat(mask_flatten, 1)
        spectral_embeds_flatten = torch.cat(spectral_embeds_flatten, 1) #(B, sum(W_l * H_l), C)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)#(B, sum(W_l * H_l), C)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)   # (n_levels, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))                          # (n_levels, )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)         # (B, n_levels, 2)

        add_tokens_flatten = torch.cat(additional_tokens_flatten, 1) #(B, sum(num_add_token), C)
        add_specs_flatten = torch.cat(additional_specs_flatten, 1) #(B, sum(num_add_token), C)
        add_pos_embeds_flatten = torch.cat(additional_pos_embeds_flatten, 1) #(B, sum(num_add_token), C)
        add_token_nums = torch.as_tensor(additional_token_nums, dtype=torch.long, device=src_flatten.device) #(n_levels, )
        add_level_start_index = torch.cat((add_token_nums.new_zeros((1,)),
                                       add_token_nums.cumsum(0)[:-1]))                          # (n_levels, )


        #TODO 这里的add_tokens是否还有用处？
        if self.use_checkpoint and (self.checkpoint_level == 2 or self.checkpoint_level == 3):
            from torch.utils.checkpoint import checkpoint
            memory, add_tokens = checkpoint(self.encoder, src_flatten, spatial_shapes, level_start_index,
                                valid_ratios, lvl_pos_embed_flatten, mask_flatten, spectral_embeds_flatten, use_reentrant=False,
                                add_tokens=add_tokens_flatten,
                                add_specs=add_specs_flatten,
                                add_pos_embeds=add_pos_embeds_flatten,
                                add_level_start_index=add_level_start_index,
            )
        else:
            memory, add_tokens = self.encoder(
                src=src_flatten, 
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index, 
                valid_ratios=valid_ratios,
                pos=lvl_pos_embed_flatten, 
                padding_mask=mask_flatten, 
                spectral=spectral_embeds_flatten,
                add_tokens=add_tokens_flatten,
                add_specs=add_specs_flatten,
                add_pos_embeds=add_pos_embeds_flatten,
                add_level_start_index=add_level_start_index,
            )
        bs, _, c = memory.shape
            
        # decoder
        if self.two_stage:
            raise RuntimeError(f"Do not support two stage model for Deformable Transformer.")
        else:
            if self.use_dab:
                tgt = query_embed
                query_embed = None
            else:
                query_embed, tgt = torch.split(query_embed, c, dim=2)   # (B, Nq, C), (B, Nq, C)
            assert ref_pts is not None, "ref_pts should not be None."
            reference_points = ref_pts.sigmoid() #! 注意这里sigmoid了
            init_reference_points = reference_points # 保留的初始化reference_points 取值[0,1]


        output, res_reference_points, inter_queries = self.decoder(
            tgt=tgt,
            reference_points=init_reference_points,
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=query_embed,
            query_mask=query_mask,
            src_padding_mask=mask_flatten
        )
        # if inter is Ture, shape = (n_layers, B, Nq, C), (n_layers, B, Nq, 4), (n_layer, B, Nq, 2C)
        # else, shape = (B, Nq, C), (B, Nq, 4)
        return output, init_reference_points, res_reference_points, inter_queries
        # output:   if inter is Ture, (n_layers, B, Nq, C)
        #           else,             (B, Nq, C)
        # init_reference_points, (B, Nq, 2/4)
        # res_reference_points: if inter is True, (n_layers, B, Nq, 4)
        #                       else,             (B, Nq, 4)
        # inter_query_spectral_weights: if inter is True, (n_layers, B, Nq, 8)
        #                                else,             (B, Nq, 8)






DeformableTransformer = DeformableTransformer20260317


def build(config: dict, rope_pos_module: Optional[nn.Module] = None):
    return DeformableTransformer20260317(
        d_model=config["HIDDEN_DIM"],
        d_ffn=config["FFN_DIM"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        n_heads=config["NUM_HEADS"],
        n_enc_points=config["NUM_ENC_POINTS"],
        n_dec_points=config["NUM_DEC_POINTS"],
        n_enc_layers=config["NUM_ENC_LAYERS"],
        n_dec_layers=config["NUM_DEC_LAYERS"],
        merge_det_track_layer=0 if "MERGE_DET_TRACK_LAYER" not in config else config["MERGE_DET_TRACK_LAYER"],
        dropout=config["DROPOUT"],
        activation=config["ACTIVATION"],
        return_intermediate_dec=config["RETURN_INTER_DEC"],
        n_det_queries=config["NUM_DET_QUERIES"],
        extra_track_attn=config["EXTRA_TRACK_ATTN"],
        two_stage=False,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"],
    )
