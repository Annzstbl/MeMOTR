
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hsmot.datasets.pipelines.channel import HeatmapFromRotateGt
from structures.track_instances import TrackInstances
from utils.GMC import compensate_rotated_boxes
from utils.nested_tensor import NestedTensor
from utils.utils import inverse_sigmoid

from ..backbone import BackboneWithPE
from ..backbone import build as build_backbone_with_pe
from ..backbone import build_woPe as build_backbone_woPe
from .deformable_transformer_20260317 import DeformableTransformer20260317
from .deformable_transformer_20260317 import build as build_deformable_transformer
from ..ffn import FFN
from ..mlp import MLP
from ..position_embedding_rope_nd import build as build_rope_pos
from ..query_updater import build as build_query_updater
from .SCEM_20260317 import build as build_scem
from ..utils import get_clones, logits_to_scores, pos_to_pos_embed
from torch.utils.checkpoint import checkpoint

from ..model_20260310.memotr_20260310 import MeMOTR20260310


class MeMOTR20260317(MeMOTR20260310):
    def __init__(self, backbone: BackboneWithPE, transformer: DeformableTransformer20260317,
                 query_updater: nn.Module,
                 num_classes: int, n_det_queries: int, n_feature_levels: int,
                 hidden_dim: int, ffn_dim: int, dropout: float,
                 scem_module: nn.Module,
                 aux_loss: bool = True, with_box_refine: bool = True,
                 use_checkpoint: bool = False, checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False,
                 ):
        '''
        
            20260310
            

        '''
        super(MeMOTR20260310, self).__init__()

        self.num_classes = num_classes
        self.n_det_queries = n_det_queries
        self.n_feature_levels = n_feature_levels
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize
        
        # Net:
        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater
        self.class_embed = nn.Linear(in_features=self.hidden_dim, out_features=num_classes)
        # Bounding box and angle embeddings
        self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
        self.angle_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=1, num_layers=3)  # 角度分支
        
        # Detection query embeddings and anchors
        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.n_det_queries, 5))  # (N_det, 5) 旋转框格式
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim))  # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim * 2))  # (N_det, 2C)
        

            
        # SCEM module
        self.scem_module = scem_module
        
        # GroupNorm for SCEM enhanced features
        # 转到scem_module中运行
        # self.scem_norms = nn.ModuleList([
        #     nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
        #     for _ in range(self.n_feature_levels)
        # ])

        assert self.n_feature_levels > 1
        n_backbone_inter_layers = backbone.n_inter_layers()
        n_backbone_inter_channels = backbone.n_inter_channels()
        feature_proj_list = []
        for i in range(n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[i], out_channels=self.hidden_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        for _ in range(self.n_feature_levels - n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[-1], out_channels=self.hidden_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        self.feature_projs = nn.ModuleList(feature_proj_list)
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Initialize class embedding for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # Initialize decoder refinement modules
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.angle_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.angle_embed.layers[-1].bias.data, 0)
        
        if self.with_box_refine:
            self.class_embed = get_clones(self.class_embed, self.transformer.get_n_dec_layers())
            self.bbox_embed = get_clones(self.bbox_embed, self.transformer.get_n_dec_layers())
            self.angle_embed = get_clones(self.angle_embed, self.transformer.get_n_dec_layers())

            # Initialize bbox and angle biases
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.angle_embed[0].layers[-1].bias.data, -math.log(3))  # le135下以水平框开始
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
            self.transformer.set_refine_angle_embed(self.angle_embed)
        else:
            raise NotImplementedError("Box refine is not implemented yet.")


        #* 每个lvl上由evidence_token和GlobalAverage token构成
        self.num_evidence_tokens = [8, 4, 2, 1]
        self.evidence_token_pos_embed = [nn.Parameter(torch.randn(self.hidden_dim, num+1)) for num in self.num_evidence_tokens]


    def forward(self, frame: NestedTensor, tracks: List[TrackInstances], 
            debug: bool = False, heatmap: Optional[torch.Tensor] = None, 
            gmc: Optional[torch.Tensor] = None):
        """
        Forward pass of MeMOTR.
        
        Args:
            frame: NestedTensor, shape = [B, C, H, W], input image frames
            tracks: List[TrackInstances], length = B, track instances for each batch
            debug: bool, whether to output debug information
            heatmap: Optional[Tensor], shape = [B, H, W], ground truth heatmap for SCEM
            gmc: Optional[Tensor], shape = [B, 2, 3], global motion compensation matrices
            
        Returns:
            dict: Dictionary containing predictions and intermediate results
        """
        # Extract features through backbone
        if self.use_checkpoint and self.checkpoint_level != 3:
            feature_tuple = checkpoint(self.backbone, frame, use_reentrant=False)
        else:
            feature_tuple = self.backbone(frame)

        # Unpack feature tuple
        pos = None
        spectral_weights = None
        features, pos, spectral_weights = feature_tuple

        # Project features to hidden_dim
        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.feature_projs[layer](src))
            masks.append(mask)
        
        # Generate additional feature levels if needed
        if self.n_feature_levels > len(srcs):
            srcs_len = len(srcs)
            for layer in range(srcs_len, self.n_feature_levels):
                if layer == srcs_len:
                    src = self.feature_projs[layer](features[-1].tensors)
                else:
                    src = self.feature_projs[layer](srcs[-1])
                mask = frame.masks
                mask = F.interpolate(mask[None, ...].float(), size=src.shape[-2:])[0].to(torch.bool)
                if pos is not None:
                    pos.append(self.backbone.position_embedding(NestedTensor(src, mask)).to(src.device))
                if spectral_weights is not None:
                    if self.backbone.weights_version == "v4":
                        spectral_weights.append(
                            self.backbone.spectral_embedding(spectral_weights[-1], NestedTensor(src, mask)).to(src.device)
                        )
                    else:
                        spectral_weights.append(
                            self.backbone.spectral_embedding(spectral_weights[0], NestedTensor(src, mask)).to(src.device)
                        )
                srcs.append(src)
                masks.append(mask)
        
        # srcs: n_feature_levels * [(B, C, H, W)]
        # masks: n_feature_levels * [(B, H, W)]
        # pos: n_feature_levels * [(B, C, H, W)]
        # spectral_weights: n_feature_levels * [(B, C=8, H, W)]

        # 生成额外的token送到encoder

        prior_map = self.get_prior_map(tracks=tracks, gmcs=gmc, frame=frame)


        # 更新srcs和spectral_weights
        (srcs, spectral_weights), (evidence_tokens, evidence_tokens_spectral_part), (global_token, global_token_spectral_part), (gamma, log_mix, spectral_dict), _ = self.scem_module(srcs, masks, prior_map=prior_map, specs=spectral_weights)

        additional_pos_embeds = [pos.unsqueeze(0).expand(srcs[0].shape[0], -1, -1).to(srcs[0].device) for pos in self.evidence_token_pos_embed]

        additional_tokens = [torch.cat((evi_token, global_token.unsqueeze(1)), dim=1) for evi_token, global_token in zip(evidence_tokens, global_token)]
        additional_specs = [torch.cat((evi_spec, global_spec.unsqueeze(1)), dim=1) for evi_spec, global_spec in zip(evidence_tokens_spectral_part, global_token_spectral_part)]

        # Prepare decoder queries
        reference_points = self.get_reference_points(tracks=tracks).to(srcs[0].device)  # (B, Nd+Nq, 2/5)
        query_embed = self.get_query_embed(tracks=tracks).to(srcs[0].device)
        query_mask = self.get_query_mask(tracks=tracks).to(srcs[0].device)  # (B, Nd+Nq)

        # DETR transformer forward
        transformer_kwargs = {
            "srcs": srcs,
            "masks": masks,
            "pos_embeds": pos,
            "spectral_weights": spectral_weights,
            "query_embed": query_embed,
            "ref_pts": reference_points,  # 值域: 负无穷到正无穷
            "query_mask": query_mask,
            "additional_tokens": additional_tokens,
            "additional_specs": additional_specs,
            "additional_pos_embeds": additional_pos_embeds,

        }

        # Transformer outputs:
        #   outputs: (n_dec_layers, B, Nd+Nq, C) - 每个layer输出的output embeddings
        #   init_reference: (B, Nd+Nq, 5) - 初始化reference_points, 值域[0,1]
        #   inter_references: (n_dec_layers, B, Nd+Nq, 5) - 每个layer输出的reference_points, 值域[0,1]
        #   inter_queries: (n_dec_layers, B, Nd+Nq, C) - 每个layer输入的query_embed
        #   init_query_spectral_weights: (B, Nd+Nq, 8) - 初始化query_spectral_weights, 值域[0,1]
        #   inter_query_spectral_weights: (n_dec_layers, B, Nd+Nq, 8) - 每个layer输出的query_spectral_weights, 值域[0,1]
        outputs, init_reference, inter_references, inter_queries = self.transformer(**transformer_kwargs)

        # Process transformer outputs
        # outputs: (n_dec_layers, B, Nd+Nq, C)
        # init_reference: (B, Nd+Nq, 2/5)
        # inter_references: (n_dec_layers, B, Nd+Nq, 2/5)
        output_classes, output_bboxes = [], []
        assert outputs.ndim == 4, (
            f"Deformable Transformer's outputs should have shape (n_dec_layers, B, Nd+Nq, C), "
            f"but got n_dim={outputs.ndim}"
        )

        for level in range(outputs.shape[0]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](outputs[level])
            bbox_tmp = self.bbox_embed[level](outputs[level])
            angle_tmp = self.angle_embed[level](outputs[level])

            bbox_tmp = torch.cat((bbox_tmp, angle_tmp), dim=-1)  # (..., 5)

            if reference.shape[-1] == 5:
                bbox_tmp += reference
            else:
                assert reference.shape[-1] == 2, f"Reference should have only 2 coord, but get {reference.shape[-1]}."
                bbox_tmp[..., :2] += reference
            output_bbox = bbox_tmp.sigmoid()
            output_classes.append(output_class)
            output_bboxes.append(output_bbox)
        output_classes = torch.stack(output_classes, dim=0) # (n_dec_layers, B, Nd+Nq, C)
        output_bboxes = torch.stack(output_bboxes, dim=0) # (n_dec_layers, B, Nd+Nq, 5)



        # Build result dictionary
        res = {
            "pred_logits": output_classes[-1],
            "pred_bboxes": output_bboxes[-1],
            # TODO 为什么？-2表示最后一层layer输入的reference_points（用于query更新）
            "last_ref_pts": inverse_sigmoid(inter_references[-2, :, :, :]),  # (B, Nd+Nq, 2/5)
            "query_mask": query_mask,  # (B, Nd+Nq)
            "det_query_embed": query_embed[0][:self.n_det_queries],
            "init_ref_pts": inverse_sigmoid(init_reference),
        }
        
        if self.aux_loss:

            # TODO 为什么？inter_queries是每个layer的输入，在set_aux_loss中会错位，
            # 使得每个queries对应每个layer的输出embedding
            res["aux_outputs"] = self.set_aux_loss(
                output_classes=output_classes,
                output_bboxes=output_bboxes,
                query_mask=query_mask,
                queries=inter_queries
            )
        
        res["outputs"] = outputs[-1]  # (B, Nd+Nq, C)
        res["spectral_weights"] = spectral_weights  # List[B, C=8, H, W]
        
        if debug:
            res["inter_references"] = inter_references
            res["inter_queries"] = inter_queries
            res["init_reference"] = init_reference
            res['all_outputs'] = outputs
            res["spectral_weights"] = spectral_weights

        res["scem_gamma"] = gamma
        res["scem_log_mix"] = log_mix
    
        return res



def build(config: dict) -> MeMOTR20260317:
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
        "hsmot_8ch": 8,
    }
    assert config["DATASET"] in dataset_num_classes, (
        f"Do not know the class num of {config['DATASET']} dataset."
    )
    num_classes = dataset_num_classes[config["DATASET"]]

    rope_pos = config["ROPE_POS"]
    if rope_pos:
        backbone = build_backbone_woPe(config=config)
        rope_pos_module = build_rope_pos(config=config)
    else:
        backbone = build_backbone_with_pe(config=config)
        rope_pos_module = None

    deformable_transformer: DeformableTransformer20260317 = build_deformable_transformer(
        config=config,
        rope_pos_module=rope_pos_module,
    )
    query_updater = build_query_updater(config=config)
    scem = build_scem(config=config["SCEM"])

    return MeMOTR20260317(
        backbone=backbone,
        transformer=deformable_transformer,
        query_updater=query_updater,
        num_classes=num_classes,
        n_det_queries=config["NUM_DET_QUERIES"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        aux_loss=True,
        with_box_refine=True,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"],
        scem_module=scem,
    )
