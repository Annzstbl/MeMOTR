# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from .mlp import MLP
from .ffn import FFN
from .backbone import BackboneWithPE
from .deformable_transformer import DeformableTransformer
from .query_updater import build as build_query_updater
from .utils import get_clones, pos_to_pos_embed

from .backbone import build as build_backbone_with_pe
from .backbone import build_woPe as build_backbone_woPe
from .position_embedding_rope_nd import build as build_rope_pos
from .deformable_transformer import build as build_deformable_transformer

from utils.nested_tensor import NestedTensor
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from .utils import logits_to_scores

from torch.utils.checkpoint import checkpoint
from .SCEM import build as build_scem
from .SCEM import SCEM
from utils.GMC import compensate_rotated_boxes
import numpy as np


class MeMOTR(nn.Module):
    def __init__(self, backbone: BackboneWithPE, transformer: DeformableTransformer,
                 query_updater: nn.Module,
                 num_classes: int, n_det_queries: int, n_feature_levels: int,
                 hidden_dim: int, ffn_dim: int, dropout: float,
                 aux_loss: bool = True, with_box_refine: bool = True,
                 use_checkpoint: bool = False, checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False,
                 decoder_spectral: bool = False,
                 decoder_spectral_refine: bool = False,
                 decoder_spectral_clusters: int = 1,
                 encoder_spectral: bool = False,
                 encoder_global_token: bool = False,
                 rope_pos_module = None,
                 scem_module = None,
                 scem_use_gt = False,
                 ): 
        super(MeMOTR, self).__init__()

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
        self.decoder_spectral = decoder_spectral
        self.decoder_spectral_refine = decoder_spectral_refine
        self.decoder_spectral_clusters = decoder_spectral_clusters
        self.encoder_global_token = encoder_global_token
        self.encoder_spectral = encoder_spectral
        
        # Net:
        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater
        self.class_embed = nn.Linear(in_features=self.hidden_dim, out_features=num_classes)
        self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
        self.angle_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=1, num_layers=3)#添加角度分支
        if self.decoder_spectral_refine:
            self.spectral_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.decoder_spectral_clusters * 8, num_layers=3)#refine spectral时候使用的

        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.n_det_queries, 5))  # (N_det, 4) #旋转框改成5
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim))       # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim * 2))   # (N_det, 2C)
        
        if self.decoder_spectral:
            self.det_spectral_anchor = nn.Parameter(torch.randn(self.n_det_queries, self.decoder_spectral_clusters* 8))  # (N_det, decoder_spectral_clusters* 8) # 8光谱              
        
        self.rope_pos_module = rope_pos_module
        if self.rope_pos_module is not None:
            self.rope_pos = True # enable bool
        else:
            self.rope_pos = False
            
        self.scem_module = scem_module
        self.use_scem = self.scem_module is not None
        self.scem_use_gt = scem_use_gt



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

        # 初始化cls_embed for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        #初始化decoder中的refine模块
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.angle_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.angle_embed.layers[-1].bias.data, 0)
        if self.decoder_spectral_refine:
            # nn.init.constant_(self.spectral_embed.layers[-1].weight.data, 0)
            # nn.init.constant_(self.spectral_embed.layers[-1].bias.data, 0)
            self.spectral_embed = get_clones(self.spectral_embed, self.transformer.get_n_dec_layers())
            self.transformer.set_refine_spectral_embed(self.spectral_embed)
        if self.with_box_refine:
            self.class_embed = get_clones(self.class_embed, self.transformer.get_n_dec_layers())
            self.bbox_embed = get_clones(self.bbox_embed, self.transformer.get_n_dec_layers())
            self.angle_embed = get_clones(self.angle_embed, self.transformer.get_n_dec_layers())

            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.angle_embed[0].layers[-1].bias.data, -math.log(3)) # le135下以水平框开始
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
            self.transformer.set_refine_angle_embed(self.angle_embed)
        else:
            raise NotImplementedError("Box refine is not implemented yet.")
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(self.transformer.get_n_dec_layers())])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(self.transformer.get_n_dec_layers())])

        # 初始化global_token的pos_embed
        if self.encoder_global_token:
            self.global_pos_embed = [nn.Parameter(torch.randn(self.hidden_dim)) for _ in range(n_feature_levels)]

    def enable_checkpoint(self, enable: bool):
        self.use_checkpoint = enable
        self.transformer.enable_checkpoint(enable)

    def forward(self, frame: NestedTensor, tracks: list[TrackInstances], debug=False, heatmap=None, gmc=None):
        """
            #解释每个参数
            frame: NestedTensor, shape = [B, C, H, W]
            tracks: list[TrackInstances], length = B 
            debug: bool, 是否打印调试信息
            heatmap: Tensor, shape = [B, H, W]
            gmc: Tensor, shape = [B, 2, 3]
        """
        # if self.visualize:
            # os.makedirs("./outputs/visualize_tmp/memotr/", exist_ok=True)

        # 图像经过 backbone
        if self.use_checkpoint and self.checkpoint_level != 3:
            feature_tuple = checkpoint(self.backbone, frame, use_reentrant=False)
        else:
            feature_tuple = self.backbone(frame)

        # 解包
        pos = None
        spectral_weights = None
        if self.rope_pos:
            if self.encoder_spectral:
                features, spectral_weights = feature_tuple
            else:
                features = feature_tuple
        else:
            if self.encoder_spectral:
                features, pos, spectral_weights = feature_tuple
            else:
                features, pos = feature_tuple

        # 特征统一映射到256维
        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.feature_projs[layer](src))
            masks.append(mask)
        
        # 新一stage的特征
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
                    spectral_weights.append(self.backbone.spectral_embedding(spectral_weights[0], NestedTensor(src, mask)).to(src.device))
                srcs.append(src)
                masks.append(mask)
        # srcs is n_feature_levels * [(B, C, H, W)]
        # masks is n_feature_levels * [(B, H, W)]
        # pos is n_features_levels * [(B, C, H, W)]
        # spectral_weights is n_features_levels * [(B, C=8, H, W)]

       
        prior_map = self.get_prior_map(tracks=tracks, gmcs=gmc, frame=frame)
        # scem增强
        if self.use_scem:
            if self.scem_module.prior_mode is not None:
                gamma, log_mix = self.scem_module(srcs, masks, prior_map=prior_map)
            else:
                gamma, log_mix = self.scem_module(srcs, masks)
            srcs = self.scem_module.apply_posterior_enhance(srcs, masks, gamma, alpha=0.5)
            
        if self.scem_use_gt:
            assert self.use_scem is False
            assert heatmap is not None
            srcs = SCEM.apply_posterior_enhance(srcs, masks, heatmap.unsqueeze(1), alpha=0.5)

        if self.encoder_global_token:
            # 通过平均值生成global_token
            global_token = []
            global_spectral_weights = []
            global_pos = [self.global_pos_embed[i].repeat(srcs[0].shape[0], 1).to(srcs[0].device) for i in range(self.n_feature_levels)]

            for _src, _maks, _spectral_weights in zip(srcs, masks, spectral_weights):
                valid_mask = (~_maks).unsqueeze(1)  # (B, 1, H, W)
                valid_count = valid_mask.sum(dim=(2, 3)).clamp(min=1)  # (B, 1)
                masked_sum = (_src * valid_mask).sum(dim=(2, 3))  # (B, C)
                global_token.append(masked_sum / valid_count)  # (B, C)
                masked_sum_sw = (_spectral_weights * valid_mask).sum(dim=(2, 3))  # (B, 8)
                global_spectral_weights.append(masked_sum_sw / valid_count)  # (B, 8)
        
        # decoder的query部分
        reference_points = self.get_reference_points(tracks=tracks).to(srcs[0].device)      # (B, Nd+Nq, 2/5)
        query_embed = self.get_query_embed(tracks=tracks).to(srcs[0].device)
        query_mask = self.get_query_mask(tracks=tracks).to(srcs[0].device)                  # (B, Nd+Nq)
        if self.decoder_spectral:
            query_spectral_weights = self.get_query_spectral_weights(tracks=tracks).to(srcs[0].device)
 


        # DETR:
        transformer_kwargs = {
            "srcs": srcs,
            "masks": masks,
            "pos_embeds": pos,
            "spectral_weights": spectral_weights,
            "query_embed": query_embed,
            "ref_pts": reference_points,#值域负无穷到正无穷
            "query_mask": query_mask,
        }

        if self.encoder_global_token:
            transformer_kwargs["global_token"] = global_token
            transformer_kwargs["global_spectral_weights"] = global_spectral_weights
            transformer_kwargs["global_pos_embeds"] = global_pos

        if self.decoder_spectral:
            transformer_kwargs["query_spectral_weights"] = query_spectral_weights #值域负无穷到正无穷
            '''
                outputs: (n_dec_layers, B, Nd+Nq, C)  每个layer输出的output embeddings
                init_reference: (B, Nd+Nq, 5) 初始化reference_points, 值域[0,1]
                inter_references: (n_dec_layers, B, Nd+Nq, 5) 每个layer输出的reference_points, 值域[0,1]
                inter_queries: (n_dec_layers, B, Nd+Nq, C) 每个layer《输入》的query_embed
                init_query_spectral_weights: (B, Nd+Nq, 8) 初始化query_spectral_weights, 值域[0,1]
                inter_query_spectral_weights: (n_dec_layers, B, Nd+Nq, 8) 每个layer输出的query_spectral_weights, 值域[0,1]
            '''
            # 返回的init_query_spectral_weights和inter_query_spectral_weights值域为0到1
            outputs, init_reference, inter_references, inter_queries, init_query_spectral_weights, inter_query_spectral_weights = self.transformer(**transformer_kwargs)
        else:
            outputs, init_reference, inter_references, inter_queries = self.transformer(**transformer_kwargs)

        # outputs: (n_dec_layers, B, Nd+Nq, C)
        # init_reference: (B, Nd+Nq, 2)
        # inter_references: (n_dec_layers, B, Nd+Nq, 4)
        output_classes, output_bboxes = [], []
        if self.decoder_spectral_refine:
            output_spectral_weights = []
        assert outputs.ndim == 4, f"Deformable Transformer's outputs should have shape (n_dec_layers, B, Nd+Nq, C, " \
                                  f"but get n_dim={outputs.ndim}"

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

        if self.decoder_spectral_refine:
            for level in range(outputs.shape[0]):
                if level == 0:
                    _query_spectral_weights = init_query_spectral_weights
                else:
                    _query_spectral_weights = inter_query_spectral_weights[level - 1]
                _query_spectral_weights = inverse_sigmoid(_query_spectral_weights)
                _output_spectral_weights = self.spectral_embed[level](outputs[level])
                _output_spectral_weights += _query_spectral_weights
                _output_spectral_weights = _output_spectral_weights.sigmoid()
                output_spectral_weights.append(_output_spectral_weights)
            output_spectral_weights = torch.stack(output_spectral_weights, dim=0) # (n_dec_layers, B, Nd+Nq, 8)

        res = {
            "pred_logits": output_classes[-1],
            "pred_bboxes": output_bboxes[-1],
            #TODO 这里是-2表示最后一层layer输入的reference_points，为什么呢？
            "last_ref_pts": inverse_sigmoid(inter_references[-2, :, :, :]) if self.use_dab       # (B, Nd+Nq, 4)
            else inverse_sigmoid(inter_references[-2, :, :, :]),                                 # (B, Nd+Nq, 2)
            "query_mask": query_mask,                   # (B, Nd+Nq)
            "det_query_embed": query_embed[0][:self.n_det_queries],
            "init_ref_pts": inverse_sigmoid(init_reference),
        }
        if self.aux_loss:
            if self.decoder_spectral_refine:
                res["aux_outputs"] = self.set_aux_loss_spectral_refine(output_classes=output_classes,
                                                                       output_bboxes=output_bboxes,
                                                                       query_mask=query_mask,
                                                                       queries=inter_queries,
                                                                       query_spectral_weights=output_spectral_weights)
            else:
                res["aux_outputs"] = self.set_aux_loss(output_classes=output_classes,
                                                    output_bboxes=output_bboxes,
                                                    query_mask=query_mask,
                                                    queries=inter_queries)#inter_queries是每个layer的输入，在set_aux_loss中会错位，使得每个queries是每个layer的输出embedding
        if self.decoder_spectral:
            res["last_query_spectral_weights"] = inverse_sigmoid(inter_query_spectral_weights[-2])#根据last_ref_pts的设置，也选择了取-2
            res["init_query_spectral_weights"] = inverse_sigmoid(init_query_spectral_weights)
            res["pred_spectral_weights"] = inter_query_spectral_weights[-1]
        res["outputs"] = outputs[-1]     # (B, Nd+Nq, C)
        res["spectral_weights"] = spectral_weights # List[B, C=8, H, W]
        if debug:
            if self.decoder_spectral:
                res["inter_query_spectral_weights"] = inter_query_spectral_weights
                res["init_query_spectral_weights"] = init_query_spectral_weights
            res["inter_references"] = inter_references
            res["inter_queries"] = inter_queries
            res["init_reference"] = init_reference
            res["outputs"] = outputs
            res["spectral_weights"] = spectral_weights

        if self.use_scem:
            res["scem_gamma"] = gamma
            res["scem_log_mix"] = log_mix
    
        return res

    @torch.jit.unused
    def set_aux_loss(self, output_classes, output_bboxes, query_mask, queries):
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes[:-1], output_bboxes[:-1], queries[1:])
        ]

    def set_aux_loss_spectral_refine(self, output_classes, output_bboxes, query_mask, queries, query_spectral_weights):
        return [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c, "pred_spectral_weights": d}
            for a, b, c, d in zip(output_classes[:-1], output_bboxes[:-1], queries[1:], query_spectral_weights[1:])
        ]

    def get_det_reference_points(self) -> torch.Tensor:
        """
        Returns: (Nd, 2)
        """
        if self.use_dab:
            return self.det_anchor
        else:
            return self.transformer.reference_points(self.det_query_embed[:, :self.hidden_dim])

    def get_track_reference_points(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2/4)
        """
        max_len = max([len(t.ref_pts) for t in tracks])
        if self.use_dab:
            references = torch.zeros((len(tracks), max_len, 5)) #for rotate
        else:
            # references = torch.zeros((len(tracks), max_len, 2))
            references = torch.zeros((len(tracks), max_len, 4))
        for i in range(len(tracks)):
            references[i, :len(tracks[i].ref_pts), :] = tracks[i].ref_pts
        return references

    def get_track_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2C)
        """
        max_len = max([len(t.query_embed) for t in tracks])
        if self.use_dab:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim))
        else:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim * 2))
        for i in range(len(tracks)):
            query_embed[i, :len(tracks[i].query_embed), :] = tracks[i].query_embed
        return query_embed

    def get_reference_points(self, tracks: list[TrackInstances]):
        '''
            检测的ref是det_anchor
            跟踪的ref是track.ref_pts
        '''
        det_references = self.get_det_reference_points().repeat(len(tracks), 1, 1)                      # (B, Nd, 2)
        if det_references.shape[-1] == 2:
            det_references = torch.cat(
                (det_references, torch.zeros_like(det_references, device=det_references.device)),
                dim=-1
            )
        track_references = self.get_track_reference_points(tracks=tracks).to(det_references.device)     # (B, Nq, 2)
        return torch.cat((det_references, track_references), dim=1)


    def get_prior_map(self, tracks: List[TrackInstances], gmcs: torch.Tensor, frame: NestedTensor) -> torch.Tensor:
        """
        根据 tracks 的位置和置信度生成先验热力图。
        
        Args:
            tracks: List[TrackInstances]，包含 ref_pts 和 logits
            gmcs: Tensor (B, 2, 3)的仿射矩阵
            frame: NestedTensor，用于获取图像尺寸
        
        Returns:
            prior_map: (B, H, W) 的先验热力图，尺寸匹配原图
        """
        # 获取图像尺寸
        H, W = frame.tensors.shape[-2:]
        B = len(tracks)
        device = frame.tensors.device
        
        prior_maps = []
        
        for b in range(B):
            if len(tracks[b]) == 0:
                # 如果没有轨迹，返回全零热力图
                prior_maps.append(torch.zeros((H, W), device=device))
                continue
            
            # 1. 从 tracks 中提取 ref_pts 和 logits
            ref_pts = tracks[b].ref_pts  # (N, 5) 或 (N, 4) 或 (N, 2)
            logits = tracks[b].logits    # (N, num_classes)
            
            # 2. 通过 sigmoid 得到归一化坐标，然后转换为物理坐标
            if ref_pts.shape[-1] == 5:
                # 旋转框格式 [cx, cy, w, h, angle]
                norm_boxes = ref_pts.sigmoid()  # (N, 5) 值域 [0, 1]
                # 转换为物理坐标
                boxes = norm_boxes.clone()
                boxes[:, 0] = boxes[:, 0] * W  # cx
                boxes[:, 1] = boxes[:, 1] * H  # cy
                boxes[:, 2] = boxes[:, 2] * W  # w
                boxes[:, 3] = boxes[:, 3] * H  # h
                # angle 已经是归一化的，需要转换为弧度
                # 根据 le135 版本：angle_range = 1, angle_offset = -1/4
                angle_range = math.pi
                angle_offset = -math.pi / 4
                boxes[:, 4] = boxes[:, 4] * angle_range + angle_offset
            # elif ref_pts.shape[-1] == 4:
            #     # 4维格式，假设是 [cx, cy, w, h]
            #     norm_boxes = ref_pts.sigmoid()  # (N, 4)
            #     boxes = norm_boxes.clone()
            #     boxes[:, 0] = boxes[:, 0] * W
            #     boxes[:, 1] = boxes[:, 1] * H
            #     boxes[:, 2] = boxes[:, 2] * W
            #     boxes[:, 3] = boxes[:, 3] * H
            #     # 添加角度维度（假设为0，即水平框）
            #     boxes = torch.cat([boxes, torch.zeros((boxes.shape[0], 1), device=device)], dim=-1)
            # elif ref_pts.shape[-1] == 2:
            #     # 2维格式，假设是 [cx, cy]
            #     norm_pts = ref_pts.sigmoid()  # (N, 2)
            #     boxes = norm_pts.clone()
            #     boxes[:, 0] = boxes[:, 0] * W
            #     boxes[:, 1] = boxes[:, 1] * H
            #     # 添加 w, h, angle（使用默认值）
            #     default_w = W * 0.1
            #     default_h = H * 0.1
            #     boxes = torch.cat([
            #         boxes,
            #         torch.full((boxes.shape[0], 1), default_w, device=device),
            #         torch.full((boxes.shape[0], 1), default_h, device=device),
            #         torch.zeros((boxes.shape[0], 1), device=device)
            #     ], dim=-1)
            else:
                raise ValueError(f"Unsupported ref_pts shape: {ref_pts.shape}")
            
            # 3. 使用 GMC 矩阵对坐标进行补偿
            if gmcs is not None and len(gmcs) > b:
                gmc_matrix = gmcs[b]  # (2, 3)
                if gmc_matrix is not None:
                    # 转换为 numpy 进行 GMC 补偿
                    boxes_np = boxes.detach().cpu().numpy()
                    gmc_matrix_np = gmc_matrix.detach().cpu().numpy() if isinstance(gmc_matrix, torch.Tensor) else gmc_matrix
                    # 补偿旋转框
                    boxes_compensated = compensate_rotated_boxes(boxes_np, gmc_matrix_np)
                    boxes = torch.from_numpy(boxes_compensated).to(device)
            
            # 4. 从 logits 中提取置信度分数
            scores = torch.max(logits_to_scores(logits=logits), dim=1).values  # (N,)
            
            # 5. 使用类似 HeatmapFromRotateGt 的方法生成热力图，根据置信度加权
            prior_map = self._generate_prior_heatmap(
                boxes=boxes,  # (N, 5) [cx, cy, w, h, angle_rad]
                scores=scores,  # (N,)
                img_shape=(H, W),
                device=device
            )
            # 大于均值的+0.5
            prior_map = prior_map + 0.5 * (prior_map > prior_map.mean())
            prior_map = prior_map.clamp_(0, 1)

            prior_maps.append(prior_map.detach())

        # 返回 (B, H, W)
        return torch.stack(prior_maps, dim=0)
    
    def _generate_prior_heatmap(self, boxes: torch.Tensor, scores: torch.Tensor, 
                                img_shape: tuple, device: torch.device,
                                mode: str = 'fixed_peak', peak: float = 0.5,
                                k: float = 5.0) -> torch.Tensor:
        """
        根据旋转框和置信度生成先验热力图。
        
        Args:
            boxes: (N, 5) [cx, cy, w, h, angle_rad]
            scores: (N,) 置信度分数
            img_shape: (H, W)
            device: 设备
            mode: 'fixed_peak' 或 'normalized'
            peak: 峰值（用于 fixed_peak 模式）
            k: 高斯核的倍数
        
        Returns:
            prior_map: (H, W) 热力图
        """
        H, W = img_shape
        N = boxes.shape[0]
        
        if N == 0:
            return torch.zeros((H, W), device=device)
        
        # 创建坐标网格
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')
        
        # 初始化累积图
        accum = torch.zeros((H, W), device=device)
        
        for i in range(N):
            box = boxes[i]  # [cx, cy, w, h, angle_rad]
            score = scores[i].item()
            
            if score <= 0:
                continue
            
            xc, yc = box[0].item(), box[1].item()
            w = max(box[2].item(), 1e-6)
            h = max(box[3].item(), 1e-6)
            th = box[4].item()
            c, s = math.cos(th), math.sin(th)
            
            # 计算 sigma（类似 HeatmapFromRotateGt）
            sigma_star = math.sqrt(1.0 / math.pi)
            s_size = math.sqrt(w * h)
            boundary_size = 8.0
            sigma_scalar = sigma_star * max(s_size / boundary_size, 1e-6)
            sig_w = sigma_scalar * (w / max(s_size, 1e-6))
            sig_h = sigma_scalar * (h / max(s_size, 1e-6))
            
            # 计算旋转后的 sigma 范围
            rx = k * math.sqrt((sig_w * c)**2 + (sig_h * s)**2)
            ry = k * math.sqrt((sig_w * s)**2 + (sig_h * c)**2)
            
            # 计算局部区域
            x0 = int(torch.clamp(torch.tensor(xc - rx, device=device), 0, W).item())
            x1 = int(torch.clamp(torch.tensor(xc + rx, device=device), 0, W).item())
            y0 = int(torch.clamp(torch.tensor(yc - ry, device=device), 0, H).item())
            y1 = int(torch.clamp(torch.tensor(yc + ry, device=device), 0, H).item())
            
            if x1 <= x0 or y1 <= y0:
                continue
            
            # 计算局部区域的偏差
            dx = X[y0:y1, x0:x1] - xc
            dy = Y[y0:y1, x0:x1] - yc
            
            # 旋转变换
            dxp = c * dx + s * dy
            dyp = -s * dx + c * dy
            
            # 计算高斯权重
            qf = (dxp / sig_w)**2 + (dyp / sig_h)**2
            
            if mode == 'fixed_peak':
                contrib = peak * torch.exp(-0.5 * qf)
            elif mode == 'normalized':
                denom = (2.0 * math.pi) * sig_w * sig_h
                contrib = torch.exp(-0.5 * qf) / denom
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            # 根据置信度加权# TODO 超参数
            contrib = contrib * score**0.5
            
            # 累积到总图中（使用 max 或 sum）
            accum[y0:y1, x0:x1] += contrib
            # accum[y0:y1, x0:x1] = torch.maximum(accum[y0:y1, x0:x1], contrib)
        
        # 归一化到 [0, 1]
        # if accum.max() > 0:
            # accum = accum / accum.max()
        
        return accum





    def get_det_spectral_weights(self):
        return self.det_spectral_anchor
    
    def get_track_spectral_weights(self, tracks: list[TrackInstances]):
        """
        Returns: (B, max_len, 8)
        获得所有batch中track的最长长度, 实际Batch = 1
        """
        max_len = max([len(t.query_spectral_weights) for t in tracks])
        spectral_weights = torch.zeros((len(tracks), max_len, self.decoder_spectral_clusters* 8))
        for i in range(len(tracks)):
            spectral_weights[i, :len(tracks[i].query_spectral_weights), :] = tracks[i].query_spectral_weights
        return spectral_weights
    
    def get_query_spectral_weights(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 8)
        检测: 预设的det_spectral_anchor
        跟踪: track.query_spectral_weights
        """
        det_spectral_weights = self.get_det_spectral_weights().repeat(len(tracks), 1, 1)
        track_references = self.get_track_spectral_weights(tracks=tracks).to(det_spectral_weights.device)
        return torch.cat((det_spectral_weights, track_references), dim=1)

    def get_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq, 2C)
        检测: 预设的det_query_embed
        跟踪: track.query_embed
        """
        if self.use_dab:
            det_query_embed = self.det_query_embed
            det_query_embed = det_query_embed.repeat(len(tracks), 1, 1)
        else:
            det_query_embed = self.det_query_embed.repeat(len(tracks), 1, 1)                    # (B, Nd, 2C)
        track_query_embed = self.get_track_query_embed(tracks).to(det_query_embed.device)       # (B, Nq, 2C)
        return torch.cat((det_query_embed, track_query_embed), dim=1)

    def get_query_mask(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nd+Nq)
        用于B>1时, 1个Batch内不同的帧之间track长度不同, 会构造最大长度的特征值。把补充的部分mask为1
        """
        track_max_len = max([len(t.query_embed) for t in tracks])
        det_query_mask = torch.zeros((len(tracks), self.n_det_queries)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed):] = 1
        track_query_mask = track_query_mask.to(torch.bool)
        return torch.cat((det_query_mask, track_query_mask), dim=1).to(self.det_query_embed.device)

    def postprocess_single_frame(self, previous_tracks: List[TrackInstances],
                                 new_tracks: List[TrackInstances],
                                 unmatched_dets: List[TrackInstances] | None,
                                 no_augment: bool = False):
        """
        Query updating.
        """
        return self.query_updater(previous_tracks, new_tracks, unmatched_dets, no_augment)


def build(config: dict):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
        "hsmot_8ch": 8,
    }
    assert config["DATASET"] in dataset_num_classes, f"Do not know the class num of {config['DATASET']} dataset."
    num_classes = dataset_num_classes[config["DATASET"]]

    rope_pos = config["ROPE_POS"]
    if rope_pos:
        backbone = build_backbone_woPe(config=config)
        rope_pos_module = build_rope_pos(config=config)

    else:
        backbone = build_backbone_with_pe(config=config)
        rope_pos_module = None

    # backbone_with_pe = build_backbone_with_pe(config=config)
    deformable_transformer = build_deformable_transformer(config=config, rope_pos_module=rope_pos_module)
    query_updater = build_query_updater(config=config)


    scem_gt = config["SCEM"]["USE_GT"]
    scem = build_scem(config = config["SCEM"])

    return MeMOTR(
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
        decoder_spectral=config["DECODER_SPECTRAL"],
        decoder_spectral_refine=config["DECODER_SPECTRAL_REFINE"],
        decoder_spectral_clusters=config["DECODER_SPECTRAL_CLUSTERS"], #decoder中spectral anchor的光谱数量
        encoder_global_token=config["ENCODER_GLOBAL_TOKEN"],
        encoder_spectral=config["ENCODER_SPECTRAL"],
        rope_pos_module=rope_pos_module,
        scem_module = scem,
        scem_use_gt = scem_gt,
    )
