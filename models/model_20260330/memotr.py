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
from .deformable_transformer import DeformableTransformer
from .deformable_transformer import build as build_deformable_transformer
from ..mlp import MLP
from ..position_embedding_rope_nd import build as build_rope_pos
from ..query_updater import build as build_query_updater
from .SCEM import build as build_scem
from ..utils import get_clones, logits_to_scores


class MeMOTR(nn.Module):
    def __init__(self, backbone: BackboneWithPE, transformer: DeformableTransformer,
                 query_updater: nn.Module,
                 num_classes: int, n_det_queries: int, n_feature_levels: int,
                 hidden_dim: int, ffn_dim: int, dropout: float,
                 scem_module: nn.Module,
                 aux_loss: bool = True, with_box_refine: bool = True,
                 use_checkpoint: bool = False, checkpoint_level: int = 2,
                 use_dab: bool = False, visualize: bool = False):
        super().__init__()

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

        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater
        self.class_embed = nn.Linear(in_features=self.hidden_dim, out_features=num_classes)
        self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
        self.angle_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=1, num_layers=3)

        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.n_det_queries, 5))
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim))
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim * 2))

        self.scem_module = scem_module

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

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.angle_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.angle_embed.layers[-1].bias.data, 0)

        if self.with_box_refine:
            self.class_embed = get_clones(self.class_embed, self.transformer.get_n_dec_layers())
            self.bbox_embed = get_clones(self.bbox_embed, self.transformer.get_n_dec_layers())
            self.angle_embed = get_clones(self.angle_embed, self.transformer.get_n_dec_layers())
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.angle_embed[0].layers[-1].bias.data, -math.log(3))
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
            self.transformer.set_refine_angle_embed(self.angle_embed)
        else:
            raise NotImplementedError("Box refine is not implemented yet.")

        self.num_evidence_tokens = [8, 4, 2, 1]
        self.evidence_token_pos_embed = [nn.Parameter(torch.randn(self.hidden_dim, num + 1)) for num in self.num_evidence_tokens]

    def forward(self, frame: NestedTensor, tracks: List[TrackInstances],
                debug: bool = False, heatmap: Optional[torch.Tensor] = None,
                gmc: Optional[torch.Tensor] = None):
        if self.use_checkpoint and self.checkpoint_level != 3:
            feature_tuple = checkpoint(self.backbone, frame, use_reentrant=False)
        else:
            feature_tuple = self.backbone(frame)

        pos = None
        spectral_weights = None
        features, pos, spectral_weights = feature_tuple

        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.feature_projs[layer](src))
            masks.append(mask)

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

        prior_map = self.get_prior_map(tracks=tracks, gmcs=gmc, frame=frame)
        # spectral_weights 作为第 3 个位置参数传入 specs：forward_hook 的 input 不含 keyword，
        # 若写 specs=... 则 hook 里只有 (srcs, masks)，不会出现 scem_module.in.2.*。
        scem_out = self.scem_module(
            srcs,
            masks,
            spectral_weights,
            prior_map=prior_map,
            return_debug=debug,
        )
        if len(scem_out) == 5:
            (srcs, spectral_weights), (evidence_tokens, evidence_tokens_spectral_part), (global_token, global_token_spectral_part), (gamma, log_mix, spectral_dict), scem_token_debug = scem_out
        else:
            (srcs, spectral_weights), (evidence_tokens, evidence_tokens_spectral_part), (global_token, global_token_spectral_part), (gamma, log_mix, spectral_dict) = scem_out
            scem_token_debug = None

        additional_pos_embeds = [p.unsqueeze(0).expand(srcs[0].shape[0], -1, -1).to(srcs[0].device) for p in self.evidence_token_pos_embed]
        additional_tokens = [torch.cat((evi_token, global_tok.unsqueeze(1)), dim=1) for evi_token, global_tok in zip(evidence_tokens, global_token)]
        additional_specs = [torch.cat((evi_spec, global_spec.unsqueeze(1)), dim=1) for evi_spec, global_spec in zip(evidence_tokens_spectral_part, global_token_spectral_part)]

        reference_points = self.get_reference_points(tracks=tracks).to(srcs[0].device)
        query_embed = self.get_query_embed(tracks=tracks).to(srcs[0].device)
        query_mask = self.get_query_mask(tracks=tracks).to(srcs[0].device)

        outputs, init_reference, inter_references, inter_queries = self.transformer(
            srcs=srcs,
            masks=masks,
            pos_embeds=pos,
            spectral_weights=spectral_weights,
            query_embed=query_embed,
            ref_pts=reference_points,
            query_mask=query_mask,
            additional_tokens=additional_tokens,
            additional_specs=additional_specs,
            additional_pos_embeds=additional_pos_embeds,
        )

        output_classes, output_bboxes = [], []
        for level in range(outputs.shape[0]):
            reference = init_reference if level == 0 else inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](outputs[level])
            bbox_tmp = self.bbox_embed[level](outputs[level])
            angle_tmp = self.angle_embed[level](outputs[level])
            bbox_tmp = torch.cat((bbox_tmp, angle_tmp), dim=-1)
            if reference.shape[-1] == 5:
                bbox_tmp += reference
            else:
                bbox_tmp[..., :2] += reference
            output_bbox = bbox_tmp.sigmoid()
            output_classes.append(output_class)
            output_bboxes.append(output_bbox)
        output_classes = torch.stack(output_classes, dim=0)
        output_bboxes = torch.stack(output_bboxes, dim=0)

        res = {
            "pred_logits": output_classes[-1],
            "pred_bboxes": output_bboxes[-1],
            "last_ref_pts": inverse_sigmoid(inter_references[-2, :, :, :]),
            "query_mask": query_mask,
            "det_query_embed": query_embed[0][:self.n_det_queries],
            "init_ref_pts": inverse_sigmoid(init_reference),
        }

        if self.aux_loss:
            res["aux_outputs"] = self.set_aux_loss(
                output_classes=output_classes,
                output_bboxes=output_bboxes,
                query_mask=query_mask,
                queries=inter_queries,
            )

        res["outputs"] = outputs[-1]
        res["spectral_weights"] = spectral_weights
        if debug:
            res["inter_references"] = inter_references
            res["inter_queries"] = inter_queries
            res["init_reference"] = init_reference
            res["all_outputs"] = outputs
            if scem_token_debug is not None:
                res["scem_token_debug"] = scem_token_debug
        res["scem_gamma"] = gamma
        res["scem_log_mix"] = log_mix
        return res

    def enable_checkpoint(self, enable: bool):
        self.use_checkpoint = enable
        self.transformer.enable_checkpoint(enable)

    @torch.jit.unused
    def set_aux_loss(self, output_classes, output_bboxes, query_mask, queries):
        return [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes[:-1], output_bboxes[:-1], queries[1:])
        ]

    def get_det_reference_points(self) -> torch.Tensor:
        if self.use_dab:
            return self.det_anchor
        return self.transformer.reference_points(self.det_query_embed[:, :self.hidden_dim])

    def get_track_reference_points(self, tracks: List[TrackInstances]) -> torch.Tensor:
        max_len = max([len(t.ref_pts) for t in tracks])
        references = torch.zeros((len(tracks), max_len, 5 if self.use_dab else 4))
        for i in range(len(tracks)):
            references[i, : len(tracks[i].ref_pts), :] = tracks[i].ref_pts
        return references

    def get_track_query_embed(self, tracks: List[TrackInstances]) -> torch.Tensor:
        max_len = max([len(t.query_embed) for t in tracks])
        query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim if self.use_dab else self.hidden_dim * 2))
        for i in range(len(tracks)):
            query_embed[i, : len(tracks[i].query_embed), :] = tracks[i].query_embed
        return query_embed

    def get_reference_points(self, tracks: List[TrackInstances]) -> torch.Tensor:
        det_references = self.get_det_reference_points().repeat(len(tracks), 1, 1)
        if det_references.shape[-1] == 2:
            det_references = torch.cat((det_references, torch.zeros_like(det_references, device=det_references.device)), dim=-1)
        track_references = self.get_track_reference_points(tracks=tracks).to(det_references.device)
        return torch.cat((det_references, track_references), dim=1)

    def get_prior_map(self, tracks: List[TrackInstances], gmcs: torch.Tensor, frame: NestedTensor) -> torch.Tensor:
        height, width = frame.tensors.shape[-2:]
        batch_size = len(tracks)
        device = frame.tensors.device
        prior_maps = []
        for b in range(batch_size):
            if len(tracks[b]) == 0:
                prior_maps.append(torch.zeros((height, width), device=device))
                continue
            ref_pts = tracks[b].ref_pts
            logits = tracks[b].logits
            if ref_pts.shape[-1] == 5:
                norm_boxes = ref_pts.sigmoid()
                boxes = norm_boxes.clone()
                boxes[:, 0] = boxes[:, 0] * width
                boxes[:, 1] = boxes[:, 1] * height
                boxes[:, 2] = boxes[:, 2] * width
                boxes[:, 3] = boxes[:, 3] * height
                boxes[:, 4] = boxes[:, 4] * math.pi - math.pi / 4
            else:
                raise ValueError(f"Unsupported ref_pts shape: {ref_pts.shape}")

            if gmcs is not None and len(gmcs) > b:
                gmc_matrix = gmcs[b]
                if gmc_matrix is not None:
                    boxes_np = boxes.detach().cpu().numpy()
                    gmc_matrix_np = gmc_matrix.detach().cpu().numpy() if isinstance(gmc_matrix, torch.Tensor) else gmc_matrix
                    boxes = torch.from_numpy(compensate_rotated_boxes(boxes_np, gmc_matrix_np)).to(device)

            scores = torch.max(logits_to_scores(logits=logits), dim=1).values
            prior_map = HeatmapFromRotateGt.heatmap_from_rotate_gt_xywha(
                gt_xywha=boxes, img_shape=(height, width), version="le135", scores=scores,
                mode="fixed_peak", peak=1.0, reduce="sum", k=5.0
            )
            prior_maps.append(prior_map.clamp_(0, 1).detach())
        return torch.stack(prior_maps, dim=0)

    def get_query_embed(self, tracks: List[TrackInstances]) -> torch.Tensor:
        det_query_embed = self.det_query_embed.repeat(len(tracks), 1, 1)
        track_query_embed = self.get_track_query_embed(tracks).to(det_query_embed.device)
        return torch.cat((det_query_embed, track_query_embed), dim=1)

    def get_query_mask(self, tracks: List[TrackInstances]) -> torch.Tensor:
        track_max_len = max([len(t.query_embed) for t in tracks])
        det_query_mask = torch.zeros((len(tracks), self.n_det_queries)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed):] = 1
        return torch.cat((det_query_mask, track_query_mask.to(torch.bool)), dim=1).to(self.det_query_embed.device)

    def postprocess_single_frame(self, previous_tracks: List[TrackInstances], new_tracks: List[TrackInstances],
                                 unmatched_dets: Optional[List[TrackInstances]], no_augment: bool = False) -> List[TrackInstances]:
        return self.query_updater(previous_tracks, new_tracks, unmatched_dets, no_augment)


def build(config: dict) -> MeMOTR:
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

    if config["ROPE_POS"]:
        backbone = build_backbone_woPe(config=config)
        rope_pos_module = build_rope_pos(config=config)
    else:
        backbone = build_backbone_with_pe(config=config)
        rope_pos_module = None

    deformable_transformer: DeformableTransformer = build_deformable_transformer(config=config, rope_pos_module=rope_pos_module)
    query_updater = build_query_updater(config=config)
    scem = build_scem(config=config["SCEM"])

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
        scem_module=scem,
    )
