# Copyright (c) Ruopeng Gao. All Rights Reserved.
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
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from utils.img_metas_utils import normalize_img_metas_list, get_img_shape, get_img_version
from utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    normalized_wasserstein_distance_cxcywh,
)
from structures.instances import Instances
from structures.track_instances import TrackInstances
from hsmot.util.dist import l1_dist_rotate, box_iou_rotated_norm_bboxes1
import torch.nn.functional as F
import math
import itertools
from utils.edge_swap import EdgeSwap

RECT_MEMOTR_VERSIONS = {"20260511_rect", "20260511_rect_gmc"}


def is_rect_memotr_version(version: str) -> bool:
    return version in RECT_MEMOTR_VERSIONS


def get_rect_box_similarity(config: dict) -> str:
    similarity = config.get("RECT_BOX_SIMILARITY", "giou").lower()
    if similarity not in {"giou", "nwd"}:
        raise ValueError(
            f"Unsupported RECT_BOX_SIMILARITY '{similarity}', only 'giou' and 'nwd' are supported."
        )
    return similarity


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    输入 outputs 和 targets，输出经过匈牙利算法之后的匹配结果。
    outputs 是一个字典，至少应该有如下两个字段：
        "pred_logits": 维度为 [B, n_det_queries, n_classes] 的 Tensor，表示分类的 logit。
        "pred_boxes": 维度为 [B, n_det_queries, 4] 的 Tensor，表示预测的 boxes 坐标，cxcywh 形式。
    targets 是一个长度为 B 的列表，每一个 item 是一个 dict，有如下字段：
        "labels": 维度为 [n_target_boxes] 的 Tensor，代表了 gt 的类别标签。
        "boxes": 维度为 [n_target_boxes, 4] 的 Tensor，代表了 gt 的 bbox 位置坐标。
    Returns 是一个长度为 B 的列表，每一个 item 是一个 tuple，有如下两个元素：
        index_i: 在 pred 中的索引序列。
        index_j: 在 target 中的索引序列。
        并且在每一个 batch 中，有如下的长度约定：len(index_i) = len(index_j) = min(n_det_queries, n_gts)
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_spectral_decoder_mse: float = 1,
                 edge_swap: bool = False,
                 rect_mode: bool = False,
                 rect_box_similarity: str = "giou"):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_spectral_decoder_mse = cost_spectral_decoder_mse
        self.edge_swap = edge_swap
        self.rect_mode = rect_mode
        self.rect_box_similarity = rect_box_similarity
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_spectral_decoder_mse != 0, "all costs cant be 0"

    def forward(self, outputs, targets, use_focal=True, img_metas=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            use_focal: use focal loss.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # img_metas 规范化为 list[dict]：兼容传入单个 dict（旧调用）或 list[dict]（推荐）。
            # 当 batch 内不同样本的 img_shape/version 可能不同（如 multi-scale 训练）时，
            # 必须传 list[dict]，以便 cost_bbox/cost_giou/edge_swap 用各样本自己的尺寸。
            img_metas_list = normalize_img_metas_list(img_metas, bs)

            if self.cost_spectral_decoder_mse != -1:
                assert 'pred_spectral_weights' in outputs, "pred_spectral_weights is not in outputs"
                pred_spectral_weights = outputs['pred_spectral_weights'].flatten(0, 1)  # [N, C]
                if isinstance(targets[0], Instances) or isinstance(targets[0], TrackInstances):
                    tgt_spectral_weights = torch.cat([gt_per_img.pred_spectral_weights for gt_per_img in targets])
                else:
                    tgt_spectral_weights = torch.cat([v["pred_spectral_weights"] for v in targets])
                tgt_spectral_weights = tgt_spectral_weights.sigmoid()  # [M, C]
                cost_spectral_decoder_mse = pairwise_min_permuted_segment_error(pred_spectral_weights, tgt_spectral_weights, reduction='mse', aggregate='mean')

            # cost_class 与 img_shape 无关，仍用 flatten 形式整 batch 计算
            if use_focal:
                out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs*Q, num_classes]

            # edge_swap 依赖各样本的 img_shape/version，必须按 b 处理
            out_bbox_per_b = []
            for b in range(bs):
                pred_b = outputs["pred_boxes"][b]
                if (not self.rect_mode) and self.edge_swap:
                    h_img, w_img = get_img_shape(img_metas_list, batch_idx=b)
                    pred_b = EdgeSwap.edge_swap(
                        pred_b, get_img_version(img_metas_list, batch_idx=b), (h_img, w_img)
                    )
                out_bbox_per_b.append(pred_b)
            out_bbox = torch.cat(out_bbox_per_b, dim=0)

            # Also concat the target labels and boxes
            if isinstance(targets[0], Instances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
                norm_tgt_bbox = torch.cat([gt_per_img.norm_boxes for gt_per_img in targets])
            elif isinstance(targets[0], TrackInstances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
                norm_tgt_bbox = torch.cat([gt_per_img.norm_boxes for gt_per_img in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])
                norm_tgt_bbox = torch.cat([v["norm_boxes"] for v in targets])

            # 提前计算 sizes（每个样本的 GT 数量）
            if isinstance(targets[0], Instances):
                sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
            elif isinstance(targets[0], TrackInstances):
                sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
            else:
                sizes = [len(v["boxes"]) for v in targets]

            # Compute the classification cost.
            if use_focal:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

            # cost_bbox / cost_giou 依赖各样本的 img_shape/version，按 b 计算并填入对角块；
            # 跨 b 的块保持 0（后续 C.split + c[i] 只取对角块，跨 b 部分会被丢弃）。
            total_M = tgt_bbox.size(0)
            cost_bbox = out_bbox.new_zeros((bs * num_queries, total_M))
            cost_giou = out_bbox.new_zeros((bs * num_queries, total_M))
            tgt_start = 0
            for b in range(bs):
                mb = sizes[b]
                tgt_end = tgt_start + mb
                if mb == 0:
                    tgt_start = tgt_end
                    continue
                pred_b = out_bbox_per_b[b]
                norm_tgt_b = norm_tgt_bbox[tgt_start:tgt_end]
                tgt_b = tgt_bbox[tgt_start:tgt_end]

                h_img, w_img = get_img_shape(img_metas_list, batch_idx=b)
                min_img_shape = min(h_img, w_img)
                if self.rect_mode:
                    l1_weight = torch.as_tensor(
                        [w_img / min_img_shape, h_img / min_img_shape,
                         w_img / min_img_shape, h_img / min_img_shape],
                        dtype=pred_b.dtype, device=pred_b.device,
                    )
                    cost_bbox_b = torch.abs(
                        pred_b[:, None, :] - norm_tgt_b[None, :, :]
                    ) * l1_weight
                    cost_bbox_b = cost_bbox_b.sum(dim=-1)
                    scale = pred_b.new_tensor([w_img, h_img, w_img, h_img])
                    pred_cxcywh = pred_b * scale
                    if self.rect_box_similarity == "nwd":
                        tgt_cxcywh = box_xyxy_to_cxcywh(tgt_b)
                        cost_giou_b = -normalized_wasserstein_distance_cxcywh(pred_cxcywh, tgt_cxcywh)
                    else:
                        pred_xyxy = box_cxcywh_to_xyxy(pred_b) * scale
                        cost_giou_b = -generalized_box_iou(pred_xyxy, tgt_b)
                else:
                    cost_bbox_b = l1_dist_rotate(pred_b, norm_tgt_b, aligned=False, cal_sum=False)
                    l1_weight = torch.as_tensor(
                        [w_img / min_img_shape, h_img / min_img_shape,
                         w_img / min_img_shape, h_img / min_img_shape, 1.0],
                        dtype=pred_b.dtype, device=pred_b.device,
                    )
                    cost_bbox_b = (cost_bbox_b * l1_weight).sum(dim=-1)
                    cost_giou_b = -box_iou_rotated_norm_bboxes1(
                        pred_b, tgt_b,
                        img_shape=(h_img, w_img),
                        version=get_img_version(img_metas_list, batch_idx=b),
                    )
                cost_bbox[b * num_queries:(b + 1) * num_queries, tgt_start:tgt_end] = cost_bbox_b
                cost_giou[b * num_queries:(b + 1) * num_queries, tgt_start:tgt_end] = cost_giou_b

                tgt_start = tgt_end

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            if self.cost_spectral_decoder_mse != -1:
                C = C + self.cost_spectral_decoder_mse * cost_spectral_decoder_mse

            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def pairwise_min_permuted_segment_error(
    A: torch.Tensor,
    B: torch.Tensor,
    C: int = 8,                 # 每段长度（例如 8）
    reduction: str = "mse",     # "mse" 或 "l1"：段内误差统计
    aggregate: str = "mean",    # "sum" 或 "mean"：对 k 段聚合
    perm_threshold: int = 7,     # k <= 该阈值时用向量化全置换，否则用匈牙利
    pairwise: bool = False      # True=逐个配对(N,)->(N,1)；False=所有配对(N,M)->(N,M)
) -> torch.Tensor:
    """
    计算 A∈R^{N×(kC)} 或 R^{N×k×C} 与 B∈R^{M×(kC)} 或 R^{M×k×C} 的最小置换误差矩阵，返回 R^{N×M}。
    - 将通道按段长 C 分段（或直接使用 3D 输入的最后一维 C）得到 k 段；
    - 段内误差按 reduction 计算（默认 MSE 段均值）；
    - 段间做最优一一匹配（置换），取最小总误差；
    - 对 k 段的总误差按 aggregate 聚合（默认 mean，即除以 k）。
    """
    assert A.dim() in (2, 3) and B.dim() in (2, 3), "A, B 维度必须为 2 或 3"

    device = A.device
    dtype = A.dtype

    # ---- 规范化到 (N, k, C) / (M, k, C) ----
    def _to_n_k_c(X: torch.Tensor, name: str):
        if X.dim() == 3:
            N, kx, Cx = X.shape
            assert Cx == C, f"{name} 最后一维必须等于 C={C}，但得到 {Cx}"
            return X, N, kx
        else:
            N, KC = X.shape
            assert KC % C == 0, f"{name} 的通道数 {KC} 必须能被 C={C} 整除"
            kx = KC // C
            return X.view(N, kx, C), N, kx

    Aseg, N, kA = _to_n_k_c(A, "A")
    Bseg, M, kB = _to_n_k_c(B, "B")
    assert kA == kB, f"A 和 B 的段数 k 必须相同，但得到 kA={kA}, kB={kB}"
    k = kA

    if pairwise:
        assert N == M, "pairwise 模式下，N 和 M 必须相等"

    if N == 0 or M == 0:
        return A.new_zeros((N, M)) if not pairwise else A.new_zeros((N,))


    # ---- 计算分段间 pairwise 误差: (N, M, k, k) ----
    # diff: (N, M/1 k, k, C)
    if pairwise:
        diff = Aseg[:,:,None,:] - Bseg[:,None,:,:] #(N, k, k, C)
        diff = diff.unsqueeze(1) #(N, 1, k, k, C)
    else:
        diff = Aseg[:, None, :, None, :] - Bseg[None, :, None, :, :]#(N,M,k,k,C)

    if reduction == "mse":
        seg_cost = (diff ** 2).mean(dim=-1)  # (N, M/1, k, k)
    elif reduction == "l1":
        seg_cost = diff.abs().mean(dim=-1)   # (N, M/1, k, k)
    else:
        raise ValueError("reduction 仅支持 'mse' 或 'l1'")

    if k == 1:
        return seg_cost.squeeze(3).squeeze(2)

    # ---- 对 k 段做最小置换匹配 ----
    if k <= perm_threshold and math.factorial(k) <= 40320:
        # 向量化全置换（适合 k<=7）
        perms = torch.tensor(
            list(itertools.permutations(range(k))),
            device=device, dtype=torch.long
        )  # (P, k)
        P = perms.shape[0]

        # seg_cost: (N, M/1, k, k) -> (N, M/1, P, k, k)
        seg_cost_exp = seg_cost.unsqueeze(2).expand(-1,-1,P,-1,-1)  # (N, M/1, P, k, k)
        # 索引列: (1,1,P,k,1)
        idx_col = perms.view(1, 1, P, k, 1)
        idx_col = idx_col.expand(N, M, P, k, 1) if not pairwise else idx_col.expand(N, 1, P, k, 1)
        # 选取置换对应的列 -> (N, M/1, P, k)
        chosen = seg_cost_exp.gather(dim=-1, index=idx_col).squeeze(-1)
        # 对行求和 -> (N, M/1, P)
        sums = chosen.sum(dim=-1)
        # 取最小置换 -> (N, M/1)
        min_cost, _ = sums.min(dim=2)
        if aggregate == "mean":
            min_cost = min_cost / k
        elif aggregate == "sum":
            pass
        else:
            raise ValueError("aggregate 仅支持 'sum' 或 'mean'")
        return min_cost.to(dtype=dtype) if not pairwise else min_cost.to(dtype=dtype).squeeze(1)

    else:
        # 匈牙利（逐 (n,m) 解 k×k 指派问题）
        seg_cost_np = seg_cost.detach().cpu().numpy()  # (N, M/1, k, k)
        out = torch.empty(seg_cost.shape[:2], device=device, dtype=dtype)
        for n in range(N):
            row_block = seg_cost_np[n]  # (M, k, k)
            for m in range(out.shape[1]):
                c = row_block[m]  # (k, k)
                ri, ci = linear_sum_assignment(c)
                val = c[ri, ci].sum()
                if aggregate == "mean":
                    val = val / k
                out[n, m] = val
        return out if not pairwise else out.squeeze(1)

def pairwise_min_permuted_segment_loss(pred_spectral_weights, tgt_spectral_weights, reduction='mse', aggregate_segment='mean', aggregate_loss="mean"):
    loss = pairwise_min_permuted_segment_error(pred_spectral_weights, tgt_spectral_weights, reduction=reduction, aggregate=aggregate_segment, pairwise=True)
    return loss.mean() if aggregate_loss == "mean" else loss.sum()




def build(config: dict):
    if "MATCH_COST_SPECTRAL_MSE" in config:
        cost_spectral_decoder_mse = config["MATCH_COST_SPECTRAL_MSE"]
    else:
        cost_spectral_decoder_mse = -1
    rect_mode = is_rect_memotr_version(config.get("MEMOTR_VERSION", ""))
    return HungarianMatcher(
        cost_class=config["MATCH_COST_CLASS"],
        cost_bbox=config["MATCH_COST_BBOX"],
        cost_giou=config["MATCH_COST_GIOU"],
        cost_spectral_decoder_mse=cost_spectral_decoder_mse,
        edge_swap=config.get("EDGE_SWAP", False),
        rect_mode=rect_mode,
        rect_box_similarity=get_rect_box_similarity(config),
    )
