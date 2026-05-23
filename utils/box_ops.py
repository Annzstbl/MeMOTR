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
from torchvision.ops.boxes import box_area


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    boxes = [
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        (x2 - x1),
        (y2 - y1)
    ]
    return torch.stack(boxes, dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    boxes = [
        (cx - 0.5 * w),
        (cy - 0.5 * h),
        (cx + 0.5 * w),
        (cy + 0.5 * h)
    ]
    return torch.stack(boxes, dim=-1)


def box_cxcywh_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    boxes = [
        (cx - 0.5 * w),
        (cy - 0.5 * h),
        w,
        h
    ]
    return torch.stack(boxes, dim=-1)


def box_iou_union(boxes1, boxes2):
    area1 = box_area(boxes1)    # [N, ]
    area2 = box_area(boxes2)    # [M, ]
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou_union(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def normalized_wasserstein_distance_cxcywh(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Normalized Wasserstein Distance (NWD) similarity for axis-aligned boxes.

    Each box is modeled as a 2D Gaussian; see AI-TOD (Tiny Object Detection).
    boxes1: [N, 4] cxcywh, boxes2: [M, 4] cxcywh (same pixel coordinate system).
    Returns [N, M] similarity in (0, 1], higher is better.
    """
    center_distance = ((boxes1[:, None, :2] - boxes2[None, :, :2]) ** 2).sum(-1)
    wh_distance = ((boxes1[:, None, 2:] - boxes2[None, :, 2:]) ** 2).sum(-1) / 4
    wasserstein = center_distance + wh_distance
    wh1 = boxes1[:, None, 2:]  # [N, 1, 2]
    wh2 = boxes2[None, :, 2:]  # [1, M, 2]
    constant = torch.max(
        wh1.max(dim=-1).values,
        wh2.max(dim=-1).values,
    ).clamp(min=eps)
    return torch.exp(-torch.sqrt(wasserstein + eps) / constant)
