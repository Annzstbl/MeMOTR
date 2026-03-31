import torch
import math
from mmcv.ops import diff_iou_rotated_2d as diff_iou_rotated_2d_mmcv
from mmcv.ops.diff_iou_rotated import box2corners, oriented_box_intersection_2d

from hsmot.util.diff_iou_rotated_util import diff_iou_rotated_2d as diff_iou_rotated_2d_hsmot
from hsmot.util.diff_iou_rotated_util import box2corners as box2corners_hsmot
from hsmot.util.diff_iou_rotated_util import oriented_box_intersection_2d as oriented_box_intersection_2d_hsmot

# ===== 用你的实现 =====
# from xxx import box2corners, diff_iou_rotated_2d

box = torch.tensor(
    [[[1161.4, 891.88, 91.858, 14.04, -0.50274]]],
    dtype=torch.float32,
    device='cuda'
)  # (1,1,5)

# IoU with itself

iou = diff_iou_rotated_2d_mmcv(box, box)
print("IoU(box, box) =", iou.item())

corners = box2corners(box)  # (1,1,4,2)
pts = corners[0, 0].cpu().numpy()

print("Corners (x,y):")
for i, p in enumerate(pts):
    print(f"P{i}: {p}")

def polygon_area(pts):
    # pts: (4,2) numpy
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(
        (x * np.roll(y, -1)).sum() -
        (y * np.roll(x, -1)).sum()
    )

import numpy as np

area_poly = polygon_area(pts)
area_wh = 91.858 * 14.04

print("Polygon area =", area_poly)
print("w*h          =", area_wh)
print("ratio        =", area_poly / area_wh)

# 单独算 intersection
corners1 = box2corners(box)
corners2 = box2corners(box)

intersection, _ = oriented_box_intersection_2d(corners1, corners2)

print("intersection =", intersection.item())
print("area1 =", area_wh)
print("IoU manual =", intersection.item() / (2*area_wh - intersection.item()))


# ===== hsmot实现 ===== #
print("===================hsmot实现===================")
iou_hsmot = diff_iou_rotated_2d_hsmot(box, box)
print("IoU(box, box) =", iou_hsmot.item())

corners1_hsmot = box2corners_hsmot(box)
pts = corners1_hsmot[0, 0].cpu().numpy()
print("Corners (x,y):")
for i, p in enumerate(pts):
    print(f"P{i}: {p}")

area_poly_hsmot = polygon_area(pts)
area_wh_hsmot = 91.858 * 14.04

print("Polygon area =", area_poly_hsmot)
print("w*h          =", area_wh_hsmot)
print("ratio        =", area_poly_hsmot / area_wh_hsmot)

corners1_hsmot = box2corners_hsmot(box)
corners2_hsmot = box2corners_hsmot(box)
intersection_hsmot, _ = oriented_box_intersection_2d_hsmot(corners1_hsmot, corners2_hsmot)
print("intersection_hsmot =", intersection_hsmot.item())
print("area1_hsmot =", area_wh_hsmot)
print("IoU manual_hsmot =", intersection_hsmot.item() / (2*area_wh_hsmot - intersection_hsmot.item()))
