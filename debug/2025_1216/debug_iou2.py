import torch
import math

# 你当前环境已经能用这些：
# 并且 oriented_box_intersection_2d 内部依赖的函数你这里能 import 到的话就导入：
from mmcv.ops.diff_iou_rotated import box_intersection, box_in_box, build_vertices, sort_indices, calculate_area, oriented_box_intersection_2d, box2corners

def debug_intersection_self(box_xywht, device="cuda"):
    """
    box_xywht: iterable of 5 numbers: x,y,w,h,theta(rad)
    """
    box = torch.tensor([[[*box_xywht]]], dtype=torch.float32, device=device).contiguous()  # (1,1,5)

    # 1) corners
    corners = box2corners(box).contiguous()  # (1,1,4,2)
    print("corners:", corners[0,0])
    pts = corners[0,0].detach().cpu().numpy()
    plot_corners_debug(pts, title="box corners", save_path="/data/users/wangying01/lth/hsmot/MeMOTR/debug/2025_1216/corners_debug.png")

    # 2) 直接调用最终接口
    inter, poly = oriented_box_intersection_2d(corners, corners)
    inter = inter.item()
    print("intersection(final):", inter)

    w = box[0,0,2].item()
    h = box[0,0,3].item()
    area = w * h
    print("w*h:", area, "ratio(inter/area):", inter/area)

    # 3) 如果你能 import 到内部函数：逐步打印中间结果
    try:
        intersections, valid_mask = box_intersection(corners, corners)
        c12, c21 = box_in_box(corners, corners)
        vertices, mask = build_vertices(corners, corners, c12, c21, intersections, valid_mask)
        sorted_indices = sort_indices(vertices, mask)
        inter2, poly2 = calculate_area(sorted_indices, vertices)

        print("intersection(step):", inter2.item(), "ratio:", inter2.item()/area)

        # 打印关键mask统计
        print("valid_mask sum:", valid_mask.sum().item(), " / ", valid_mask.numel())
        print("mask sum:", mask.sum().item(), " / ", mask.numel())

        # 打印 vertices 的有效点（非 padding）
        v = vertices[0,0]      # (9,2)
        m = mask[0,0]          # (9,)
        print("vertices valid:")
        for i in range(v.shape[0]):
            if m[i].item():
                print(i, v[i].tolist())
    except Exception as e:
        print("Cannot import internal steps or failed:", repr(e))

    # 4) 再做一个 CPU 对照（如果你的函数支持 CPU；很多 mmcv ops 不支持）
    return inter

import numpy as np
import matplotlib.pyplot as plt

def sort_corners_ccw_np(pts4x2: np.ndarray) -> np.ndarray:
    """按中心极角排序为 CCW 顺序（保证形成简单多边形）"""
    c = pts4x2.mean(axis=0, keepdims=True)
    v = pts4x2 - c
    ang = np.arctan2(v[:, 1], v[:, 0])
    idx = np.argsort(ang)  # CCW
    return pts4x2[idx], idx, c.squeeze(0)

def plot_corners_debug(pts4x2: np.ndarray, title="corners debug", save_path=None):
    """
    pts4x2: shape (4,2), float, corners from box2corners
    """
    pts = np.asarray(pts4x2, dtype=np.float64)

    # 原始顺序闭环
    loop_raw = np.vstack([pts, pts[0]])

    # 排序后的闭环
    pts_ccw, idx_ccw, center = sort_corners_ccw_np(pts)
    loop_ccw = np.vstack([pts_ccw, pts_ccw[0]])

    plt.figure()
    # 原始连线（你的 box2corners 输出顺序）
    plt.plot(loop_raw[:, 0], loop_raw[:, 1], marker='o')
    for i, (x, y) in enumerate(pts):
        plt.text(x, y, f"raw{i}", fontsize=10)

    # 排序后连线（参考）
    plt.plot(loop_ccw[:, 0], loop_ccw[:, 1], marker='o', linestyle='--')
    for j, (x, y) in enumerate(pts_ccw):
        plt.text(x, y, f"ccw{j}", fontsize=10)

    # 中心点
    plt.scatter([center[0]], [center[1]], marker='x')
    plt.text(center[0], center[1], "center", fontsize=10)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title + f"\nccw idx order: {idx_ccw.tolist()}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ===== 用法示例 =====
# 假设你已有 corners = box2corners(box) 形状 (1,1,4,2)
# pts = corners[0,0].detach().cpu().numpy()
# plot_corners_debug(pts, title="box corners", save_path="corners_debug.png")



# 你的那个样本
debug_intersection_self([1161.4, 891.88, 91.858, 14.04, -0.50274])
