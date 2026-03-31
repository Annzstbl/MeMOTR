"""
Prob IoU 单元测试：probiou / batch_probiou / probiou_loss（xywhr 格式）。

运行方式（在项目根或 MeMOTR 下）:
  python MeMOTR/debug/test_iou/test_prob_iou.py
  python debug/test_iou/test_prob_iou.py   # 若已在 MeMOTR 下
"""
import os
import sys

import torch

# 保证能 import hsmot：从 MeMOTR/debug/test_iou 上溯到仓库根
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hsmot.hsmot.loss.prob_iou_loss import (
    batch_probiou,
    get_covariance_matrix,
    probiou,
    probiou_loss,
)


def _xywhr(cx, cy, w, h, angle_rad):
    return torch.tensor([[cx, cy, w, h, angle_rad]], dtype=torch.float32)


def test_same_box_iou_one():
    """相同框 probiou 应为 1。"""
    box = _xywhr(100.0, 100.0, 50.0, 30.0, 0.5)
    iou = probiou(box, box, CIoU=False)
    assert iou.shape == (1,), iou.shape
    assert torch.allclose(iou, torch.tensor([1.0]), atol=1e-5), f"same box iou={iou}"
    print("  [OK] same box -> probiou = 1")


def test_same_box_batch():
    """多对相同框，probiou 全为 1。"""
    N = 4
    boxes = torch.tensor(
        [
            [10.0, 10.0, 20.0, 15.0, 0.0],
            [50.0, 50.0, 30.0, 25.0, 0.3],
            [100.0, 80.0, 40.0, 20.0, -0.2],
            [200.0, 200.0, 60.0, 40.0, 0.7],
        ],
        dtype=torch.float32,
    )
    iou = probiou(boxes, boxes, CIoU=False)
    assert iou.shape == (N,), iou.shape
    assert torch.allclose(iou, torch.ones(N), atol=1e-5), f"batch same iou={iou}"
    print("  [OK] batch same boxes -> probiou all 1")


def test_disjoint_boxes_low_iou():
    """中心相距很远、不重叠的两框，probiou 应接近 0。"""
    a = _xywhr(0.0, 0.0, 10.0, 10.0, 0.0)
    b = _xywhr(1000.0, 1000.0, 10.0, 10.0, 0.0)
    iou = probiou(a, b, CIoU=False)
    assert iou.shape == (1,), iou.shape
    assert iou.item() < 0.01, f"disjoint iou should be ~0, got {iou.item()}"
    print(f"  [OK] disjoint boxes -> probiou ≈ {iou.item():.6f} (< 0.01)")


def test_partial_overlap():
    """同中心、不同尺寸，应有中等相似度（约 0~1 之间）。"""
    # 同中心，一大一小
    big = _xywhr(50.0, 50.0, 40.0, 40.0, 0.0)
    small = _xywhr(50.0, 50.0, 20.0, 20.0, 0.0)
    iou = probiou(big, small, CIoU=False)
    assert 0 < iou.item() < 1, f"partial overlap iou={iou.item()}"
    print(f"  [OK] same center different size -> probiou = {iou.item():.4f}")


def test_ciou_vs_iou():
    """CIoU 与 IoU 对相同框应一致；对宽高比不同框 CIoU 可能更低。"""
    same = _xywhr(10.0, 10.0, 20.0, 20.0, 0.0)
    iou = probiou(same, same, CIoU=False).item()
    ciou = probiou(same, same, CIoU=True).item()
    assert abs(iou - 1.0) < 1e-5 and abs(ciou - 1.0) < 1e-5
    # 同面积不同宽高比
    a = _xywhr(0.0, 0.0, 20.0, 5.0, 0.0)   # 扁
    b = _xywhr(0.0, 0.0, 10.0, 10.0, 0.0)  # 方
    iou_ab = probiou(a, b, CIoU=False).item()
    ciou_ab = probiou(a, b, CIoU=True).item()
    assert ciou_ab <= iou_ab + 1e-5, "CIoU should penalize aspect ratio"
    print(f"  [OK] CIoU vs IoU: same box both 1; aspect diff -> iou={iou_ab:.4f}, ciou={ciou_ab:.4f}")


def test_batch_probiou_shape():
    """batch_probiou 返回 (N, M)。"""
    N, M = 3, 5
    obb1 = torch.rand(N, 5) * 100
    obb1[:, 2:4] = obb1[:, 2:4].abs() + 1
    obb1[:, 4] = obb1[:, 4] % (3.14159265)
    obb2 = torch.rand(M, 5) * 100
    obb2[:, 2:4] = obb2[:, 2:4].abs() + 1
    obb2[:, 4] = obb2[:, 4] % (3.14159265)
    mat = batch_probiou(obb1, obb2)
    assert mat.shape == (N, M), mat.shape
    print(f"  [OK] batch_probiou shape ({N}, {M}) = {mat.shape}")


def test_batch_probiou_diagonal():
    """同一组框 batch_probiou(boxes, boxes) 对角线应为 1。"""
    boxes = torch.tensor(
        [[10.0, 10.0, 20.0, 15.0, 0.0], [50.0, 50.0, 30.0, 25.0, 0.5]],
        dtype=torch.float32,
    )
    mat = batch_probiou(boxes, boxes)
    diag = torch.diag(mat)
    assert torch.allclose(diag, torch.ones(2), atol=1e-5), f"diag={diag}"
    print("  [OK] batch_probiou(boxes, boxes) diagonal = 1")


def test_probiou_loss_reduction():
    """probiou_loss 的 reduction 与梯度。"""
    pred = torch.tensor([[10.0, 10.0, 20.0, 15.0, 0.0]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[12.0, 11.0, 22.0, 14.0, 0.1]], dtype=torch.float32)
    loss_mean = probiou_loss(pred, target, reduction="mean")
    loss_sum = probiou_loss(pred, target, reduction="sum")
    loss_none = probiou_loss(pred, target, reduction="none")
    assert loss_none.shape == (1,), loss_none.shape
    assert torch.allclose(loss_sum, loss_mean * 1.0)
    loss_mean.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    print(f"  [OK] probiou_loss reduction + grad: mean={loss_mean.item():.4f}, grad norm={pred.grad.norm().item():.4f}")


def test_covariance_matrix_shape():
    """get_covariance_matrix 返回 (a,b,c) 每项 (N,1)。"""
    boxes = torch.rand(4, 5)
    boxes[:, 2:4] = boxes[:, 2:4].abs() + 1
    a, b, c = get_covariance_matrix(boxes)
    assert a.shape == (4, 1) and b.shape == (4, 1) and c.shape == (4, 1)
    print("  [OK] get_covariance_matrix (N,5) -> (N,1) each")


def test_numpy_input_batch_probiou():
    """batch_probiou 接受 numpy 输入并转为 tensor。"""
    import numpy as np
    obb1 = np.array([[10.0, 10.0, 20.0, 15.0, 0.0]], dtype=np.float32)
    obb2 = np.array([[10.0, 10.0, 20.0, 15.0, 0.0]], dtype=np.float32)
    mat = batch_probiou(obb1, obb2)
    assert isinstance(mat, torch.Tensor) and mat.shape == (1, 1)
    assert torch.allclose(mat, torch.tensor([[1.0]], dtype=mat.dtype)), mat
    print("  [OK] batch_probiou with numpy input -> (1,1) = 1")


def run_all():
    print("Prob IoU tests (xywhr)")
    print("-" * 50)
    test_same_box_iou_one()
    test_same_box_batch()
    test_disjoint_boxes_low_iou()
    test_partial_overlap()
    test_ciou_vs_iou()
    test_batch_probiou_shape()
    test_batch_probiou_diagonal()
    test_probiou_loss_reduction()
    test_covariance_matrix_shape()
    test_numpy_input_batch_probiou()
    print("-" * 50)
    print("All tests passed.")


if __name__ == "__main__":
    run_all()
