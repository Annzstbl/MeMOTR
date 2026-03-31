"""
Prob IoU 随 x/y/w/h/theta 变化曲线，及对 x,y,w,h,angle 的梯度曲线，并保存图片。

- 四种 wh：(40, 30), (308.21, 33.12), (200, 50), (400, 200)，theta=0。
- 两种情形：两框完全重叠；两框 h 相差 1（ref 为 h，pred 为 h+1）。
- 每张图：上半部分 Prob IoU 曲线，下半部分 5 条梯度曲线（x/y/w/h/angle 颜色区分）。

运行（在仓库根或 MeMOTR 下）:
  python MeMOTR/debug/test_iou/test_prob_iou_sweep.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hsmot.loss.prob_iou_loss import probiou

# 固定中心，便于画图
CX, CY = 200.0, 200.0

# 四种 (w, h)，theta 统一 0
WH_CASES = [
    (40.0, 30.0),
    (308.218364151133, 33.12346705650146),
    (200.0, 50.0),
    (400.0, 200.0),
]
THETA = [0.3, 0.5, 0.2, 0.9]

#  sweep 范围与点数
SWEEP_N = 81  # 奇数，中心点正好在 nominal
DX_RANGE = 500.0   # x 方向 ±
DY_RANGE = 500.0   # y 方向 ±
DW_RATIO = 0.5    # w 方向 ± w * ratio
DH_RATIO = 0.5    # h 方向 ± h * ratio
DTHETA_RANGE = np.pi / 4  # theta ± rad

OUT_DIR = os.path.join(_CURRENT_DIR, "iou_sweep_results", "prob_iou")

# 5 条梯度曲线颜色：x, y, w, h, angle
GRAD_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
GRAD_LABELS = ["d(IoU)/dx", "d(IoU)/dy", "d(IoU)/dw", "d(IoU)/dh", "d(IoU)/dθ"]


def make_ref(cx, cy, w, h, th, device="cpu"):
    return torch.tensor([[cx, cy, w, h, th]], dtype=torch.float32, device=device)


def compute_iou_and_gradient(ref, pred_val, eps=1e-7):
    """在给定 pred 处计算 iou 与 d(iou)/d(pred)，返回 (iou, grad_5)。"""
    ref = ref.detach()
    pred = pred_val.clone().detach().requires_grad_(True)
    iou = probiou(pred, ref, CIoU=False, eps=eps).squeeze()
    iou.backward()
    g = pred.grad.squeeze().detach().cpu().numpy()
    return iou.item(), g


def sweep_one_axis_with_gradients(ref, pred_nominal, axis, values, eps=1e-7):
    """沿 axis 扫描，每个点返回 iou 和完整梯度 (5,)。返回 ious (N,), grads (N,5)。"""
    ious = []
    grads = []
    for v in values:
        pred = pred_nominal.clone().detach()
        pred[0, axis] = float(v)
        iou, g = compute_iou_and_gradient(ref, pred, eps=eps)
        ious.append(iou)
        grads.append(g)
    return np.array(ious), np.array(grads)


def plot_iou_and_gradients(
    axis_name, values, ious, grads, nominal_value, save_path, title_suffix=""
):
    """上半：Prob IoU vs 横轴；下半：5 条梯度曲线，x/y/w/h/angle 颜色区分。"""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [1, 1]}, sharex=True
    )
    # 上图：IoU
    ax_top.plot(values, ious, "b-", linewidth=2, label="Prob IoU")
    ax_top.axvline(nominal_value, color="gray", linestyle="--", alpha=0.7, label="nominal")
    ax_top.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax_top.set_ylabel("Prob IoU")
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(f"IoU vs {axis_name} {title_suffix}")
    # 下图：5 条梯度
    for k in range(5):
        ax_bot.plot(values, grads[:, k], color=GRAD_COLORS[k], linewidth=1.5, label=GRAD_LABELS[k])
    ax_bot.axvline(nominal_value, color="gray", linestyle="--", alpha=0.7)
    ax_bot.axhline(0, color="gray", linewidth=0.5)
    ax_bot.set_xlabel(axis_name)
    ax_bot.set_ylabel("gradient")
    ax_bot.legend(loc="best", fontsize=7, ncol=2)
    ax_bot.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_case(case_idx, w, h, th, out_subdir):
    ref = make_ref(CX, CY, w, h, th)
    pred_full = make_ref(CX, CY, w, h, th)
    pred_add1 = make_ref(CX, CY, w, h + 1.0, th)

    dw_range = max(w * DW_RATIO, 1.0)
    dh_range = max(h * DH_RATIO, 1.0)
    dth_range = max(th * DTHETA_RANGE, 1.0)

    for mode_name, pred_nominal, file_suffix in [
        ("full overlap", pred_full, ""),
        ("h diff 1", pred_add1, "_add1"),
    ]:
        nominal_x = pred_nominal[0, 0].item()
        nominal_y = pred_nominal[0, 1].item()
        nominal_w = pred_nominal[0, 2].item()
        nominal_h = pred_nominal[0, 3].item()
        nominal_th = pred_nominal[0, 4].item()

        # dx
        values_dx = np.linspace(CX - DX_RANGE, CX + DX_RANGE, SWEEP_N)
        ious_dx, grads_dx = sweep_one_axis_with_gradients(ref, pred_nominal, 0, values_dx)
        plot_iou_and_gradients(
            "x (pred center)",
            values_dx,
            ious_dx,
            grads_dx,
            nominal_x,
            os.path.join(out_subdir, f"iou_vs_dx{file_suffix}.png"),
            title_suffix=f"({mode_name})",
        )
        # dy
        values_dy = np.linspace(CY - DY_RANGE, CY + DY_RANGE, SWEEP_N)
        ious_dy, grads_dy = sweep_one_axis_with_gradients(ref, pred_nominal, 1, values_dy)
        plot_iou_and_gradients(
            "y (pred center)",
            values_dy,
            ious_dy,
            grads_dy,
            nominal_y,
            os.path.join(out_subdir, f"iou_vs_dy{file_suffix}.png"),
            title_suffix=f"({mode_name})",
        )
        # dw
        values_dw = np.linspace(nominal_w - dw_range, nominal_w + dw_range, SWEEP_N)
        ious_dw, grads_dw = sweep_one_axis_with_gradients(ref, pred_nominal, 2, values_dw)
        plot_iou_and_gradients(
            "w (pred width)",
            values_dw,
            ious_dw,
            grads_dw,
            nominal_w,
            os.path.join(out_subdir, f"iou_vs_dw{file_suffix}.png"),
            title_suffix=f"({mode_name})",
        )
        # dh
        values_dh = np.linspace(nominal_h - dh_range, nominal_h + dh_range, SWEEP_N)
        ious_dh, grads_dh = sweep_one_axis_with_gradients(ref, pred_nominal, 3, values_dh)
        plot_iou_and_gradients(
            "h (pred height)",
            values_dh,
            ious_dh,
            grads_dh,
            nominal_h,
            os.path.join(out_subdir, f"iou_vs_dh{file_suffix}.png"),
            title_suffix=f"({mode_name})",
        )
        # dtheta
        values_dt = np.linspace(0, 1, SWEEP_N)
        ious_dt, grads_dt = sweep_one_axis_with_gradients(ref, pred_nominal, 4, values_dt)
        plot_iou_and_gradients(
            "theta (pred rad)",
            values_dt,
            ious_dt,
            grads_dt,
            nominal_th,
            os.path.join(out_subdir, f"iou_vs_dtheta{file_suffix}.png"),
            title_suffix=f"({mode_name})",
        )
    return


def main():
    for case_idx, ((w, h), th) in enumerate(zip(WH_CASES, THETA)):
        # 文件夹名与现有 hsmot 风格接近
        w_str = f"{w:.2f}" if w != int(w) else str(int(w))
        h_str = f"{h:.2f}" if h != int(h) else str(int(h))
        th_str = f"{th:.2f}"
        out_subdir = os.path.join(OUT_DIR, f"case{case_idx}_w{w_str}_h{h_str}_th{th_str}")
        run_case(case_idx, w, h, th, out_subdir)
        print(f"Case {case_idx} (w={w}, h={h}, th={th}) -> {out_subdir}")

    print(f"Done. Results under {OUT_DIR}")


if __name__ == "__main__":
    main()
