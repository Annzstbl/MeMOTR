# -*- coding: utf-8 -*-
# Two-pass 1D RoPE along x and y axes over the SAME complex pairs:
# (K R_x R_y) (R_y R_x Q). This matches user's requested form.
#
# Notes:
# - We keep d_head even. No splitting into x/y halves. We apply
#   1D RoPE once along x positions, then again along y positions,
#   on the SAME complex pairs (so rotations add).
# - This demo visualizes attention to a single key after the two-pass RoPE.
#
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Config ----------------
W, H = 96, 96
d_head = 32           # 必须为偶数
base = 10000.0
torch.manual_seed(0)

# 路径
file_path = os.path.dirname(os.path.abspath(__file__))


# ---------------- Helpers ----------------
def inv_freq_1d(d: int, base: float = 10000.0):
    """与 1D RoPE 一致的频率刻度：pairs = d//2"""
    assert d % 2 == 0
    freqs = 1.0 / (base ** (torch.arange(0, d, 2)[: (d // 2)].float() / d))
    return freqs

def rope_1d_apply_along_axis(T: torch.Tensor, positions: torch.Tensor, base: float = 10000.0):
    """
    在最后一维 d 上按 1D RoPE 旋转（成对为一个复平面），旋转角度由 positions 给出。
    T: [W, H, d]
    positions: 形状可广播到 [W, H]（比如 [W,1] 或 [1,H] 或 [W,H]）
    返回: 与 T 同形
    """
    W, H, d = T.shape
    pairs = d // 2
    v = T.reshape(W, H, pairs, 2).contiguous()
    vc = torch.view_as_complex(v)   # [W,H,pairs]
    invf = inv_freq_1d(d, base=base).to(T.device).to(T.dtype)   # [pairs]

    theta = positions.unsqueeze(-1) * invf                      # [W,H,pairs]
    cis = torch.polar(torch.ones_like(theta), theta)            # cis(theta)
    outc = vc * cis
    out = torch.view_as_real(outc).reshape(W, H, d)
    return out

def apply_rope_2d_by_two_1d(Q: torch.Tensor, K: torch.Tensor, base: float = 10000.0):
    """
    通过两次一维 RoPE 实现二维：
        K' = (((K) R_x) R_y)
        Q' = (((Q) R_y) R_x)
    即 (K R_x R_y)(R_y R_x Q)，与用户所述形式一致。
    """
    W, H, d = Q.shape
    # x 轴位置（宽度方向）：shape [W,1] 广播到 [W,H]
    pos_x = torch.arange(W, dtype=Q.dtype, device=Q.device).view(W, 1).expand(W, H)
    # y 轴位置（高度方向）：shape [1,H] 广播到 [W,H]
    pos_y = torch.arange(H, dtype=Q.dtype, device=Q.device).view(1, H).expand(W, H)

    # K: 先按 x 旋转，再按 y 旋转
    # Kx = rope_1d_apply_along_axis(K, pos_x, base=base)
    # Ky = rope_1d_apply_along_axis(Kx, pos_y, base=base)

    Ky = rope_1d_apply_along_axis(K, pos_y, base=base)
    Kx = rope_1d_apply_along_axis(Ky, pos_x, base=base)


    # Q: 先按 y 旋转，再按 x 旋转（顺序可交换，但按用户要求写）
    Qy = rope_1d_apply_along_axis(Q, pos_y, base=base)
    Qx = rope_1d_apply_along_axis(Qy, pos_x, base=base)

    # return Qx, Ky
    return Qx, Kx

# ---------------- Demo 数据 ----------------
x = torch.ones((W, H, d_head))     # toy token features
# Wq = torch.randn(d_head, d_head)
# Wk = torch.randn(d_head, d_head)
# Q  = x @ Wq
# K  = x @ Wk
Q = x
K = x

# 二维位置编码（两次一维）
Q_rope, K_rope = apply_rope_2d_by_two_1d(Q, K, base=base)

# 取一个 key 位置，观察全图对它的注意力
i0, j0 = W // 3, H // 2
k_vec = K_rope[i0, j0, :]                      # [d]
scores = (Q_rope * k_vec).sum(-1) / math.sqrt(d_head)   # [W,H]
scores = F.softmax(scores.flatten(), dim=0).view(W, H)

# ---------------- Plot ----------------
plt.figure(figsize=(6,5))
plt.title("Attention Weights — Two-pass 1D RoPE: (K R_x R_y)(R_y R_x Q)")
plt.xlabel("Key position (y)")
plt.ylabel("Query position (x)")
plt.imshow(scores, aspect='auto')
plt.colorbar()
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(file_path, "attn_weights_rope_2D_fullFrequency.png"))

