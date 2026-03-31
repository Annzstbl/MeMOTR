# ROPE demo for language-style 1D sequence, plus attention weight visualization
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from typing import Tuple
import math
import torch.nn.functional as F

file_path = os.path.dirname(os.path.abspath(__file__))




def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == "__main__":
    # ----- Configs -----
    L = 128            # sequence length
    d_head = 64       # per-head dimension (even)
    scale = 1.0/np.sqrt(d_head)  # attention scaling

    # ----- Make random token embeddings and project to Q/K -----
    x = torch.ones((L, d_head))     # toy token features


    # set seed
    torch.manual_seed(0)


    Q = x
    K = x

    freqs_cis = precompute_freqs_cis(d_head, L)
    Q_rope, K_rope = apply_rotary_emb(Q, K, freqs_cis)
    Q_rope = Q_rope.flatten(1)
    K_rope = K_rope.flatten(1)

    print(Q_rope.shape, K_rope.shape)
    scores = torch.matmul(Q_rope, K_rope.transpose(0, 1)) / math.sqrt(d_head)
    scores = F.softmax(scores.float(), dim=-1)

    # ----- Plot: Two separate figures so each chart stands alone -----

    plt.figure(figsize=(6, 5))
    plt.title("Attention Weights (With RoPE)")
    plt.xlabel("Key position (j)")
    plt.ylabel("Query position (i)")
    plt.imshow(scores, aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(file_path, "attn_weights_rope.png"))

