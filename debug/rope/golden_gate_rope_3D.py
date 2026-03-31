import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from models.position_embedding_rope_nd import GoldenGateRoPENd

# def _phi(m: int) -> float:
#     x = 2.0
#     for _ in range(10):
#         x = (1 + x) ** (1.0 / (m + 1.0))
#     return x


# # def make_directions(n:int, d:int) -> torch.Tensor:
# #     directions = torch.randn(1, n, d)
# #     directions = directions / directions.norm(dim=1, keepdim=True)
# #     return directions

# # def make_directions(n: int, d: int) -> torch.Tensor:
# #     # return (1, n, d)
# #     directions = torch.ones(1, n, d)
# #     return directions

# def make_directions(n: int, d: int) -> torch.Tensor:
#     g = _phi(d)
#     alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64)
#     i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
#     z = torch.fmod(i * alpha, 1.0)
#     directions = torch.erfinv(2.0 * z - 1.0)
#     directions = directions / directions.norm(dim=1, keepdim=True)
#     return directions.float()


# class GoldenGateRoPENd(nn.Module):
#     def __init__(
#         self,
#         pos_dim: int,
#         n_heads: int,
#         head_dim: int,
#         min_freq: float,
#         max_freq: float,
#         p_zero_freqs: float = 0.0,
#     ):
#         """
#         Args:
#             pos_dim: dimensionality of the token positions
#             n_heads: number of attention heads
#             head_dim: attention head dimensionality
#             min_freq, max_freq: lowest and highest nonzero frequency magnitudes
#             p_zero_freqs: proportion of frequencies set to 0

#         Dimension key:
#             N: batch size
#             L: number of tokens per sample
#             P: pos_dim
#             h: n_heads
#             d: head_dim
#             F: num_freqs == head_dim // 2
#         """
#         super().__init__()
#         n_freqs = head_dim // 2
#         n_zero_freqs = round(p_zero_freqs * n_freqs)
#         omega_F = torch.cat(
#             (
#                 torch.zeros(n_zero_freqs),
#                 min_freq
#                 * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
#             )
#         )

#         directions_hFP = make_directions(n_heads * n_freqs, pos_dim).reshape(
#             n_heads, n_freqs, pos_dim
#         )
#         self.register_buffer("freqs_hFP", directions_hFP * omega_F.reshape(n_freqs, 1))

#     def forward(self, input_NLhd: torch.Tensor, pos_NLP: torch.Tensor) -> torch.Tensor:
#         x_NLhF, y_NLhF = input_NLhd.float().chunk(2, dim=-1)
#         theta_NLhF = (self.freqs_hFP * pos_NLP[..., None, :].float()).sum(dim=-1)
#         cos_NLhF = torch.cos(theta_NLhF)
#         sin_NLhF = torch.sin(theta_NLhF)
#         x_out_NLhF = x_NLhF * cos_NLhF - y_NLhF * sin_NLhF
#         y_out_NLhF = x_NLhF * sin_NLhF + y_NLhF * cos_NLhF
#         output_NLhd = torch.cat((x_out_NLhF, y_out_NLhF), dim=-1)
#         return output_NLhd.type_as(input_NLhd)

def pos_to_index(H, W, lvl, H_0, W_0, lvl_0):
    sum_WH = W*H
    index = H_0*W[lvl_0] + W_0 + sum_WH[:lvl_0].sum()
    return index

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))

    pos_dim = 3
    n_heads = 8
    head_dim = 32
    min_freq = 1
    max_freq = 64
    p_zero_freqs = 0.0
    weights = [1, 1, 0.5]

    rope = GoldenGateRoPENd(pos_dim, n_heads, head_dim, min_freq, max_freq, weights, p_zero_freqs)

    W = torch.tensor([113, 57, 29, 15])
    H = torch.tensor([129, 65, 33, 17])
    lvl = 4

    Bs = 1

    input_NLhd = torch.ones(Bs, (W*H).sum(), n_heads* head_dim)

    # input_NLhd = torch.ones(Bs, W*H*lvl, n_heads, head_dim)
    

    # pos_NLP = torch.stack(torch.meshgrid(torch.arange(H)/H *weights[0] , torch.arange(W)/W *weights[1], torch.arange(lvl)/lvl *weights[2]), dim=-1).reshape(1, -1, pos_dim)

    # pos_NLP = torch.stack(
    #     torch.meshgrid(
    #         torch.linspace(0, 1, H) * weights[0],
    #         torch.linspace(0, 1, W) * weights[1],
    #         torch.linspace(0, 1, lvl) * weights[2],
    #         indexing="ij"   # 显式指定，避免警告
    #     ),
    #     dim=-1
    # ).reshape(1, -1, 1,  pos_dim).repeat(Bs, 1, n_heads, 1)

    pos_rope_list = []

    lvl_rope = torch.linspace(-1, 1, lvl)

    for lvl_, (H_, W_) in enumerate(zip(H, W)):
        pos_y, pos_x = torch.meshgrid(torch.linspace(-1, 1, H_), torch.linspace(-1, 1, W_))
        pos_y = pos_y.reshape(-1)
        pos_x = pos_x.reshape(-1)
        pos_lvl = torch.ones_like(pos_y) * lvl_rope[lvl_]
        pos_rope = torch.stack([pos_x, pos_y, pos_lvl], -1) #(H_ * W_, 3)
        pos_rope_list.append(pos_rope)
    pos_rope = torch.cat(pos_rope_list, 0) #(sum(H_ * W_), 3)
    pos_NLP = pos_rope[None]

    for lvl_ in range(lvl):
        print(pos_NLP[0, pos_to_index(H, W, lvl, H[lvl_]//2, W[lvl_]//2, lvl_)])
    

    # pos_NLP = torch.stack(
    #     torch.meshgrid(
    #         torch.linspace(-1, 1, H) * weights[0],
    #         torch.linspace(-1, 1, W) * weights[1],
    #         torch.linspace(-1, 1, lvl) * weights[2],
    #         indexing="ij"   # 显式指定，避免警告
    #     ),
    #     dim=-1
    # ).reshape(1, -1, 1,  pos_dim).repeat(Bs, 1, n_heads, 1)

    # pos_NLP_debug = pos_NLP.view(H, W, lvl, 3)

    # input_NLhd = torch.randn(1, 1, head_dim)
    # pos_NLP = torch.randn(1, 1, pos_dim)
    output_NLhd = rope.forward_integrate_head(input_NLhd, pos_NLP)
    print(output_NLhd.shape)

    lvl_0 = lvl//2
    W_0 = W[lvl_0]//2
    H_0 = H[lvl_0]//2
    sum_WH = W*H
    index = H_0*W[lvl_0] + H_0 + sum_WH[:lvl_0].sum()


    k_vec = output_NLhd[0, index, :].view(1, n_heads, head_dim)
    scores = (output_NLhd.view(1, -1, n_heads, head_dim) * k_vec).sum(-1) / math.sqrt(head_dim) #[1, sum WH, n_heads]
    scores = F.softmax(scores, dim=-2)
    scores = scores[0]

    for lvl_ in range(lvl):
        print(scores[pos_to_index(H, W, lvl, H[lvl_]//2, W[lvl_]//2, lvl_)])

    plt.figure(figsize=(15,15))
    for i in range(n_heads):
        score_head = scores[:, i]
        score_lvl = []
        max_socre_per_head = score_head.max()
        for lvls in range(lvl):
            start = sum_WH[:lvls].sum()
            end = start + H[lvls]*W[lvls]
            score_lvl.append(score_head[start:end].view(H[lvls], W[lvls]))
            plt.subplot(n_heads, lvl, i*lvl + lvls + 1)
            plt.imshow(score_lvl[lvls], aspect='auto',vmin=0, vmax=max_socre_per_head)
            plt.tight_layout()
            plt.colorbar()
    plt.savefig(os.path.join(file_path, f"attn_weights_rope_3D_golden_gate_weights_{weights}.png"))


    # scores = scores.permute(0,2, 1, 3).reshape(H, W*lvl, n_heads)

    # # scores = scores.permute(0,2,1).reshape(H*n_heads, -1)

    # plt.figure(figsize=(15,15))
    # plt.title("Attention Weights — Two-pass 1D RoPE: (K R_x R_y)(R_y R_x Q)")
    # # plt.xlabel("Key position (y)")
    # # plt.ylabel("Query position (x)")

    # for i in range(n_heads):

    #     plt.subplot(n_heads, 1, i+1)
    #     plt.imshow(scores[:,:,i], aspect='auto',vmin=0, vmax=scores[:,:,i].max())
    #     plt.tight_layout()
    #     plt.colorbar()
    # plt.savefig(os.path.join(file_path, f"attn_weights_rope_3D_golden_gate.png"))


    # # for k in range(lvl):
    # #     plt.figure(figsize=(5,6))
    # #     plt.title("Attention Weights — Two-pass 1D RoPE: (K R_x R_y)(R_y R_x Q)")
    # #     plt.xlabel("Key position (y)")
    # #     plt.ylabel("Query position (x)")
    # #     plt.imshow(scores[:,:,k], aspect='auto',vmin=0, vmax=scores[:,:,k].max())
    # #     plt.tight_layout()
    # #     plt.colorbar()
    # #     plt.savefig(os.path.join(file_path, f"attn_weights_rope_3D_golden_gate_k_{k}.png"))
