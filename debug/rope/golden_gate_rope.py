import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

def _phi(m: int) -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x


# def make_directions(n: int, d: int) -> torch.Tensor:
#     # return (1, n, d)
#     directions = torch.ones(1, n, d)
#     return directions

def make_directions(n: int, d: int) -> torch.Tensor:
    g = _phi(d)
    alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64)
    i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    directions = torch.erfinv(2.0 * z - 1.0)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return directions.float()


class GoldenGateRoPENd(nn.Module):
    def __init__(
        self,
        pos_dim: int,
        n_heads: int,
        head_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float = 0.0,
    ):
        """
        Args:
            pos_dim: dimensionality of the token positions
            n_heads: number of attention heads
            head_dim: attention head dimensionality
            min_freq, max_freq: lowest and highest nonzero frequency magnitudes
            p_zero_freqs: proportion of frequencies set to 0

        Dimension key:
            N: batch size
            L: number of tokens per sample
            P: pos_dim
            h: n_heads
            d: head_dim
            F: num_freqs == head_dim // 2
        """
        super().__init__()
        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)
        omega_F = torch.cat(
            (
                torch.zeros(n_zero_freqs),
                min_freq
                * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
            )
        )

        directions_hFP = make_directions(n_heads * n_freqs, pos_dim).reshape(
            n_heads, n_freqs, pos_dim
        )
        self.register_buffer("freqs_hFP", directions_hFP * omega_F.reshape(n_freqs, 1))

    def forward(self, input_NLhd: torch.Tensor, pos_NLP: torch.Tensor) -> torch.Tensor:
        x_NLhF, y_NLhF = input_NLhd.float().chunk(2, dim=-1)
        theta_NLhF = (self.freqs_hFP * pos_NLP[..., None, :].float()).sum(dim=-1)
        cos_NLhF = torch.cos(theta_NLhF)
        sin_NLhF = torch.sin(theta_NLhF)
        x_out_NLhF = x_NLhF * cos_NLhF - y_NLhF * sin_NLhF
        y_out_NLhF = x_NLhF * sin_NLhF + y_NLhF * cos_NLhF
        output_NLhd = torch.cat((x_out_NLhF, y_out_NLhF), dim=-1)
        return output_NLhd.type_as(input_NLhd)



class GoldenGateRoPE2d(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        n_heads: int,
        head_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float = 0.0,
        weights: tuple[float, float] = (1.0, 1.0),
        # direction_spacing: float = math.pi * (math.sqrt(5) - 1) / 2,
        direction_spacing: float = 2 * math.pi * (math.sqrt(5) - 1) / 2,
        # direction_spacing: float = math.pi/2,
    ):
        """
        Args:
            image_size: expected height and width of (patchified) input
            n_heads: number of attention heads
            head_dim: attention head dimensionality
            min_freq, max_freq: lowest and highest nonzero frequency magnitudes
            p_zero_freqs: proportion of frequencies set to 0
            direction_spacing: difference in radians between adjacent directions along
                which position is measured
        
        Dimension key:
            N: batch size
            H: image_size[0]
            W: image_size[1]
            h: n_heads
            d: head_dim
            F: num_freqs == d // 2
        """
        super().__init__()
        assert head_dim % 2 == 0
        assert 0 <= p_zero_freqs <= 1
        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)
        omega_F = torch.cat(
            (
                torch.zeros(n_zero_freqs),
                min_freq
                * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
            )
        )
        phi_hF = (
            torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs)
            * direction_spacing
        )
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2

        H, W = image_size
        H_weights, W_weights = weights
        # xlim, ylim = math.sqrt(W / H) * H_weights, math.sqrt(H / W) * W_weights
        # xlim, ylim = W / H * H_weights, H / W * W_weights
        xlim = 1 * W_weights
        ylim = 1 * H_weights
        x_HW = torch.linspace(-xlim, xlim, W).reshape(1, W).expand(H, W)
        y_HW = torch.linspace(-ylim, ylim, H).reshape(H, 1).expand(H, W)
        positions_HW112 = torch.stack((x_HW, y_HW), dim=-1).reshape(H, W, 1, 1, 2)

        theta_HWhF = (freqs_hF2 * positions_HW112).sum(dim=-1)
        self.register_buffer("cos_HWhF", torch.cos(theta_HWhF))
        self.register_buffer("sin_HWhF", torch.sin(theta_HWhF))

    def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
        x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)
        x_out_NHWhF = x_NHWhF * self.cos_HWhF - y_NHWhF * self.sin_HWhF
        y_out_NHWhF = x_NHWhF * self.sin_HWhF + y_NHWhF * self.cos_HWhF
        output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
        return output_NHWhd.type_as(input_NHWhd)

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))

    pos_dim = 2
    n_heads = 1
    head_dim = 32
    min_freq = 1
    max_freq = 64
    p_zero_freqs = 0.0

    W = 136
    H = 34

    weights = [0.25, 1]


    rope = GoldenGateRoPE2d((H,W), n_heads, head_dim, min_freq, max_freq, p_zero_freqs, weights)


    input = torch.ones(H, W, 1, head_dim)
    # input_NLhd = torch.ones(1, W*H, head_dim)
    output_NLhd = rope(input)
    # pos_NLP = torch.stack(torch.meshgrid(torch.arange(W)/(W-1), torch.arange(H)/(H-1)), dim=-1).reshape(1, -1, 2)



    # input_NLhd = torch.randn(1, 1, head_dim)
    # pos_NLP = torch.randn(1, 1, pos_dim)
    # output_NLhd = rope(input_NLhd, pos_NLP)
    print(output_NLhd.shape)
    output_NLhd = output_NLhd.squeeze()

    i0, j0 = H // 2, W // 2
    k_vec = output_NLhd[i0, j0, :]
    scores = (output_NLhd * k_vec).sum(-1) / math.sqrt(head_dim)
    # scores = scores.view(W, H)
    scores = F.softmax(scores.flatten(), dim=0)
    scores = scores.view(H,W)

    max_row, max_col = divmod(scores.argmax().item(), W)


    # plt.figure(figsize=(6,5))
    # plt.title(f"max position of scores: {max_row}, {max_col}")
    # plt.xlabel("Key position (y)")
    # plt.ylabel("Query position (x)")
    # plt.imshow(scores, aspect='auto')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(os.path.join(file_path, "attn_weights_rope_2D_golden_gate.png"))


    col_idx_per_row = torch.argmax(scores, dim=1) # 每行最大值的列号 [H]
    max_vals_per_row = scores[torch.arange(H), col_idx_per_row]  # 每行最大值 [H]

    row_ids = torch.arange(H)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(scores.cpu().numpy())  # 热力图

    # # 在每一行最大值位置打一个标记（×）
    # ax.scatter(col_idx_per_row.cpu().numpy(),
    #         row_ids.cpu().numpy(),
    #         marker='x', s=30, c='red')

    # 在每一行右侧写上“列号: 值(科学计数)”
    for r, (c, v) in enumerate(zip(col_idx_per_row.cpu().tolist(),
                                max_vals_per_row.cpu().tolist())):
        ax.text(W + 0.5, r, f"{c}: {v:.2e}", va='center', fontsize=8)

    # 预留右侧空白，以显示文字
    ax.set_xlim(-0.5, W + 15)

    ax.set_title("Per-row max col & value")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "attn_weights_rope_2D_golden_gate.png"))