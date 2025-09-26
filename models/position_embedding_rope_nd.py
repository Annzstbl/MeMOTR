# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import torch
import math
from torch import nn

from utils.nested_tensor import NestedTensor
from torch import Tensor
from typing import List

def _phi(m: int) -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x

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
        rope_weights: List[float],
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
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.pos_dim = pos_dim
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.p_zero_freqs = p_zero_freqs
        self.rope_weights = rope_weights
        
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

        
    def forward(self, input: Tensor, pos: Tensor) -> Tensor:

        '''
        Args:
            input: (B, L, h, d)   batch_size, length, heads, dimension
            pos: (B, L, h, n)   batch_size, length, heads, number of positions
        Returns:
            (B, L, h, d)
        '''

        x_NLhF, y_NLhF = input.float().chunk(2, dim=-1)

        rope_weights = torch.tensor(self.rope_weights, device=input.device).reshape(1, 1, 1, -1)
        pos = pos * rope_weights

        theta_NLhF = (self.freqs_hFP * pos[..., None, :].float()).sum(dim=-1)# (B, L, h, f)
        cos_NLhF = torch.cos(theta_NLhF)
        sin_NLhF = torch.sin(theta_NLhF)
        x_out_NLhF = x_NLhF * cos_NLhF - y_NLhF * sin_NLhF
        y_out_NLhF = x_NLhF * sin_NLhF + y_NLhF * cos_NLhF
        output_NLhd = torch.cat((x_out_NLhF, y_out_NLhF), dim=-1)
        return output_NLhd.type_as(input)

    def forward_integrate_head(self, input: Tensor, pos: Tensor) -> Tensor:
        '''
        Args:
            对于所有head整合到一起的情况
            input: (B, L, D)   batch_size, length, heads, dimension
            pos: (B, L, n)   batch_size, length, heads, number of positions
        Returns:
            (B, L, D)
        '''
        input = input.view(*input.shape[:-1], self.n_heads, -1)
        pos = pos.unsqueeze(-2).repeat(1, 1, self.n_heads, 1)
        output = self.forward(input, pos)
        return output.view(*output.shape[:-2], -1)



def build(config: dict):
    hidden_num = config["HIDDEN_DIM"]
    num_heads = config["NUM_HEADS"]
    head_dim = hidden_num // num_heads
    assert head_dim % 2 == 0, f"Head dim should be 2x, but get {head_dim}."

    min_freq = config["ROPE_MIN_FREQ"]
    max_freq = config["ROPE_MAX_FREQ"]
    p_zero_freqs = config["ROPE_P_ZERO_FREQS"]
    pos_Dim = config["ROPE_DIMENSION"]

    rope_weights = config["ROPE_WEIGHTS"]

    return GoldenGateRoPENd(pos_dim=pos_Dim, n_heads=num_heads, head_dim=head_dim, min_freq=min_freq, max_freq=max_freq, p_zero_freqs=p_zero_freqs, rope_weights=rope_weights)
