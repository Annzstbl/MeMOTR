import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, StudentT
from models.GDN import GDN


# =========================
# Utils
# =========================

def _log_sigma_from_raw(raw_log_sigma: torch.Tensor, eps: float = 1e-4):
    return torch.log(F.softplus(raw_log_sigma) + eps)

def GN(ch, num_groups=32):
    return nn.GroupNorm(min(num_groups, ch), ch)

def _zero_invalid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x:    [B,C,H,W]
    mask: [B,H,W], True=padding or [B,1,H,W], True=padding
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B,1,H,W]
    elif mask.dim() == 4 and mask.size(1) == 1:
        mask = mask
    else:
        raise ValueError(f"mask shape should be [B,H,W] or [B,1,H,W], got {mask.shape}")
    valid = (~mask).to(dtype=x.dtype)  # [B,1,H,W]
    return x * valid

# =========================
# CoordConv with cache & dynamic size
# =========================

class CoordConv(nn.Module):
    """
    CoordConv with dynamic resolution (no cache).
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_cache=True, cache_size=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + 2, out_ch, k, s, p, bias=False)
        self.bn   = GN(out_ch)
        self.act  = nn.SiLU(inplace=False)

    def forward(self, x, pad_mask: torch.Tensor = None):
        """
        x: [B, C, H, W]
        pad_mask: [B,H,W] or [B,1,H,W], True 表示 padding 区域。

        要求:
          - 只在有效区域 (pad_mask=False) 上把坐标归一化到 [-1, 1]
          - padding 区域 (pad_mask=True) 的坐标落在边界上（右/下边界 → 1)
        """
        B, _, H, W = x.shape

        # 没有 pad_mask 时，退化为原始的全局 [-1,1] 网格
        # 统一成 [B,1,H,W]，True = padding
        if pad_mask.dim() == 3:
            pad = pad_mask.unsqueeze(1)   # [B,1,H,W]
        elif pad_mask.dim() == 4 and pad_mask.size(1) == 1:
            pad = pad_mask
        else:
            raise ValueError(f"pad_mask shape should be [B,H,W] or [B,1,H,W], got {pad_mask.shape}")

        valid = ~pad  # [B,1,H,W], True = 有效像素

        # 每个样本的有效高/宽（从左上开始，右/下为 padding）
        row_has_valid = valid.any(dim=3).squeeze(1)   # [B,H]
        col_has_valid = valid.any(dim=2).squeeze(1)   # [B,W]

        h_valid = row_has_valid.sum(dim=1).view(B, 1, 1).clamp(min=1)  # [B,1,1]
        w_valid = col_has_valid.sum(dim=1).view(B, 1, 1).clamp(min=1)  # [B,1,1]

        # 行/列索引网格（不含 batch 维）
        yy = torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1).expand(B, H, W)
        xx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, W).expand(B, H, W)

        # 对超出有效高/宽的行/列做 clamp，使 padding 行/列都贴在边界上
        yy_max = (h_valid - 1).clamp(min=0)  # [B,1,1]
        xx_max = (w_valid - 1).clamp(min=0)  # [B,1,1]
        yy = torch.minimum(yy, yy_max)
        xx = torch.minimum(xx, xx_max)

        # 将 [0, h_valid-1] / [0, w_valid-1] 线性映射到 [-1,1]
        den_h = (h_valid - 1).clamp(min=1)  # 避免除 0
        den_w = (w_valid - 1).clamp(min=1)
        yy = (yy / den_h) * 2.0 - 1.0
        xx = (xx / den_w) * 2.0 - 1.0

        cc = torch.stack([xx, yy], dim=1)  # [B,2,H,W]

        x = torch.cat([x, cc], dim=1)
        return self.act(self.bn(self.conv(x)))


# =========================
# Center-masked DW Conv + ASPP + PIHead
# =========================

class CenterMaskedConv3x3(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=False)
        mask = torch.ones(3, 3)
        mask[1, 1] = 0.0
        self.register_buffer("mask", mask.view(1, 1, 3, 3))
        self.bn  = GN(ch)
        self.act = nn.SiLU(inplace=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        self.conv.weight.data = self.conv.weight.data * self.mask
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 2, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, dilation=r, padding=r, bias=False),
                GN(out_ch),
                nn.SiLU(inplace=False)
            ) for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, 1, 1, 0, bias=False),
            GN(out_ch),
            nn.SiLU(inplace=False)
        )

    def forward(self, x):
        ys = [m(x) for m in self.branches]
        return self.proj(torch.cat(ys, dim=1))


class SpectralManifold(nn.Module):
    """
    Spectral Manifold Learning Module

    输入:
        spec: [B, C, H, W]   原始光谱

    输出:
        embed: [B, C+1, H, W]

        前 C 个通道:
            spectral reconstruction embedding

        最后 1 个通道:
            reconstruction error (spectral deviation)
    """

    def __init__(self,
                 spectral_channels: int = 8,
                 num_prototypes: int = 64,
                 hidden_ratio: float = 2.0,
                 eps: float = 1e-6):

        super().__init__()

        C = spectral_channels
        K = num_prototypes
        hidden = int(C * hidden_ratio)

        self.C = C
        self.K = K
        self.eps = eps

        # ---------- spectral encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(C, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, C, 1, bias=False)
        )

        # ---------- spectral prototypes ----------
        self.prototypes = nn.Parameter(torch.randn(K, C))
        nn.init.normal_(self.prototypes, std=0.02)

    def forward(self, spec):

        B, C, H, W = spec.shape
        assert C == self.C

        # -------------------------------------------------
        # 1. spectral encoder
        # -------------------------------------------------

        s_enc = self.encoder(spec)        # [B,C,H,W]

        # -------------------------------------------------
        # 2. spectral shape normalization
        # -------------------------------------------------

        s_norm = s_enc - s_enc.mean(dim=1, keepdim=True)
        s_norm = F.normalize(s_norm, dim=1)

        # -------------------------------------------------
        # 3. normalized prototypes (for similarity)
        # -------------------------------------------------

        proto = self.prototypes                        # raw prototypes
        proto_norm = proto - proto.mean(dim=1, keepdim=True)
        proto_norm = F.normalize(proto_norm, dim=1)

        # -------------------------------------------------
        # 4. cosine similarity
        # -------------------------------------------------

        sim = F.conv2d(
            s_norm,
            proto_norm.view(self.K, self.C, 1, 1)
        )  # [B,K,H,W]

        # -------------------------------------------------
        # 5. soft spectral assignment
        # -------------------------------------------------

        weight = F.softmax(sim, dim=1)  # [B,K,H,W]

        # -------------------------------------------------
        # 6. spectral reconstruction (raw spectral space)
        # -------------------------------------------------

        recon = torch.einsum(
            "bkhw,kc->bchw",
            weight,
            proto
        )  # [B,C,H,W]

        # -------------------------------------------------
        # 7. reconstruction error (original spectral space)
        # -------------------------------------------------

        error = (spec - recon).pow(2).sum(dim=1, keepdim=True)

        # -------------------------------------------------
        # 8. spectral embedding
        # -------------------------------------------------

        embed = torch.cat([recon, error], dim=1)

        return embed


class SpectralPi(nn.Module):
    
    def __init__(self, spectral_database_num=64, in_ch=8, eps=1e-6):
        super().__init__()
        self.K = spectral_database_num
        self.C = in_ch
        self.eps = eps
        # 学习的光谱库 logits，先过 sigmoid 再用
        self.spectral_db_logits = nn.Parameter(
            torch.randn(self.K, self.C)
        )

    def forward(self, spec):
        """
        spec: [B, C, H, W]，假定已经在 [0,1] 区间
        return: prior_sim [B, K, H, W]  余弦相似度先验图
        """
        B, C, H, W = spec.shape
        assert C == self.C, f"SpectralPi expects {self.C} channels, got {C}"

        # ----- 像素光谱：减自身均值再 L2 归一化 -----
        # s = spec.clamp(0.0, 1.0)                          # [B,C,H,W]
        s = spec                        # [B,C,H,W]
        mu_s = s.mean(dim=1, keepdim=True)                # [B,1,H,W]
        s_zero = s - mu_s                                 # 去均值
        s_norm = s_zero / (s_zero.norm(dim=1, keepdim=True) + self.eps)  # [B,C,H,W]

        # ----- 光谱库：sigmoid 后也减均值+归一 -----
        # db = torch.sigmoid(self.spectral_db_logits)       # [K,C] in (0,1)
        db = self.spectral_db_logits
        mu_db = db.mean(dim=1, keepdim=True)              # [K,1]
        db_zero = db - mu_db
        db_norm = db_zero / (db_zero.norm(dim=1, keepdim=True) + self.eps)  # [K,C]

        # ----- 1x1 conv 实现余弦相似度 -----
        weight = db_norm.view(self.K, self.C, 1, 1)       # [K,C,1,1]
        prior_sim = F.conv2d(s_norm, weight=weight, bias=None)  # [B,K,H,W]

        # 现在 prior_sim 真正可以覆盖 [-1,1]，差异会大很多
        return prior_sim


class SpectralGraphDictionaryPrior(nn.Module):
    """
    Spectral Graph Dictionary Prior

    输入:
        spec: [B, C, H, W]   光谱 (0~1)

    输出:
        pi_spec: [B,1,H,W]   spectral foreground probability
    """

    def __init__(self,
                 spectral_channels=8,
                 num_prototypes=64,
                 tau=1.0):

        super().__init__()

        C = spectral_channels
        K = num_prototypes

        self.C = C
        self.K = K
        self.tau = tau

        # ----- spectral prototypes -----
        self.prototypes = nn.Parameter(
            torch.randn(K, C)
        )

        nn.init.normal_(self.prototypes, std=0.02)

        # ----- prototype objectness -----
        self.proto_obj = nn.Parameter(
            torch.full((K,), 0.0)   # 初始 objectness 
        )

        # self.proto_obj.requires_grad = False

    # ---------------------------------------------------------
    # prototype diversity regularization
    # ---------------------------------------------------------
    def diversity_loss(self):

        P = F.normalize(self.prototypes, dim=1)

        gram = torch.matmul(P, P.t())

        I = torch.eye(self.K, device=P.device)

        return ((gram - I) ** 2).mean()

    # ---------------------------------------------------------
    # forward
    # ---------------------------------------------------------
    def forward(self, spec):

        B, C, H, W = spec.shape
        K = self.K

        # -----------------------------------------------------
        # 1 spectral shape normalization
        # -----------------------------------------------------

        s = spec - spec.mean(dim=1, keepdim=True)

        s = F.normalize(s, dim=1)

        # -----------------------------------------------------
        # 2 prototype normalization
        # -----------------------------------------------------

        proto = self.prototypes

        proto_norm = proto - proto.mean(dim=1, keepdim=True)

        proto_norm = F.normalize(proto_norm, dim=1)

        # -----------------------------------------------------
        # 3 similarity
        # -----------------------------------------------------

        sim = F.conv2d(
            s,
            proto_norm.view(K, C, 1, 1)
        )                          # [B,K,H,W]

        # -----------------------------------------------------
        # 4 spectral coordinate
        # -----------------------------------------------------

        w = F.softmax(sim / self.tau, dim=1)

        # -----------------------------------------------------
        # 5 prototype graph reasoning
        # -----------------------------------------------------

        A = torch.matmul(proto_norm, proto_norm.t())

        A = F.softmax(A, dim=1)

        w_flat = w.view(B, K, -1)

        w_graph = torch.matmul(A, w_flat)

        w_graph = w_graph.view(B, K, H, W)

        # -----------------------------------------------------
        # 6 similarity-conditioned objectness
        # -----------------------------------------------------

        obj = self.proto_obj.view(1, K, 1, 1)

        # score = obj
        # score = torch.sigmoid(obj * sim)
        score = torch.sigmoid(obj)

        # -----------------------------------------------------
        # 7 dictionary voting
        # -----------------------------------------------------

        # pi_spec = torch.sum(
        #     w_graph * score,
        #     dim=1,
        #     keepdim=True
        # )
        pi_spec = w_graph * score

        return pi_spec


class PIHead(nn.Module):
    def __init__(self, in_ch, ch=128, use_cache=True, spectral_type=None, spectral_databse_num=64):
        super().__init__()
        self.coord = CoordConv(in_ch, ch // 2, use_cache=use_cache)  # +2 coords
        self.mask1 = CenterMaskedConv3x3(ch // 2)
        self.aspp  = ASPP(ch // 2, ch // 2, rates=(1, 2, 3, 5))
        self.local_fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            GN(ch),
            nn.SiLU(inplace=False)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc   = nn.Sequential(
            nn.Conv2d(ch // 2, ch // 2, 1, 1, 0, bias=False),
            nn.SiLU(inplace=False)
        )
        self.out = nn.Conv2d(ch, 1, 1, 1, 0)

        # 光谱相关

        self.spectral_type = spectral_type.lower()
        if self.spectral_type == "pi":
            self.spec_proj = nn.Sequential(
                nn.Conv2d(spectral_databse_num, ch, 1, 1, 0, bias=False),
                GN(ch),
                nn.SiLU(inplace=False)
            )
            self.spec_pi = SpectralPi(spectral_databse_num)
        elif self.spectral_type == "manifold":
            self.spec_proj = nn.Sequential(
                nn.Conv2d(8+1, ch, 1, 1, 0, bias=False),
                GN(ch),
                nn.SiLU(inplace=False)
            )
            self.spec_manifold = SpectralManifold(num_prototypes=spectral_databse_num)
        elif self.spectral_type == "graph_dictionary":
            self.spec_proj = nn.Sequential(
                nn.Conv2d(spectral_databse_num, ch, 1, 1, 0, bias=False),
                GN(ch),
                nn.SiLU(inplace=False)
            )
            self.spec_graph_dictionary = SpectralGraphDictionaryPrior(num_prototypes=spectral_databse_num)
        else:
            raise ValueError(f"Invalid spectral method: {self.spectral_type}")

    def forward(self, x, spec, pad_mask):
        # 根据空间信息做出的判断；将 padding mask 传入 CoordConv，使其只在有效区域内使用 [-1,1] 坐标
        a = self.coord(x, pad_mask=pad_mask)
        a = self.mask1(a)
        a = self.aspp(a)

        a = _zero_invalid(a, pad_mask)

        g = self.global_fc(self.global_pool(a))
        g = g.expand_as(a).contiguous()
        f_space = torch.cat([a, g], dim=1)

        
        if self.spectral_type == "pi":
            f_spec = self.spec_proj(self.spec_pi(spec))
        elif self.spectral_type == "manifold":
            f_spec = self.spec_proj(self.spec_manifold(spec))
        elif self.spectral_type == "graph_dictionary":
            f_spec = self.spec_proj(self.spec_graph_dictionary(spec))
        else:
            raise ValueError(f"Invalid spectral method: {self.spectral_type}")

        h = self.local_fuse(f_space + f_spec)
        pi = torch.sigmoid(self.out(h))
        return pi

# =========================
# Background head
# =========================
class BGHead(nn.Module):
    def __init__(self, in_ch, out_ch, interval_width: float = 1.0):
        super().__init__()
        self.interval_width = interval_width
        self.mu_b_head = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.logsig_b_head = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)

        with torch.no_grad():
            nn.init.zeros_(self.logsig_b_head.weight)
            self.logsig_b_head.bias.fill_(1.313)


    def forward(self, x, Z):
        mu_b = self.mu_b_head(x)
        log_sigb = self.logsig_b_head(x)
        log_pb = self.log_gaussian_interval(
            Z, mu_b, log_sigb, width=self.interval_width
        )   # [B,1,H,W] <= 0
        return log_pb

    def log_gaussian_interval(self, z, mu, raw_log_sigma, width: float = 1.0, eps: float = 1e-8):
        """
        对每个通道建模为独立高斯，返回 log P(z ∈ [z-Δ/2, z+Δ/2]) 的和：
        log P = sum_c log ∫_{z_c-Δ/2}^{z_c+Δ/2} N(u; μ_c, σ_c^2) du
        这里 Δ=width, 默认 1.0 → [x-0.5, x+0.5]
        输入:
        z, mu, raw_log_sigma: [B,C,H,W]
        输出:
        log_prob: [B,1,H,W]  (通道和)
        """
        log_sigma = _log_sigma_from_raw(raw_log_sigma)   # [B,C,H,W]
        sigma = torch.exp(log_sigma)

        half = width * 0.5
        dist = Normal(loc=mu, scale=sigma)

        upper = dist.cdf(z + half)
        lower = dist.cdf(z - half)
        p = (upper - lower).clamp_min(eps)   # [B,C,H,W], 每通道区间概率 ∈ (0,1]

        log_p = torch.log(p)
        # 按通道相加得到 joint 概率的 log
        return log_p.sum(dim=1, keepdim=True)   # [B,1,H,W]

# =========================
# Mixture posterior head
# =========================

class MixBGFG(nn.Module):
    def __init__(self, C, depth=4, width=0.5, with_foreground=True, use_cache=True, tau_mode="mean", prior_mode="gate", spectral_type="pi", spectral_databse_num=128):
        super().__init__()
        ch = int(C * width)
        self.tau_mode  = tau_mode  # "mean" or "sqrt"
        self.tau = 1.0 / ch if tau_mode == "mean" else 1.0 / math.sqrt(ch)

        self.stem  = nn.Sequential(nn.Conv2d(C, ch, 3, 1, 1, bias=False),
                                   GDN(ch),
                                   nn.SiLU(),
                                   nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                                   GDN(ch),
                                   nn.SiLU())

        #背景概率建模
        self.bg_head = BGHead(in_ch=ch, out_ch=C)

        # 先验建模
        self.pi_head = PIHead(in_ch=ch, use_cache=use_cache, spectral_type=spectral_type, spectral_databse_num=spectral_databse_num)

        self.prior_mode = prior_mode.lower()
        assert self.prior_mode == "gate"
        self.gate_head = nn.Sequential(nn.Conv2d(ch+1, ch//2, 3,1,1), GN(ch//2), nn.SiLU(),
                            nn.Conv2d(ch//2, 1, 1,1,0))

    def forward(self, Z, pad_mask, interval_width: float = 1.0, prior_map = None, spec=None):
        """
        Z: [B,C,H,W]
        pad_mask: [B,H,W] or [B,1,H,W], True = padding
        interval_width: 区间 Δ，默认为 1.0 → [x-0.5, x+0.5]
        """

        if pad_mask.dim() == 3:
            pad_mask = pad_mask.unsqueeze(1)  # [B,1,H,W]
            #? 是否有必要归零？
            #特征在mask的地方归零
            # Z = Z * pad_mask

        # x = self.trunk(self.stem(Z))
        x = self.stem(Z)

        # 背景区间 log prob
        log_pb = self.tau * self.bg_head(x, Z)


        # π 先验：将 pad_mask 传入 PIHead / CoordConv，使坐标只在有效区域内为 [-1,1]
        pi_net = self.pi_head(x, spec, pad_mask).clamp(1e-6, 1 - 1e-6)   # [B,1,H,W]

        # pi先验融合时序先验
        logit_net = torch.logit(pi_net)
        prior_map = torch.clamp(prior_map, 1e-6, 1 - 1e-6)
        logit_prior = torch.logit(prior_map.unsqueeze(1))

        logit_prior_downscale = F.interpolate(logit_prior, size=pi_net.shape[-2:], mode="bilinear", align_corners=False)
        g = torch.sigmoid(self.gate_head(torch.cat([x, logit_prior_downscale], dim=1)))
        l = (1-g) * logit_net + g * logit_prior_downscale
        pi = torch.sigmoid(l)


        # 混合区间概率的 loglik
        a = torch.log1p(-pi) + log_pb        # log((1-π)·P_b)
        b = torch.log(pi)                    # log(π·P_f)  默认前景概率=1
        log_mix = torch.logaddexp(a, b)      # log(P_mix)，仍然 ≤ 0
        gamma   = torch.exp(b - log_mix)     # 后验 P(F=1 | z)


        pad_mask_float = (~pad_mask).float() #True的部分置0
        log_mix = log_mix * pad_mask_float
        gamma   = gamma   * pad_mask_float
        pi      = pi      * pad_mask_float

        return {
            "log_mix": log_mix,   # 区间混合概率的 log，≤0
            "gamma":   gamma,     # 前景后验
            "pi":      pi,        # 前景先验
            "valid_mask": pad_mask, # 与之前的模型匹配，返回的值叫做valid_mask
        }


# =========================
# 特征融合
# =========================
class SCEMFeatureFusion(nn.Module):
    """
    轻量多尺度融合：
      - 以最高分辨率 level 作为主尺度
      - 其他尺度 feature / spectral state 上采样到主尺度
      - feature 用 3x3 融合，spectral 用 1x1 融合
      - mask 直接使用主尺度 mask

    输入:
      features: List[Tensor], each [B, C_i, H_i, W_i]
      masks:    List[Bool],   each [B, H_i, W_i], True = padding
      specs:    List[Tensor], each [B, C_spec, H_i, W_i]

    输出:
      feat: [B, out_channels, H0, W0]
      mask: [B, H0, W0]                  (主尺度 mask)
      spec: [B, spec_channels, H0, W0]
    """

    def __init__(
        self,
        feat_in_channels,
        out_channels=256,
        spec_channels=8,
        feat_hidden=None,
        use_residual_sum=False,
    ):
        super().__init__()

        self.num_levels = len(feat_in_channels)
        self.out_channels = out_channels
        self.spec_channels = spec_channels
        self.use_residual_sum = use_residual_sum

        if feat_hidden is None:
            feat_hidden = out_channels

        # 每层 feature 对齐到统一通道
        self.feat_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                GN(out_channels),
                nn.SiLU(inplace=False),
            )
            for c in feat_in_channels
        ])

        # 每层 spectral state 轻量投影
        self.spec_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(spec_channels, spec_channels, kernel_size=1, bias=False),
                GN(spec_channels),
                nn.SiLU(inplace=False),
            )
            for _ in feat_in_channels
        ])

        if use_residual_sum:
            # 学习每个低层/高层的贡献
            self.feat_scales = nn.Parameter(torch.ones(self.num_levels))
            self.spec_scales = nn.Parameter(torch.ones(self.num_levels))

            self.feat_fuse = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                GN(out_channels),
                nn.SiLU(inplace=False),
            )
            self.spec_fuse = nn.Sequential(
                nn.Conv2d(spec_channels, spec_channels, kernel_size=1, bias=False),
                GN(spec_channels),
                nn.SiLU(inplace=False),
            )
        else:
            # concat 融合
            self.feat_fuse = nn.Sequential(
                nn.Conv2d(self.num_levels * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                GN(out_channels),
                nn.SiLU(inplace=False),
            )
            self.spec_fuse = nn.Sequential(
                nn.Conv2d(self.num_levels * spec_channels, spec_channels, kernel_size=1, bias=False),
                GN(spec_channels),
                nn.SiLU(inplace=False),
            )

    @staticmethod
    def _upsample_like(x: torch.Tensor, size_hw):
        if x.shape[-2:] == size_hw:
            return x
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _upsample_mask_like(mask: torch.Tensor, size_hw):
        """
        mask: [B,H,W], True=padding
        不需要使用
        """
        if mask.shape[-2:] == size_hw:
            return mask
        # nearest 保持 bool 语义
        mask_f = mask.unsqueeze(1).float()
        mask_up = F.interpolate(mask_f, size=size_hw, mode="nearest")
        return mask_up[:, 0] > 0.5

    def forward(self, features, masks, specs):
        assert len(features) == self.num_levels
        assert len(masks) == self.num_levels
        assert len(specs) == self.num_levels

        # 主尺度：最高分辨率
        H0, W0 = features[0].shape[-2:]
        out_mask = masks[0]  # 直接用主尺度 mask

        feat_list = []
        spec_list = []

        for i, (feat, mask, spec) in enumerate(zip(features, masks, specs)):
            # 先把无效区域清零，避免双线性上采样污染有效区域
            feat = _zero_invalid(feat, mask)
            spec = _zero_invalid(spec, mask)

            # 对齐到主尺度
            feat = self._upsample_like(feat, (H0, W0))
            spec = self._upsample_like(spec, (H0, W0))

            # 通道投影
            feat = self.feat_proj[i](feat)
            spec = self.spec_proj[i](spec)

            # 再用主尺度 mask 清零一次，保证最终输出无 padding 污染
            feat = _zero_invalid(feat, out_mask)
            spec = _zero_invalid(spec, out_mask)

            feat_list.append(feat)
            spec_list.append(spec)

        if self.use_residual_sum:
            feat = 0.0
            spec = 0.0
            for i in range(self.num_levels):
                feat = feat + self.feat_scales[i] * feat_list[i]
                spec = spec + self.spec_scales[i] * spec_list[i]

            feat = self.feat_fuse(feat)
            spec = self.spec_fuse(spec)
        else:
            feat = self.feat_fuse(torch.cat(feat_list, dim=1))
            spec = self.spec_fuse(torch.cat(spec_list, dim=1))

        # 最终再清零一次主尺度 padding 区域
        feat = _zero_invalid(feat, out_mask)
        spec = _zero_invalid(spec, out_mask)

        return feat, out_mask, spec
# =========================
# SCEM (返回 scem_out=posterior, aux=损失所需)
# =========================

def build(config):
    scem_enable = config.get("ENABLE", True)
    scem_gt = config.get("USE_GT")
    if scem_gt:
        scem_enable = False
        
    if not scem_enable:
        return None
    else:
        return SCEM(config)




class SCEM(nn.Module):
    """
    - 用 features[0] 生成后验概率 (gamma)
    - 返回: features, masks, scem_out(=gamma), aux(损失所需 log_mix/pi/valid_mask)
    """
    def __init__(self, config):
        super().__init__()
        self.cfg = dict(config)
        # self.with_foreground  = bool(self.cfg.get("WITH_FOREGROUND", False))
        self.depth      = int(self.cfg.get("DEPTH", 1))
        self.width      = float(self.cfg.get("WIDTH", 0.5))
        self.use_cache  = bool(self.cfg.get("USE_CACHE", True))
        self.in_ch      = int(self.cfg.get("IN_CHANNELS", 256))
        self.prior_mode = self.cfg.get("PRIOR_MODE")
        self.spectral_databse_num = int(self.cfg.get("SPECTRAL_DATABASE_NUM"))
        self.norm       = bool(self.cfg.get("NORM"))#在外部调用
        self.spectral_type = self.cfg.get("SPECTRAL_TYPE").lower()

        # self.lazy_built = False
        self.posterior = MixBGFG(
            C=self.in_ch,
            depth=self.depth,
            width=self.width,
            # with_foreground=self.with_foreground,
            use_cache=self.use_cache,
            prior_mode=self.prior_mode,
            spectral_databse_num=self.spectral_databse_num,
            spectral_type=self.spectral_type
        )

        self.feature_fusion = SCEMFeatureFusion(
            feat_in_channels=[256, 256, 256, 256],
            out_channels=self.in_ch,
            spec_channels=8,
            feat_hidden=self.in_ch,
        )
        #打印所有参数及其大小
        # for name, param in self.named_parameters():
            # print(f"SCEM param: {name}, size: {param.size()}")


    @torch.no_grad()
    def _check_masks(self, masks):
        assert isinstance(masks, (list, tuple)) and len(masks) > 0
        for m in masks:
            assert m.dtype == torch.bool and m.dim() == 3, "mask should be [B,H,W] bool"

    def forward(self, features, masks, specs, prior_map=None):
        """
        features: List[Tensor]   each [B, C_i, H_i, W_i]
        masks:    List[Bool]     each [B, H_i, W_i]  True=pad
        specs:    List[Tensor]   each [B, 8, H, W]

        Returns:
          features_passthrough,
          masks_passthrough,
          scem_out      -> posterior gamma [B,1,H0,W0]
          aux           -> dict for losses: {log_mix, pi, valid_mask}
        """
        assert isinstance(features, (list, tuple)) and len(features) > 0
        self._check_masks(masks)

        # TODO 多尺度特征融合
        feat, mask, spec  = self.feature_fusion(features, masks, specs)

        # feat0 = features[0]
        # mask0 = masks[0]  # [B,H0,W0] bool
        # 计算概率图
        out = self.posterior(feat, pad_mask=mask, prior_map=prior_map, spec=spec)

        # 使用_resize_posterior_to给出每个尺度的gamma
        gamma = out["gamma"]
        gamma_list = []
        for feature, mask in zip(features, masks):
            gamma_list.append(_zero_invalid(self._resize_posterior_to(feature, gamma), mask))
            

        return out["gamma"], out["log_mix"], gamma_list


    # =========================
    # Posterior-based multi-scale enhancement
    # =========================

    # @torch.no_grad()
    @staticmethod
    def _resize_posterior_to(feat: torch.Tensor, posterior: torch.Tensor):
        """
        posterior: [B,1,H0,W0]
        feat:      [B,C,Hi,Wi]
        returns:   [B,1,Hi,Wi]
        """

        # assert 长宽存在倍数关系
        assert posterior.shape[2] % feat.shape[2] == 0
        assert posterior.shape[3] % feat.shape[3] == 0

        ratio_h = posterior.shape[2] / feat.shape[2]
        ratio_w = posterior.shape[3] / feat.shape[3]
        kernel_size = (int(ratio_h), int(ratio_w))
        stride = (int(ratio_h), int(ratio_w))

        # return F.max_pool2d(posterior, kernel_size=kernel_size, stride=stride)
        return F.avg_pool2d(posterior, kernel_size=kernel_size, stride=stride)
        # return F.interpolate(posterior, size=feat.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def apply_posterior_enhance(features, masks, scem_out, alpha: float = 1.0):
        """
        使用后验概率(来自高分辨率 scem_out)对多尺度特征做门控增强：
        X_i' = X_i * (1 + alpha * γ_i)
        其中 γ_i 是将 scem_out 下采样到第 i 个尺度的后验图；在 mask 无效处置零。

        Args:
        features: List[Tensor], each [B,C_i,H_i,W_i]
        masks:    List[Bool],   each [B,H_i,W_i]
        scem_out: Tensor [B,1,H0,W0] (posterior from SCEM)
        alpha:    float, 增强强度

        Returns:
        enhanced_features: List[Tensor] (与 features 对齐)
        """
        assert isinstance(features, (list, tuple)) and len(features) > 0
        assert isinstance(masks,    (list, tuple)) and len(masks)    == len(features)
        B = features[0].size(0)
        assert scem_out.dim() == 4 and scem_out.size(0) == B and scem_out.size(1) == 1

        enhanced = []
        for xi, mi in zip(features, masks):
            # 1) 将 posterior 下采样到该尺度
            gi = SCEM._resize_posterior_to(xi, scem_out).clamp(0.0, 1.0)
            # 2) mask 无效处置零
            if mi.dim() == 3:
                gi = gi * (~mi).unsqueeze(1).float()
            # 3) 门控增强
            xi_enh = xi * (1.0 + alpha * gi)
            enhanced.append(xi_enh)
        return enhanced
