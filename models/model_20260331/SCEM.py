import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.GDN import GDN
from typing_extensions import override

def _log_sigma_from_raw(raw_log_sigma: torch.Tensor, eps: float = 1e-4):
    return torch.log(F.softplus(raw_log_sigma) + eps)


def GN(ch, num_groups=32):
    return nn.GroupNorm(min(num_groups, ch), ch)


def _zero_invalid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif not (mask.dim() == 4 and mask.size(1) == 1):
        raise ValueError(f"mask shape should be [B,H,W] or [B,1,H,W], got {mask.shape}")
    valid = (~mask).to(dtype=x.dtype)
    return x * valid


class CoordConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_cache=True, cache_size=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + 2, out_ch, k, s, p, bias=False)
        self.bn = GN(out_ch)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x, pad_mask: torch.Tensor = None):
        batch_size, _, height, width = x.shape
        if pad_mask.dim() == 3:
            pad = pad_mask.unsqueeze(1)
        elif pad_mask.dim() == 4 and pad_mask.size(1) == 1:
            pad = pad_mask
        else:
            raise ValueError(f"pad_mask shape should be [B,H,W] or [B,1,H,W], got {pad_mask.shape}")

        valid = ~pad
        row_has_valid = valid.any(dim=3).squeeze(1)
        col_has_valid = valid.any(dim=2).squeeze(1)

        h_valid = row_has_valid.sum(dim=1).view(batch_size, 1, 1).clamp(min=1)
        w_valid = col_has_valid.sum(dim=1).view(batch_size, 1, 1).clamp(min=1)

        yy = torch.arange(height, device=x.device, dtype=x.dtype).view(1, height, 1).expand(batch_size, height, width)
        xx = torch.arange(width, device=x.device, dtype=x.dtype).view(1, 1, width).expand(batch_size, height, width)

        yy_max = (h_valid - 1).clamp(min=0)
        xx_max = (w_valid - 1).clamp(min=0)
        yy = torch.minimum(yy, yy_max)
        xx = torch.minimum(xx, xx_max)

        den_h = (h_valid - 1).clamp(min=1)
        den_w = (w_valid - 1).clamp(min=1)
        yy = (yy / den_h) * 2.0 - 1.0
        xx = (xx / den_w) * 2.0 - 1.0

        cc = torch.stack([xx, yy], dim=1)
        x = torch.cat([x, cc], dim=1)
        return self.act(self.bn(self.conv(x)))


class CenterMaskedConv3x3(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=False)
        mask = torch.ones(3, 3)
        mask[1, 1] = 0.0
        self.register_buffer("mask", mask.view(1, 1, 3, 3))
        self.bn = GN(ch)
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
                nn.SiLU(inplace=False),
            )
            for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, 1, 1, 0, bias=False),
            GN(out_ch),
            nn.SiLU(inplace=False),
        )

    def forward(self, x):
        ys = [m(x) for m in self.branches]
        return self.proj(torch.cat(ys, dim=1))


class SpectralManifold(nn.Module):
    def __init__(self, spectral_channels: int = 8, num_prototypes: int = 64, hidden_ratio: float = 2.0, eps: float = 1e-6):
        super().__init__()
        ch = spectral_channels
        num_proto = num_prototypes
        hidden = int(ch * hidden_ratio)
        self.ch = ch
        self.num_proto = num_proto
        self.eps = eps
        self.encoder = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False),
        )
        self.prototypes = nn.Parameter(torch.randn(num_proto, ch))
        nn.init.normal_(self.prototypes, std=0.02)

    def forward(self, spec):
        _, ch, _, _ = spec.shape
        assert ch == self.ch
        s_enc = self.encoder(spec)
        s_norm = s_enc - s_enc.mean(dim=1, keepdim=True)
        s_norm = F.normalize(s_norm, dim=1)
        proto = self.prototypes
        proto_norm = proto - proto.mean(dim=1, keepdim=True)
        proto_norm = F.normalize(proto_norm, dim=1)
        sim = F.conv2d(s_norm, proto_norm.view(self.num_proto, self.ch, 1, 1))
        weight = F.softmax(sim, dim=1)
        recon = torch.einsum("bkhw,kc->bchw", weight, proto)
        error = (spec - recon).pow(2).sum(dim=1, keepdim=True)
        return torch.cat([recon, error], dim=1)


class SpectralPi(nn.Module):
    def __init__(self, spectral_database_num=64, in_ch=8, eps=1e-6):
        super().__init__()
        self.K = spectral_database_num
        self.C = in_ch
        self.eps = eps
        self.spectral_db_logits = nn.Parameter(torch.randn(self.K, self.C))

    def forward(self, spec):
        _, channels, _, _ = spec.shape
        assert channels == self.C, f"SpectralPi expects {self.C} channels, got {channels}"
        s = spec
        mu_s = s.mean(dim=1, keepdim=True)
        s_zero = s - mu_s
        s_norm = s_zero / (s_zero.norm(dim=1, keepdim=True) + self.eps)

        db = self.spectral_db_logits
        mu_db = db.mean(dim=1, keepdim=True)
        db_zero = db - mu_db
        db_norm = db_zero / (db_zero.norm(dim=1, keepdim=True) + self.eps)
        weight = db_norm.view(self.K, self.C, 1, 1)
        return F.conv2d(s_norm, weight=weight, bias=None)


class SpectralGraphDictionaryPrior(nn.Module):
    def __init__(self, spectral_channels=8, num_prototypes=64, tau=1.0):
        super().__init__()
        self.C = spectral_channels
        self.K = num_prototypes
        self.tau = tau
        self.prototypes = nn.Parameter(torch.randn(self.K, self.C))
        nn.init.normal_(self.prototypes, std=0.02)
        self.proto_obj = nn.Parameter(torch.full((self.K,), 0.0))

    def diversity_loss(self):
        p = F.normalize(self.prototypes, dim=1)
        gram = torch.matmul(p, p.t())
        identity = torch.eye(self.K, device=p.device)
        return ((gram - identity) ** 2).mean()

    def forward(self, spec):
        batch_size, channels, height, width = spec.shape
        s = spec - spec.mean(dim=1, keepdim=True)
        s = F.normalize(s, dim=1)
        proto = self.prototypes
        proto_norm = proto - proto.mean(dim=1, keepdim=True)
        proto_norm = F.normalize(proto_norm, dim=1)
        sim = F.conv2d(s, proto_norm.view(self.K, channels, 1, 1))
        w = F.softmax(sim / self.tau, dim=1)
        a = torch.matmul(proto_norm, proto_norm.t())
        a = F.softmax(a, dim=1)
        w_flat = w.view(batch_size, self.K, -1)
        w_graph = torch.matmul(a, w_flat).view(batch_size, self.K, height, width)
        obj = self.proto_obj.view(1, self.K, 1, 1)
        score = torch.sigmoid(obj)
        return w_graph * score


class PIHead(nn.Module):
    def __init__(self, in_ch, ch=128, use_cache=True, spectral_type=None, spectral_databse_num=64):
        nn.Module.__init__(self)

        space_ch = ch // 2
        global_ch = max(ch // 4, 1)
        gate_hidden_ch = max(ch // 4, 1)
        self.spectral_type = spectral_type.lower()

        if self.spectral_type == "pi":
            self.spec_channels = spectral_databse_num
        elif self.spectral_type == "manifold":
            self.spec_channels = 8 + 1
        elif self.spectral_type == "graph_dictionary":
            self.spec_channels = spectral_databse_num
        else:
            raise ValueError(f"Invalid spectral method: {self.spectral_type}")

        # 构造空间特征
        self.coord = CoordConv(in_ch, space_ch, use_cache=use_cache)
        self.mask1 = CenterMaskedConv3x3(space_ch)
        self.aspp = ASPP(space_ch, space_ch, rates=(1, 2, 3, 5))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(space_ch, global_ch, 1, 1, 0, bias=False),
            # GN(global_ch), 是单一像素，没办法norm
            nn.SiLU(inplace=False)
        )
        self.space_fuse = nn.Sequential(
            nn.Conv2d(space_ch + global_ch, ch, 3, 1, 1, bias=False),
            GN(ch),
            nn.SiLU(inplace=False)
        )

        # 空间特征对光谱特征调制
        self.space_to_spec_gate = nn.Sequential(
            nn.Conv2d(ch, gate_hidden_ch, 3, 1, 1, bias=False),
            GN(gate_hidden_ch),
            nn.SiLU(inplace=False),
            nn.Conv2d(gate_hidden_ch, self.spec_channels, 1, 1, 0),
        )
        self.out = nn.Conv2d(self.spec_channels, 1, 3, 1, 1, bias=False)
        # self.enhance_alpha = nn.Parameter(torch.tensor(1.0))

        # 光谱相似度
        if self.spectral_type == "pi":
            self.spec_pi = SpectralPi(spectral_databse_num)
        elif self.spectral_type == "manifold":
            self.spec_manifold = SpectralManifold(num_prototypes=spectral_databse_num)
        elif self.spectral_type == "graph_dictionary":
            self.spec_graph_dictionary = SpectralGraphDictionaryPrior(num_prototypes=spectral_databse_num)

    def specpi(self, spec):
        if self.spectral_type == "pi":
            return self.spec_pi(spec)
        if self.spectral_type == "manifold":
            return self.spec_manifold(spec)
        if self.spectral_type == "graph_dictionary":
            return self.spec_graph_dictionary(spec)
        raise ValueError(f"Invalid spectral method: {self.spectral_type}")

    @override
    def forward(self, x, spec, pad_mask):
        a = self.coord(x, pad_mask=pad_mask)
        a = self.mask1(a)
        a = self.aspp(a)
        a = _zero_invalid(a, pad_mask)

        g = self.global_fc(self.global_pool(a))
        g = g.expand(-1, -1, a.shape[-2], a.shape[-1]).contiguous()
        f_space = self.space_fuse(torch.cat([a, g], dim=1))
        f_space = _zero_invalid(f_space, pad_mask) #[B, ch, H, W]

        f_spec = _zero_invalid(self.specpi(spec), pad_mask) #[B, spec_channels, H, W]
        w = torch.tanh(self.space_to_spec_gate(f_space)) #[B, spec_channels, H, W]
        e_k = (1 + w) * f_spec# [B, ch, H, W]

        pi = torch.sigmoid(self.out(e_k))#[B, 1, H, W]

        return pi, e_k


class BGHead(nn.Module):
    def __init__(self, in_ch, out_ch, interval_width: float = 1.0):
        super().__init__()
        self.interval_width = interval_width
        self.mu_b_head = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.logsig_b_head = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.logsig_b_head.weight)
            self.logsig_b_head.bias.fill_(1.313)

    def forward(self, x, z):
        mu_b = self.mu_b_head(x)
        log_sigb = self.logsig_b_head(x)
        return self.log_gaussian_interval(z, mu_b, log_sigb, width=self.interval_width)

    def log_gaussian_interval(self, z, mu, raw_log_sigma, width: float = 1.0, eps: float = 1e-8):
        log_sigma = _log_sigma_from_raw(raw_log_sigma)
        sigma = torch.exp(log_sigma)
        half = width * 0.5
        dist = Normal(loc=mu, scale=sigma)
        upper = dist.cdf(z + half)
        lower = dist.cdf(z - half)
        p = (upper - lower).clamp_min(eps)
        return torch.log(p).sum(dim=1, keepdim=True)


class MixBGFG(nn.Module):
    '''
        与20260310相比, 想要输出的是由光谱库得到的余弦相似度类似的东西
    '''
    def __init__(self, C, depth=4, width=0.5, with_foreground=True, use_cache=True, tau_mode="mean", prior_mode="gate", spectral_type="pi", spectral_databse_num=128):
        nn.Module.__init__(self)
        ch = int(C * width)
        self.tau_mode = tau_mode
        self.tau = 1.0 / ch if tau_mode == "mean" else 1.0 / math.sqrt(ch)

        self.stem = nn.Sequential(
            nn.Conv2d(C, ch, 3, 1, 1, bias=False),
            GDN(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            GDN(ch),
            nn.SiLU(),
        )

        self.bg_head = BGHead(in_ch=ch, out_ch=C)
        self.pi_head = PIHead(
            in_ch=ch,
            use_cache=use_cache,
            spectral_type=spectral_type,
            spectral_databse_num=spectral_databse_num,
        )

        self.prior_mode = prior_mode.lower()
        assert self.prior_mode == "gate"
        # self.gate_head = nn.Sequential(
        #     nn.Conv2d(ch + 1, ch // 2, 3, 1, 1),
        #     GN(ch // 2),
        #     nn.SiLU(),
        #     nn.Conv2d(ch // 2, 1, 1, 1, 0),
        # )

    @override
    def forward(self, Z, pad_mask, interval_width: float = 1.0, prior_map = None, spec=None):
        """
        Z: [B,C,H,W]
        pad_mask: [B,H,W] or [B,1,H,W], True = padding
        interval_width: 区间 Δ，默认为 1.0 → [x-0.5, x+0.5]
        """
        #TODO 暂时没有使用到prior_map

        if pad_mask.dim() == 3:
            pad_mask = pad_mask.unsqueeze(1)  # [B,1,H,W]

        x = self.stem(Z)

        # 背景区间 log prob
        log_pb = self.tau * self.bg_head(x, Z)


        # π 先验：将 pad_mask 传入 PIHead / CoordConv，使坐标只在有效区域内为 [-1,1]
        pi_net, spectral_evidence = self.pi_head(x, spec, pad_mask)   # [B,1,H,W]
        pi = pi_net.clamp(1e-6, 1 - 1e-6)



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
            "spectral_evidence": spectral_evidence, # 光谱证据
            "spectral_dict": self.pi_head.spec_pi.spectral_db_logits, # 光谱字典
        }


class SCEMFeatureFusion(nn.Module):
    """
        与20260310相比，光谱通道采用 主层+残差补充的方式实现
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

        # 主特征用concat 融合
        self.feat_fuse = nn.Sequential(
            nn.Conv2d(self.num_levels * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            GN(out_channels),
            nn.SiLU(inplace=False),
        )

        
        # 光谱特征用残差
        # spec: 主层 + 残差补充
        self.spec_scales = nn.Parameter(torch.tensor([1.0] + [0.2]*(self.num_levels-1)))
        self.spec_fuse = nn.Sequential(
            nn.Conv2d(spec_channels, 32, 3, padding=1, bias=False),
            GN(32),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, spec_channels, 1, bias=False),
            GN(spec_channels),
            nn.SiLU(inplace=False),
        )

    @staticmethod
    @override
    def _upsample_like(x: torch.Tensor, size_hw):
        raise NotImplementedError("This method is not implemented")
    
    @staticmethod
    def _upsample_like_feat(x, size_hw):
        if x.shape[-2:] == size_hw:
            return x
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _upsample_like_spec(x, size_hw):
        if x.shape[-2:] == size_hw:
            return x
        return F.interpolate(x, size=size_hw, mode="nearest")

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
            feat = self._upsample_like_feat(feat, (H0, W0))
            spec = self._upsample_like_spec(spec, (H0, W0))

            # 通道投影
            feat = self.feat_proj[i](feat)
            spec = self.spec_proj[i](spec)

            # 再用主尺度 mask 清零一次，保证最终输出无 padding 污染
            feat = _zero_invalid(feat, out_mask)
            spec = _zero_invalid(spec, out_mask)

            feat_list.append(feat)
            spec_list.append(spec)

        feat = self.feat_fuse(torch.cat(feat_list, dim=1))
        spec = self.spec_scales[0] * spec_list[0]
        for i in range(1, self.num_levels):
            spec = spec + self.spec_scales[i] * spec_list[i]
        spec = self.spec_fuse(spec)

        # 最终再清零一次主尺度 padding 区域
        feat = _zero_invalid(feat, out_mask)
        spec = _zero_invalid(spec, out_mask)

        return feat, out_mask, spec


class SCEM(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.cfg = dict(config)
        self.depth = int(self.cfg.get("DEPTH", 1))
        self.width = float(self.cfg.get("WIDTH", 0.5))
        self.use_cache = bool(self.cfg.get("USE_CACHE", True))
        self.in_ch = int(self.cfg.get("IN_CHANNELS", 256))
        self.prior_mode = self.cfg.get("PRIOR_MODE")
        self.spectral_database_num = int(self.cfg.get("SPECTRAL_DATABASE_NUM"))
        self.spectral_type = self.cfg.get("SPECTRAL_TYPE").lower()

        self.posterior = MixBGFG(
            C=self.in_ch,
            depth=self.depth,
            width=self.width,
            use_cache=self.use_cache,
            prior_mode=self.prior_mode,
            spectral_databse_num=self.spectral_database_num,
            spectral_type=self.spectral_type,
        )

        self.feature_fusion = SCEMFeatureFusion(
            feat_in_channels=[256, 256, 256, 256],
            out_channels=self.in_ch,
            spec_channels=8,
            feat_hidden=self.in_ch,
        )

        # GroupNorm for SCEM enhanced features
        self.scem_norms = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=self.in_ch)
            for _ in range(4)
        ])
        
        self.gamma_resize_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, 3, padding=1, bias=True),
                nn.Tanh(),
            )
            for _ in range(3)
        ])

        self.spectral_evidence_resize_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.spectral_database_num, self.spectral_database_num, 3, padding=1, bias=False),
                GN(self.spectral_database_num),
                nn.SiLU(inplace=False),
                nn.Conv2d(self.spectral_database_num, self.spectral_database_num, 1, bias=True),
            )
            for _ in range(3)
        ])


        self.evi_hidden_dim = 64
        #! 每个evidence使用自己的推理网络
        self.evidence_relation_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.spectral_database_num, self.evi_hidden_dim, 1, bias=False),
                GN(self.evi_hidden_dim),
                nn.SiLU(inplace=False),

                nn.Conv2d(self.evi_hidden_dim, self.evi_hidden_dim, 3, padding=1, groups=self.evi_hidden_dim, bias=False),
                GN(self.evi_hidden_dim),
                nn.SiLU(inplace=False),

                nn.Conv2d(self.evi_hidden_dim, self.evi_hidden_dim, 1, bias=False),
                GN(self.evi_hidden_dim),
                nn.SiLU(inplace=False),
            )
            for _ in range(4)
        ])
        
        self.evidence_weight_head_pos = nn.ModuleList([
            nn.Conv2d(self.evi_hidden_dim, num_tokens * self.spectral_database_num, kernel_size=1, bias=True)
            for num_tokens in [8, 4, 2, 1]
        ])
        self.evidence_weight_head_neg = nn.ModuleList([
        nn.Conv2d(self.evi_hidden_dim, num_tokens * self.spectral_database_num, kernel_size=1, bias=True)
        for num_tokens in [8, 4, 2, 1]
        ])
        # 每层一个负证据抑制系数，初值 0.5
        self.lambda_neg = nn.Parameter(torch.full((4,), 0.5))

        # 每层每个token一个gate偏置，抬阈值压背景
        self.gate_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_tokens, 1, 1))
            for num_tokens in [8, 4, 2, 1]
        ])

        # 每层每个token一个空间温度，初值 0.5
        self.gate_tau = nn.ParameterList([
            nn.Parameter(torch.full((1, num_tokens, 1, 1), 0.5))
            for num_tokens in [8, 4, 2, 1]
        ])

        # token输出后做norm，再和feature token拼接
        self.evidence_token_norms = nn.ModuleList([
            nn.LayerNorm(self.in_ch) for _ in range(4)
        ])

        # # 如果你后面要把 spectral_part 投到同一维再融合
        self.evidence_spec_proj = nn.ModuleList([
            nn.Linear(8, self.in_ch, bias=True) for _ in range(4)
        ])
        self.evidence_temperature = nn.Parameter(torch.full((4,), 1.0))



        # 初始化gamma_resize_heads的conv，使其输出全0
        with torch.no_grad():
            for i in range(3):
                nn.init.zeros_(self.gamma_resize_heads[i][0].weight)
                nn.init.zeros_(self.gamma_resize_heads[i][0].bias)

                nn.init.zeros_(self.spectral_evidence_resize_heads[i][-1].weight)
                nn.init.zeros_(self.spectral_evidence_resize_heads[i][-1].bias)

    @torch.no_grad()
    def _check_masks(self, masks):
        assert isinstance(masks, (list, tuple)) and len(masks) > 0
        for m in masks:
            assert m.dtype == torch.bool and m.dim() == 3, "mask should be [B,H,W] bool"


    def _resize_by_mixed_pool(self, x: torch.Tensor, out_hw, lam:float):
        H0, W0 = x.shape[-2:]
        H, W = out_hw
        if (H0, W0) == (H, W):
            return x
        assert H0 % H == 0 and W0 % W == 0, \
            f"Only integer downsampling supported, but got {H0}x{W0} -> {H}x{W}"
        kh = H0 // H
        kw = W0 // W
        avg_pool = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        max_pool = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        return lam * avg_pool + (1 - lam) * max_pool


    @override
    def _resize_posterior_to(self, feat: torch.Tensor, posterior: torch.Tensor, alpha=0.5):
        raise NotImplementedError("This method is not implemented")

    def apply_posterior_enhance(self, features, masks, scem_out, alpha: float = 1.0):
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
        for lvl, (xi, mi) in enumerate(zip(features, masks)):
            gi = self._resize_posterior_to(xi, scem_out).clamp(0.0, 1.0)
            gi = self.gamma_resize_heads[lvl](gi)
            if mi.dim() == 3:
                gi = gi * (~mi).unsqueeze(1).float()
            xi_enh = xi * (1.0 + alpha * gi)
            enhanced.append(xi_enh)
        return enhanced

    @override
    def forward(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor],
        specs: list[torch.Tensor],
        prior_map=None,
        return_debug: bool = False,
    ):
        """
        features: List[Tensor]   each [B, C_i, H_i, W_i]
        masks:    List[Bool]     each [B, H_i, W_i]  True=pad
        specs:    List[Tensor]   each [B, 8, H, W]

        Returns:

            返回所有的src和新筛选出来的tokne
        """
        assert isinstance(features, (list, tuple)) and len(features) > 0
        self._check_masks(masks)
        B = features[0].size(0)
        
        feat, mask, spec  = self.feature_fusion(features, masks, specs)

        # feat0 = features[0]
        # mask0 = masks[0]  # [B,H0,W0] bool
        # 计算概率图
        # return out: log_mix gamma pi valid_mask spectral_evidence spectral_dict
        out = self.posterior(feat, pad_mask=mask, prior_map=prior_map, spec=spec) 
        
        # 特征增强
        gamma = out["gamma"]
        feat_enhanced = []
        gamma_levels = []
        for i, feat in enumerate(features):
            Hi, Wi = feat.shape[-2:]
            if i == 0:
                gamma_i = gamma
            else:
                gamma_resize_i = self._resize_by_mixed_pool(gamma, (Hi, Wi), 0.5)
                gamma_calib_i = self.gamma_resize_heads[i-1](gamma_resize_i) #经过tanh激活
                gamma_i = gamma_resize_i * (1.0 + 0.5 * gamma_calib_i) #-0.5 -> 1.5门控
            feat_enh_i = feat * (1.0 + 1.0 * gamma_i)
            feat_enh_i = self.scem_norms[i](feat_enh_i)
            feat_enhanced.append(feat_enh_i)
            gamma_levels.append(gamma_i)


        # 光谱resize
        spectral_evidence = out["spectral_evidence"]
        spectral_evidence_multilevel = []
        for i, feat in enumerate(features):
            Hi, Wi = feat.shape[-2:]
            if i == 0:
                spectral_evidence_resize_i = spectral_evidence
            else:
                spectral_evidence_resize_i = self._resize_by_mixed_pool(spectral_evidence, (Hi, Wi), 0.5)
                spectral_evidence_calib_i = self.spectral_evidence_resize_heads[i-1](spectral_evidence_resize_i) 
                spectral_evidence_resize_i = spectral_evidence_resize_i + spectral_evidence_calib_i
            spectral_evidence_multilevel.append(spectral_evidence_resize_i)

        # token生成
        evidence_tokens = []
        evidence_tokens_spectral_part = []
        debug_evidence_weights = []
        debug_pool_weights = []
        debug_pool_logits = []
        debug_token_feature_corr = []
        debug_gate = []

        pool_weights_levels: list[torch.Tensor] = []

        spectral_dict = out["spectral_dict"]  # [K, 8]
        token_nums = [8, 4, 2, 1]

        for i, (feat, mask, token_num) in enumerate(zip(features, masks, token_nums)):
            H_i, W_i = feat.shape[-2:]

            # --------------------------------------------------
            # 1) evidence relation feature
            # --------------------------------------------------
            evi = spectral_evidence_multilevel[i]   # [B, K, H, W], signed
            relation_feat = self.evidence_relation_nets[i](evi)

            # --------------------------------------------------
            # 2) 正负evidence分解
            # --------------------------------------------------
            evi_pos = F.relu(evi)
            evi_neg = F.relu(-evi)

            # --------------------------------------------------
            # 3) 位置型 K维 evidence mixing
            #    仅在K维做softmax，不做token维softmax
            # --------------------------------------------------
            logits_pos = self.evidence_weight_head_pos[i](relation_feat)
            logits_neg = self.evidence_weight_head_neg[i](relation_feat)

            logits_pos = logits_pos.view(B, token_num, self.spectral_database_num, H_i, W_i)
            logits_neg = logits_neg.view(B, token_num, self.spectral_database_num, H_i, W_i)

            invalid_mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,H,W]
            logits_pos = logits_pos.masked_fill(invalid_mask, -1e4)
            logits_neg = logits_neg.masked_fill(invalid_mask, -1e4)

            a_pos = torch.softmax(logits_pos / self.evidence_temperature[i], dim=2)   # [B,T,K,H,W]
            a_neg = torch.softmax(logits_neg / self.evidence_temperature[i], dim=2)   # [B,T,K,H,W]

            a_pos = a_pos.masked_fill(invalid_mask, 0.0)
            a_neg = a_neg.masked_fill(invalid_mask, 0.0)

            # --------------------------------------------------
            # 4) signed gate，不再做 spatial softmax
            # --------------------------------------------------
            gate, pool_logits, support_map, suppress_map = self._build_signed_gate(
                evi_pos=evi_pos,
                evi_neg=evi_neg,
                a_pos=a_pos,
                a_neg=a_neg,
                gate_bias=self.gate_bias[i],
                gate_tau=self.gate_tau[i],
                lambda_neg=self.lambda_neg[i],
                mask=mask,
            )  # gate: [B,T,H,W]

            # --------------------------------------------------
            # 5) weighted average token
            # --------------------------------------------------
            tokens, pool_weights = self._masked_weighted_avg_tokens(
                feat=feat,
                gate=gate,
                mask=mask,
            )  # tokens:[B,T,C], pool_weights:[B,T,H,W] 仅用于调试/可视化
            pool_weights_levels.append(pool_weights)

            # token后归一化，后续送DETR encoder更稳
            tokens = self.evidence_token_norms[i](tokens)

            # --------------------------------------------------
            # 6) beta：只从正证据路得到token-level evidence mixture
            #    更符合“token由哪些支持性evidence构成”
            # --------------------------------------------------
            beta = a_pos * gate.unsqueeze(2)                     # [B,T,K,H,W]
            beta = beta.flatten(3).sum(dim=-1)                  # [B,T,K]
            beta = beta / beta.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            spectral_part = torch.einsum("btk,kc->btc", beta, spectral_dict)   # [B,T,8]

            evidence_tokens.append(tokens)
            evidence_tokens_spectral_part.append(spectral_part)

            if return_debug:
                debug_evidence_weights.append({
                    "a_pos": a_pos.detach(),
                    "a_neg": a_neg.detach(),
                })
                debug_pool_weights.append(pool_weights.detach())    # 归一化后仅用于可视化
                debug_pool_logits.append(pool_logits.detach())
                debug_gate.append(gate.detach())

                token_corr_all_levels = []
                for token_idx in range(tokens.shape[1]):
                    token_vec = F.normalize(tokens[:, token_idx, :], dim=-1)  # [B, C]
                    corr_per_level = []
                    for feat_level in features:
                        feat_norm = F.normalize(feat_level, dim=1)  # [B, C, H, W]
                        corr_map = torch.einsum("bc,bchw->bhw", token_vec, feat_norm).unsqueeze(1)  # [B,1,H,W]
                        corr_per_level.append(corr_map.detach())
                    token_corr_all_levels.append(corr_per_level)
                debug_token_feature_corr.append(token_corr_all_levels)

        
        # 构造global token
        global_token = []
        global_token_spectral_part = []
        for i, (feat, spec) in enumerate(zip(features, specs)):
            global_token_i = feat.mean(dim=(2, 3))
            global_token.append(global_token_i)
            global_token_spectral_part_i = spec.mean(dim=(2, 3))
            global_token_spectral_part.append(global_token_spectral_part_i)

        device = features[0].device
        dtype = features[0].dtype
        loss_pool_div_total = torch.zeros((), device=device, dtype=dtype)
        loss_gamma_cover_total = torch.zeros((), device=device, dtype=dtype)
        if pool_weights_levels:
            for pool_weights, gamma_i, mask in zip(pool_weights_levels, gamma_levels, masks):
                loss_pool_div_total = loss_pool_div_total + self._loss_pool_diversity(pool_weights, mask)
                loss_gamma_cover_total = loss_gamma_cover_total + self._loss_gamma_coverage(
                    pool_weights, gamma_i, mask
                )
            n_lvl = float(len(pool_weights_levels))
            loss_pool_div_total = loss_pool_div_total / n_lvl
            loss_gamma_cover_total = loss_gamma_cover_total / n_lvl

        scem_aux_losses = {
            "loss_pool_div": loss_pool_div_total,
            "loss_gamma_cover": loss_gamma_cover_total,
        }

        # return 特征 + evidence特征 + 损失计算部分
        # feat_enhanced: n_feature_levels * [(B, C, H, W)]
        # specs : n_feature_levels * [(B, 8, H, W)]
        # evidence_tokens: n_feature_levels * [(B, top_m, C)]
        # evidence_tokens_spectral_part: n_feature_levels * [(B, top_m, 8)]
        # global_token: [(B, C)]
        # global_token_spectral_part: [(B, 8)]
        # gamma: [B,1,H0,W0]
        # log_mix: [B,1,H0,W0]
        # spectral_dict: [K, 8]
        
        if return_debug:
            debug_info = {
                "evidence_weights": debug_evidence_weights,      # level -> [B,T,K,H,W]
                "pool_weights": debug_pool_weights,              # level -> [B,T,H,W]
                "pool_logits": debug_pool_logits,                # level -> [B,T,H,W]
                "gate": debug_gate,                              # level -> [B,T,H,W]
                "evidence_tokens": [t.detach() for t in evidence_tokens],  # level -> [B,T,C]
                "token_feature_corr": debug_token_feature_corr,  # src_level -> token -> tgt_level -> [B,1,H,W]
                "spectral_evidence_multilevel": spectral_evidence_multilevel,
                "scem_aux_losses": scem_aux_losses,
            }
            return (
                (feat_enhanced, specs),
                (evidence_tokens, evidence_tokens_spectral_part),
                (global_token, global_token_spectral_part),
                (gamma, out['log_mix'], spectral_dict),
                debug_info,
            )

        return (
            (feat_enhanced, specs),
            (evidence_tokens, evidence_tokens_spectral_part),
            (global_token, global_token_spectral_part),
            (gamma, out["log_mix"], spectral_dict, scem_aux_losses),
        )
        

    def _masked_weighted_avg_tokens(self, feat, gate, mask=None, eps=1e-6):
        """
        feat: [B, C, H, W]
        gate: [B, T, H, W], non-negative
        mask: [B, H, W], True=invalid
        return:
            tokens: [B, T, C]
            pool_weights_vis: [B, T, H, W]  # 仅用于可视化/调试
        """
        if mask is not None:
            gate = gate.masked_fill(mask.unsqueeze(1), 0.0)

        feat_flat = feat.flatten(2)              # [B, C, HW]
        gate_flat = gate.flatten(2)              # [B, T, HW]

        num = torch.einsum("bts,bcs->btc", gate_flat, feat_flat)   # [B, T, C]
        den = gate_flat.sum(dim=-1, keepdim=True).clamp_min(eps)   # [B, T, 1]
        tokens = num / den

        pool_weights_vis = gate / gate.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return tokens, pool_weights_vis


    def _build_signed_gate(self, evi_pos, evi_neg, a_pos, a_neg, gate_bias, gate_tau, lambda_neg, mask=None):
        """
        evi_pos/evi_neg: [B, K, H, W]
        a_pos/a_neg    : [B, T, K, H, W]
        return:
            gate: [B, T, H, W] non-negative
            pool_logits: [B, T, H, W]
            support/suppress: [B, T, H, W]
        """
        support = (a_pos * evi_pos.unsqueeze(1)).sum(dim=2)      # [B,T,H,W]
        suppress = (a_neg * evi_neg.unsqueeze(1)).sum(dim=2)     # [B,T,H,W]

        pool_logits = support - lambda_neg * suppress

        tau = F.softplus(gate_tau) + 1e-4
        gate = F.softplus((pool_logits - gate_bias) / tau)

        if mask is not None:
            gate = gate.masked_fill(mask.unsqueeze(1), 0.0)

        return gate, pool_logits, support, suppress


    def _masked_softmax_spatial(
                            self,
                            logits: torch.Tensor,
                            mask: torch.Tensor,
                            eps: float = 1e-6) -> torch.Tensor:
        """
        对每个 token 的空间维做 masked softmax
        logits: [B, T, H, W]
        mask  : [B, H, W]，True 表示 invalid
        return: [B, T, H, W]
        """
        B, T, H, W = logits.shape
        x = logits.view(B, T, -1)  # [B, T, HW]

        mask_flat = mask.view(B, 1, -1).expand(-1, T, -1)  # [B,T,HW]
        x = x.masked_fill(mask_flat, float("-inf"))

        out = torch.softmax(x, dim=-1)
        out = out.view(B, T, H, W)

        out = out.masked_fill(mask.unsqueeze(1), 0.0)
        denom = out.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        out = out / denom
  
        return out


    def _top_p_mean(self, score_map: torch.Tensor, top_p: int):
        """
        score_map: [B, K, H, W]
        返回每个prototype的top-p 平均分数:[B, K]
        """
        B, K, H, W = score_map.shape
        flat = score_map.flatten(2)
        p = min(top_p, flat.shape[-1])
        topv, _ = torch.topk(flat, p, dim=-1)
        return topv.mean(dim=-1)

    def build_evidence_tokens(
        self, 
        feat: torch.Tensor,
        evidence: torch.Tensor,
        top_m: int = 2,
        top_p: int = 16,
        tau: float = 0.07,
        mask: torch.Tensor = None,
    ):
        """
        根据某一层的 evidence 从 feat 中池化出 token

        输入:
            feat:     [B, C, H, W]
            evidence: [B, K, H, W]   signed evidence
            mask:     [B, H, W], True=padding
        输出:
            tokens:   [B, top_m, C]
            top_idx:  [B, top_m]
            assign:   [B, K, H, W]   prototype assignment map after softmax
        """
        B, C, H, W = feat.shape
        _, K, _, _ = evidence.shape
        eps = 1e-6

        # 1) tokenization 时再转成非负 assignment
        assign = torch.softmax(evidence / tau, dim=1)  # [B,K,H,W]

        if mask is not None:
            valid = (~mask).unsqueeze(1).float()       # [B,1,H,W]
            assign = assign * valid

            # 重新按 prototype 维归一化，避免 mask 后概率和变小
            denom = assign.sum(dim=1, keepdim=True).clamp_min(eps)
            assign = assign / denom


        # 2) 用 top-p spatial average 选最“目标化”的 prototype
        proto_score = self._top_p_mean(assign, top_p=top_p)   # [B,K]
        top_idx = torch.topk(proto_score, k=top_m, dim=1).indices  # [B,top_m]

        # 3) 每个被选中的 prototype 单独池化出 token
        M = top_m
        gather_idx = top_idx[:,:,None,None].expand(-1, -1, H, W)
        A = torch.gather(assign, dim=1, index=gather_idx)

        if mask is not None:
            valid = (~mask).unsqueeze(1).float()
            A = A * valid

        A = A / (A.sum(dim=(2, 3), keepdim=True) + eps)
        tokens = torch.einsum("bchw, bmhw->bmc", feat, A)

        return tokens, top_idx, assign


    def _loss_pool_diversity(self, pool_weights, mask=None):
        """
        pool_weights: [B, T, H, W]
        """
        pw = pool_weights
        if pw.size(1) < 2:
            return pw.sum() * 0.0
        if mask is not None:
            pw = pw.masked_fill(mask.unsqueeze(1), 0.0)

        pw = pw.flatten(2)  # [B,T,HW]
        pw = F.normalize(pw, dim=-1)

        sim = torch.matmul(pw, pw.transpose(1, 2))  # [B,T,T]
        eye = torch.eye(sim.size(-1), device=sim.device, dtype=torch.bool).unsqueeze(0)
        loss = sim.masked_select(~eye).mean()
        return loss

    def _loss_token_diversity(self, tokens):
        """
        tokens: [B, T, C]
        """
        tok = F.normalize(tokens, dim=-1)
        sim = torch.matmul(tok, tok.transpose(1, 2))  # [B,T,T]
        eye = torch.eye(sim.size(-1), device=sim.device, dtype=torch.bool).unsqueeze(0)
        loss = sim.masked_select(~eye).mean()
        return loss

    def _loss_gamma_coverage(self, pool_weights, gamma_i, mask=None):
        """
        pool_weights: [B, T, H, W]
        gamma_i:      [B, 1, H, W] or [B, H, W]
        """
        if gamma_i.dim() == 3:
            gamma_i = gamma_i.unsqueeze(1)

        pw = pool_weights
        if mask is not None:
            pw = pw.masked_fill(mask.unsqueeze(1), 0.0)
            gamma_i = gamma_i.masked_fill(mask.unsqueeze(1), 0.0)

        # 联合覆盖，而不是简单sum
        cover = 1.0 - torch.prod(1.0 - pw.clamp(0.0, 1.0), dim=1, keepdim=True)  # [B,1,H,W]

        loss = F.l1_loss(cover, gamma_i)
        return loss

    def _loss_token_entropy_roles(self, pool_weights, target_entropy):
        """
        pool_weights: [B, T, H, W]
        target_entropy: [T]
        """
        pw = pool_weights.flatten(2).clamp_min(1e-8)
        entropy = -(pw * pw.log()).sum(dim=-1)  # [B,T]
        target = target_entropy.view(1, -1).to(entropy.device)
        return F.l1_loss(entropy, target.expand_as(entropy))

def build(config):
    scem_enable = config.get("ENABLE", True)
    scem_gt = config.get("USE_GT")
    if scem_gt:
        scem_enable = False
    if not scem_enable:
        return None
    return SCEM(config)
