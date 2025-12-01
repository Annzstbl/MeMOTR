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

# =========================
# CoordConv with cache & dynamic size
# =========================

class CoordConv(nn.Module):
    """
    CoordConv with:
      - dynamic resolution support (default)
      - small LRU cache for (H,W,device)
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_cache=True, cache_size=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + 2, out_ch, k, s, p, bias=False)
        self.bn   = GN(out_ch)
        self.act  = nn.SiLU(inplace=False)
        self.use_cache  = use_cache
        self.cache_size = cache_size
        # 简易 LRU：dict + key 队列
        self._grid_cache = {}
        self._grid_keys  = []

    def _cache_put(self, key, grid):
        if not self.use_cache:
            return
        if key in self._grid_cache:
            return
        self._grid_cache[key] = grid
        self._grid_keys.append(key)
        if len(self._grid_keys) > self.cache_size:
            old = self._grid_keys.pop(0)
            self._grid_cache.pop(old, None)

    def _cache_get(self, key):
        if not self.use_cache:
            return None
        g = self._grid_cache.get(key, None)
        if g is not None:
            # 更新 LRU
            self._grid_keys.remove(key)
            self._grid_keys.append(key)
        return g

    def _get_grid(self, B, H, W, device, dtype=torch.float32):
        key = (H, W, device.index if hasattr(device, "index") else -1, dtype)
        g = self._cache_get(key)
        if g is None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing='ij'
            )
            g = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]
            self._cache_put(key, g)
        return g.expand(B, -1, -1, -1)

    def forward(self, x):
        B, _, H, W = x.shape
        cc = self._get_grid(B, H, W, x.device, x.dtype)
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
        # w = self.conv.weight * self.mask
        # out = F.conv2d(x, w, bias=None, stride=1, padding=1, groups=self.conv.groups)
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


class PIHead(nn.Module):
    def __init__(self, in_ch, ch=128, use_cache=True):
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

    def forward(self, x):
        a = self.coord(x)
        a = self.mask1(a)
        a = self.aspp(a)
        g = self.global_fc(self.global_pool(a))
        g = g.expand_as(a).contiguous()
        h = self.local_fuse(torch.cat([a, g], dim=1))
        pi = torch.sigmoid(self.out(h))
        return pi


# =========================
# Likelihoods
# =========================


def log_gaussian_interval(z, mu, raw_log_sigma, width: float = 1.0, eps: float = 1e-8):
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


def log_student_t_interval(z, mu, raw_log_scale, raw_nu=None,
                           width: float = 1.0, eps: float = 1e-8):
    """
    与上类似，但通道独立 Student-t:
      T_ν(loc=μ_c, scale=σ_c)
    """
    log_scale = _log_sigma_from_raw(raw_log_scale)
    scale = torch.exp(log_scale)      # [B,C,H,W]

    if raw_nu is not None:
        nu = F.softplus(raw_nu) + 2.0   # 标量或 broadcast
    else:
        nu = z.new_tensor(3.0)
    # StudentT 支持 broadcast
    dist = StudentT(df=nu, loc=mu, scale=scale)

    half = width * 0.5
    upper = dist.cdf(z + half)
    lower = dist.cdf(z - half)
    p = (upper - lower).clamp_min(eps)   # [B,C,H,W]

    log_p = torch.log(p)
    return log_p.sum(dim=1, keepdim=True)   # [B,1,H,W]


# =========================
# Mixture posterior head
# =========================

class MixBGFG(nn.Module):
    def __init__(self, C, depth=4, width=0.5, with_foreground=True, use_cache=True, tau_mode="mean", prior_mode=None):
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

        # likelihood heads
        self.mu_b_head     = nn.Conv2d(ch, C, 3, 1, 1)
        self.logsig_b_head = nn.Conv2d(ch, C, 3, 1, 1)

        self.with_foreground = with_foreground
        if self.with_foreground:
            self.mu_f_head       = nn.Conv2d(ch, C, 3, 1, 1)
            self.logsig_f_head = nn.Conv2d(ch, C, 3, 1, 1)
            # self.logscale_f_head = nn.Conv2d(ch, C, 3, 1, 1)
            # self.raw_nu          = nn.Parameter(torch.tensor(1.0))  # learnable ν

        self.pi_head = PIHead(in_ch=ch, use_cache=use_cache)
        self.prior_mode = prior_mode.lower()
        if self.prior_mode == "gate":
            self.gate_head = nn.Sequential(nn.Conv2d(ch+1, ch//2, 3,1,1), GN(ch//2), nn.SiLU(),
                                nn.Conv2d(ch//2, 1, 1,1,0))

        # 初始化方差头 bias 使 sigma≈1
        with torch.no_grad():
            nn.init.zeros_(self.logsig_b_head.weight)
            self.logsig_b_head.bias.fill_(1.313)
            if self.with_foreground:
                nn.init.zeros_(self.logsig_f_head.weight)
                self.logsig_f_head.bias.fill_(1.313)

    def forward(self, Z, valid_mask=None, interval_width: float = 1.0, prior_map = None):
        """
        Z: [B,C,H,W]
        valid_mask: [B,H,W] or [B,1,H,W], True=valid
        interval_width: 区间 Δ，默认为 1.0 → [x-0.5, x+0.5]
        """
        if valid_mask is not None and valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)  # [B,1,H,W]

        # x = self.trunk(self.stem(Z))
        x = self.stem(Z)

        # 背景区间 log prob
        mu_b     = self.mu_b_head(x)
        log_sigb = self.logsig_b_head(x)
        log_pb   = self.tau * log_gaussian_interval(
            Z, mu_b, log_sigb, width=interval_width
        )   # [B,1,H,W] <= 0

        # 前景区间 log prob
        if self.with_foreground:
            mu_f       = self.mu_f_head(x)
            log_sigf = self.logsig_f_head(x)
            log_pf = self.tau * log_gaussian_interval(
                Z, mu_f, log_sigf, width=interval_width
            )
            # log_pf     = self.tau * log_student_t_interval(
            #     Z, mu_f, log_scalef, self.raw_nu, width=interval_width
            # )   # [B,1,H,W] <= 0
        else:
            log_pf = torch.zeros_like(log_pb)

        # π 先验
        pi_net = self.pi_head(x).clamp(1e-6, 1 - 1e-6)   # [B,1,H,W]
        if self.prior_mode is not None:
            logit_net = torch.logit(pi_net)
            prior_map = torch.clamp(prior_map, 1e-6, 1 - 1e-6)
            logit_prior = torch.logit(prior_map.unsqueeze(1))
            logit_prior_downscale = F.interpolate(logit_prior, size=pi_net.shape[-2:], mode="bilinear", align_corners=False)
            g = torch.sigmoid(self.gate_head(torch.cat([x, logit_prior_downscale], dim=1)))
            l = (1-g) * logit_net + g * logit_prior_downscale
            # TODO 是否需要超参数tau
            # pi = torch.sigmoid(l / max(self.tau, 1e-6))
            pi = torch.sigmoid(l)
        else:
            pi = pi_net


        # 混合区间概率的 loglik
        a = torch.log1p(-pi) + log_pb        # log((1-π)·P_b)
        b = torch.log(pi)    + log_pf        # log(π·P_f)
        log_mix = torch.logaddexp(a, b)      # log(P_mix)，仍然 ≤ 0
        gamma   = torch.exp(b - log_mix)     # 后验 P(F=1 | z)

        if valid_mask is not None:
            vmf = (~valid_mask).float()         # True=not valid
            log_mix = log_mix * vmf
            gamma   = gamma   * vmf
            pi      = pi      * vmf

        return {
            "log_mix": log_mix,   # 区间混合概率的 log，≤0
            "gamma":   gamma,     # 前景后验
            "pi":      pi,
            "valid_mask": valid_mask,
        }

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
        self.with_foreground  = bool(self.cfg.get("WITH_FOREGROUND", False))
        self.depth      = int(self.cfg.get("DEPTH", 1))
        self.width      = float(self.cfg.get("WIDTH", 0.5))
        self.use_cache  = bool(self.cfg.get("USE_CACHE", True))
        self.in_ch      = int(self.cfg.get("IN_CHANNELS", 256))
        self.prior_mode = self.cfg.get("PRIOR_MODE", None)
        # self.lazy_built = False
        self.posterior = MixBGFG(
            C=self.in_ch,
            depth=self.depth,
            width=self.width,
            with_foreground=self.with_foreground,
            use_cache=self.use_cache,
            prior_mode=self.prior_mode
        )

        #打印所有参数及其大小
        for name, param in self.named_parameters():
            print(f"SCEM param: {name}, size: {param.size()}")


    @torch.no_grad()
    def _check_masks(self, masks):
        assert isinstance(masks, (list, tuple)) and len(masks) > 0
        for m in masks:
            assert m.dtype == torch.bool and m.dim() == 3, "mask should be [B,H,W] bool"

    def forward(self, features, masks, prior_map=None):
        """
        features: List[Tensor]   each [B, C_i, H_i, W_i]
        masks:    List[Bool]     each [B, H_i, W_i]  True=valid

        Returns:
          features_passthrough,
          masks_passthrough,
          scem_out      -> posterior gamma [B,1,H0,W0]
          aux           -> dict for losses: {log_mix, pi, valid_mask}
        """
        assert isinstance(features, (list, tuple)) and len(features) > 0
        self._check_masks(masks)

        feat0 = features[0]
        mask0 = masks[0]  # [B,H0,W0] bool


        out = self.posterior(feat0, valid_mask=mask0, prior_map=prior_map)
        # gamma = out["gamma"]              # [B,1,H0,W0]
        # aux   = {"log_mix": out["log_mix"], "pi": out["pi"], "valid_mask": out["valid_mask"]}

        return out["gamma"], out["log_mix"]
        # return features


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

        return F.max_pool2d(posterior, kernel_size=kernel_size, stride=stride)
        # return F.interpolate(posterior, size=feat.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def apply_posterior_enhance(features, masks, scem_out, alpha: float = 0.5):
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
            # gi = SCEM._resize_posterior_to(xi, scem_out).clamp_(0.0, 1.0)   # [B,1,Hi,Wi]
            # gi = SCEM._resize_posterior_to(xi, scem_out.detach()).clamp(0.0, 1.0)
            gi = SCEM._resize_posterior_to(xi, scem_out).clamp(0.0, 1.0)
            # 2) mask 无效处置零
            if mi.dim() == 3:
                gi = gi * (~mi).unsqueeze(1).float()
            # 3) 门控增强
            xi_enh = xi * (1.0 + alpha * gi)
            enhanced.append(xi_enh)
        return enhanced

# =========================
# Losses
# =========================

def nll_loss_from_aux(aux):
    log_mix = aux["log_mix"]                  # [B,1,H,W]
    valid   = aux["valid_mask"].bool()        # [B,1,H,W]
    if valid.sum() == 0:
        return log_mix.new_zeros(())
    return (-log_mix[valid]).mean() 