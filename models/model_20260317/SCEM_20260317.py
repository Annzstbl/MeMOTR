import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_20260310.SCEM_20260310 import ASPP as ASPP20260310
from ..model_20260310.SCEM_20260310 import BGHead as BGHead20260310
from ..model_20260310.SCEM_20260310 import CenterMaskedConv3x3 as CenterMaskedConv3x3Base20260310
from ..model_20260310.SCEM_20260310 import CoordConv as CoordConv20260310
from ..model_20260310.SCEM_20260310 import GDN
from ..model_20260310.SCEM_20260310 import GN
from ..model_20260310.SCEM_20260310 import MixBGFG as MixBGFG20260310
from ..model_20260310.SCEM_20260310 import PIHead as PIHead20260310
from ..model_20260310.SCEM_20260310 import SCEM as SCEM20260310
from ..model_20260310.SCEM_20260310 import SpectralGraphDictionaryPrior as SpectralGraphDictionaryPrior20260310
from ..model_20260310.SCEM_20260310 import SpectralManifold as SpectralManifold20260310
from ..model_20260310.SCEM_20260310 import SpectralPi as SpectralPi20260310
from ..model_20260310.SCEM_20260310 import _zero_invalid
from typing_extensions import override

class CoordConv20260317(CoordConv20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CenterMaskedConv3x3_20260317(CenterMaskedConv3x3Base20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ASPP20260317(ASPP20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpectralManifold20260317(SpectralManifold20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpectralPi20260317(SpectralPi20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpectralGraphDictionaryPrior20260317(SpectralGraphDictionaryPrior20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PIHead20260317(PIHead20260310):
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
        self.enhance_alpha = nn.Parameter(torch.tensor(1.0))

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
        w = torch.softmax(self.space_to_spec_gate(f_space), dim=1)
        e_k = w * f_spec# [B, ch, H, W]

        pi = torch.sigmoid(self.out(e_k))#[B, 1, H, W]

        return pi, e_k


class BGHead20260317(BGHead20260310):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MixBGFG20260317(MixBGFG20260310):
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

        self.bg_head = BGHead20260317(in_ch=ch, out_ch=C)
        self.pi_head = PIHead20260317(
            in_ch=ch,
            use_cache=use_cache,
            spectral_type=spectral_type,
            spectral_databse_num=spectral_databse_num,
        )

        self.prior_mode = prior_mode.lower()
        assert self.prior_mode == "gate"
        self.gate_head = nn.Sequential(
            nn.Conv2d(ch + 1, ch // 2, 3, 1, 1),
            GN(ch // 2),
            nn.SiLU(),
            nn.Conv2d(ch // 2, 1, 1, 1, 0),
        )

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


class SCEMFeatureFusion20260317(nn.Module):
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


class SCEM20260317(SCEM20260310):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.cfg = dict(config)
        self.depth = int(self.cfg.get("DEPTH", 1))
        self.width = float(self.cfg.get("WIDTH", 0.5))
        self.use_cache = bool(self.cfg.get("USE_CACHE", True))
        self.in_ch = int(self.cfg.get("IN_CHANNELS", 256))
        self.prior_mode = self.cfg.get("PRIOR_MODE")
        self.spectral_databse_num = int(self.cfg.get("SPECTRAL_DATABASE_NUM"))
        self.spectral_type = self.cfg.get("SPECTRAL_TYPE").lower()

        self.posterior = MixBGFG20260317(
            C=self.in_ch,
            depth=self.depth,
            width=self.width,
            use_cache=self.use_cache,
            prior_mode=self.prior_mode,
            spectral_databse_num=self.spectral_databse_num,
            spectral_type=self.spectral_type,
        )

        self.feature_fusion = SCEMFeatureFusion20260317(
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
                nn.Conv2d(self.spectral_databse_num, self.spectral_databse_num, 3, padding=1, bias=False),
                GN(self.spectral_databse_num),
                nn.SiLU(inplace=False),
                nn.Conv2d(self.spectral_databse_num, self.spectral_databse_num, 1, bias=True),
            )
            for _ in range(3)
        ])


        # 初始化gamma_resize_heads的conv，使其输出全0
        with torch.no_grad():
            for i in range(3):
                nn.init.zeros_(self.gamma_resize_heads[i][0].weight)
                nn.init.zeros_(self.gamma_resize_heads[i][0].bias)

                nn.init.zeros_(self.spectral_evidence_resize_heads[i][-1].weight)
                nn.init.zeros_(self.spectral_evidence_resize_heads[i][-1].bias)


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
    def forward(self, features, masks, specs, prior_map=None):
        """
        features: List[Tensor]   each [B, C_i, H_i, W_i]
        masks:    List[Bool]     each [B, H_i, W_i]  True=pad
        specs:    List[Tensor]   each [B, 8, H, W]

        Returns:

            返回所有的src和新筛选出来的tokne
        """
        assert isinstance(features, (list, tuple)) and len(features) > 0
        self._check_masks(masks)

        feat, mask, spec  = self.feature_fusion(features, masks, specs)

        # feat0 = features[0]
        # mask0 = masks[0]  # [B,H0,W0] bool
        # 计算概率图
        # return out: log_mix gamma pi valid_mask spectral_evidence spectral_dict
        out = self.posterior(feat, pad_mask=mask, prior_map=prior_map, spec=spec) 
        
        # 特征增强
        gamma = out["gamma"]
        feat_enhanced = []
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
        top_idx_list = []
        assign_list = []

        spectral_dict = out["spectral_dict"]#[K, 8]
        token_nums = [8, 4, 2, 1]
        for i, (feat, token_num) in enumerate(zip(features, token_nums)):
            tokens, top_idx, assign = self.build_evidence_tokens(feat, spectral_evidence_multilevel[i], top_m=token_num, mask=masks[i])
            spectral_part = spectral_dict[top_idx] #[K, 8] + [B, token_num] 高级索引->[B, token_num, 8]
            evidence_tokens_spectral_part.append(spectral_part)
            evidence_tokens.append(tokens)
            top_idx_list.append(top_idx)
            assign_list.append(assign)
        
        # 构造global token
        global_token = []
        global_token_spectral_part = []
        for i, (feat, spec) in enumerate(zip(features, specs)):
            global_token_i = feat.mean(dim=(2, 3))
            global_token.append(global_token_i)
            global_token_spectral_part_i = spec.mean(dim=(2, 3))
            global_token_spectral_part.append(global_token_spectral_part_i)


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
        
        return (feat_enhanced, specs), (evidence_tokens, evidence_tokens_spectral_part), (global_token, global_token_spectral_part), (gamma, out['log_mix'], spectral_dict)
        




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



CoordConv = CoordConv20260317
CenterMaskedConv3x3 = CenterMaskedConv3x3_20260317
ASPP = ASPP20260317
SpectralManifold = SpectralManifold20260317
SpectralPi = SpectralPi20260317
SpectralGraphDictionaryPrior = SpectralGraphDictionaryPrior20260317
PIHead = PIHead20260317
BGHead = BGHead20260317
MixBGFG = MixBGFG20260317
SCEMFeatureFusion = SCEMFeatureFusion20260317
SCEM = SCEM20260317


def build(config):
    scem_enable = config.get("ENABLE", True)
    scem_gt = config.get("USE_GT")
    if scem_gt:
        scem_enable = False
    if not scem_enable:
        return None
    return SCEM20260317(config)
