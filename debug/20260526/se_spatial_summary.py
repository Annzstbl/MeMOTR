"""
从 test3.py 输出的逐像素 SE npy，去掉空间维度 (H, W)，得到 8 通道标量统计。

输入（由 --analyze-se 生成）：
  se_all_mean.npy  [8, H, W]  每像素、跨帧均值
  se_all_var.npy   [8, H, W]  每像素、跨帧方差
  se_all_count.npy [H, W]     每像素累计帧数

输出：
  se_all_channel_summary.json / .npy
  se_all_channel_profile.png  （可选）

用法（hsmot 环境）：
  conda activate hsmot
  python debug/20260526/se_spatial_summary.py \\
    --stats-dir debug/20260526/20260511-2/se_stats \\
    --prefix se_all \\
    --save-vis
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

SPECTRAL_BAND_CENTERS_NM = [422.5, 487.5, 550.0, 602.5, 660.0, 725.0, 785.0, 887.2]
BAND_COLORS = [
    "#4B3BFF", "#0072FF", "#1FA64A", "#FFA31A",
    "#D7191C", "#FF6B4A", "#C5161D", "#6E0000",
]


def load_spatial_stats(stats_dir: str, prefix: str) -> Dict[str, np.ndarray]:
    mean_path = os.path.join(stats_dir, f"{prefix}_mean.npy")
    var_path = os.path.join(stats_dir, f"{prefix}_var.npy")
    count_path = os.path.join(stats_dir, f"{prefix}_count.npy")
    for path in (mean_path, var_path, count_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing: {path}")
    return {
        "mean": np.load(mean_path),
        "var": np.load(var_path),
        "count": np.load(count_path),
    }


def summarize_se_spatial(
    stats: Dict[str, np.ndarray],
    min_count: int = 1,
    weighted: bool = True,
) -> Dict[str, Any]:
    """
    对 [8, H, W] 逐像素统计做空间聚合，得到 [8] 通道级标量。

    - channel_mean: 空间平均后的 SE 均值（主曲线）
    - channel_var_temporal: 各像素时间方差的空间平均（时间波动）
    - channel_var_spatial: 各像素时间均值的空间方差（空间不均匀性）
    - channel_var_total: 上两者之和（全方差分解）
    """
    mean_map = stats["mean"]
    var_map = stats["var"]
    count_map = stats["count"]

    if mean_map.ndim != 3 or var_map.shape != mean_map.shape:
        raise ValueError(f"Expected mean/var shape [8,H,W], got {mean_map.shape}, {var_map.shape}")
    n_ch = mean_map.shape[0]
    valid = count_map >= min_count
    if not np.any(valid):
        raise ValueError(f"No pixel with count >= {min_count}")

    mu = mean_map[:, valid].astype(np.float64)
    var_pix = var_map[:, valid].astype(np.float64)

    if weighted:
        w = count_map[valid].astype(np.float64)
        w = w / w.sum()
    else:
        w = np.full(int(valid.sum()), 1.0 / valid.sum(), dtype=np.float64)

    channel_mean = (mu * w[None]).sum(axis=1)
    channel_var_temporal = (var_pix * w[None]).sum(axis=1)
    channel_var_spatial = (w[None] * (mu - channel_mean[:, None]) ** 2).sum(axis=1)
    channel_var_total = channel_var_temporal + channel_var_spatial

    return {
        "n_channels": n_ch,
        "n_valid_pixels": int(valid.sum()),
        "min_count": min_count,
        "weighted_by_frame_count": weighted,
        "band_centers_nm": SPECTRAL_BAND_CENTERS_NM,
        "channel_mean": channel_mean.astype(np.float32),
        "channel_var_temporal": channel_var_temporal.astype(np.float32),
        "channel_var_spatial": channel_var_spatial.astype(np.float32),
        "channel_var_total": channel_var_total.astype(np.float32),
        "channel_std_temporal": np.sqrt(np.maximum(channel_var_temporal, 0)).astype(np.float32),
        "channel_std_spatial": np.sqrt(np.maximum(channel_var_spatial, 0)).astype(np.float32),
        "channel_std_total": np.sqrt(np.maximum(channel_var_total, 0)).astype(np.float32),
    }


def save_summary(out_dir: str, prefix: str, summary: Dict[str, Any]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{prefix}_channel_summary.json")
    npy_path = os.path.join(out_dir, f"{prefix}_channel_summary.npy")

    json_payload = {k: v for k, v in summary.items() if not isinstance(v, np.ndarray)}
    for key in (
        "channel_mean", "channel_var_temporal", "channel_var_spatial", "channel_var_total",
        "channel_std_temporal", "channel_std_spatial", "channel_std_total",
    ):
        json_payload[key] = summary[key].tolist()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    np.save(
        npy_path,
        {
            "mean": summary["channel_mean"],
            "var_temporal": summary["channel_var_temporal"],
            "var_spatial": summary["channel_var_spatial"],
            "var_total": summary["channel_var_total"],
            "std_total": summary["channel_std_total"],
            "wavelengths_nm": np.array(SPECTRAL_BAND_CENTERS_NM, dtype=np.float32),
        },
    )
    return {"json": json_path, "npy": npy_path}


def save_profile_figure(summary: Dict[str, Any], out_dir: str, prefix: str) -> str:
    mean = summary["channel_mean"]
    std = summary["channel_std_total"]
    x = np.arange(summary["n_channels"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax0 = axes[0]
    ax0.plot(x, mean, linewidth=2.2, marker="o", color="#333333")
    ax0.fill_between(x, mean - std, mean + std, alpha=0.25, color="#666666")
    for i in range(len(x)):
        ax0.scatter(i, mean[i], s=70, color=BAND_COLORS[i], edgecolors="black", linewidths=0.6, zorder=3)
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{w:.1f}" for w in SPECTRAL_BAND_CENTERS_NM], fontsize=9)
    ax0.set_xlabel("Wavelength (nm)")
    ax0.set_ylabel("Spatially aggregated SE mean")
    ax0.set_title("Channel mean ± total std (spatial avg)")
    ax0.grid(False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = axes[1]
    bars_t = ax1.bar(x - 0.15, summary["channel_var_temporal"], width=0.3, label="temporal")
    bars_s = ax1.bar(x + 0.15, summary["channel_var_spatial"], width=0.3, label="spatial")
    for i, b in enumerate(bars_t):
        b.set_color(BAND_COLORS[i])
        b.set_alpha(0.55)
        b.set_edgecolor("black")
        b.set_linewidth(0.4)
    for i, b in enumerate(bars_s):
        b.set_color(BAND_COLORS[i])
        b.set_edgecolor("black")
        b.set_linewidth(0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{w:.1f}" for w in SPECTRAL_BAND_CENTERS_NM], fontsize=9)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Variance")
    ax1.set_title("Variance decomposition")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_channel_profile.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-pixel SE npy to 8-channel scalars.")
    parser.add_argument("--stats-dir", type=str, required=True, help="Directory containing se_*_mean.npy etc.")
    parser.add_argument("--prefix", type=str, default="se_all", help="File prefix, e.g. se_all or se")
    parser.add_argument("--min-count", type=int, default=1, help="Ignore pixels with fewer frame counts.")
    parser.add_argument("--unweighted", action="store_true", help="Use uniform spatial average instead of count-weighted.")
    parser.add_argument("--save-vis", action="store_true", help="Save channel profile figure.")
    args = parser.parse_args()

    stats_dir = os.path.abspath(args.stats_dir)
    stats = load_spatial_stats(stats_dir, args.prefix)
    summary = summarize_se_spatial(stats, min_count=args.min_count, weighted=not args.unweighted)
    paths = save_summary(stats_dir, args.prefix, summary)

    print(f"[Done] channel summary json: {paths['json']}")
    print(f"[Done] channel summary npy:  {paths['npy']}")
    print("\nchannel_mean:", np.round(summary["channel_mean"], 4))
    print("channel_var_total:", np.round(summary["channel_var_total"], 4))

    if args.save_vis:
        fig_path = save_profile_figure(summary, stats_dir, args.prefix)
        print(f"[Done] figure: {fig_path}")


if __name__ == "__main__":
    main()
