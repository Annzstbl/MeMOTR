"""
从 checkpoint 读取 SCEM 光谱字典（prototype 向量），绘制 8 通道光谱曲线。

SPECTRAL_TYPE=pi 时字典为 MixBGFG.pi_head.spec_pi.spectral_db_logits，形状 [K, 8]。
横轴：SPECTRAL_BAND_CENTERS_NM；纵轴：logit（及与推理一致的 per-prototype 归一化曲线）。

示例：
conda activate hsmot
cd /data1/users/litianhao01/hsmot/MeMOTR
python debug/20260526/scem_proto_curves.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.utils import load_checkpoint, get_model
from utils.utils import load_yaml_with_inheritance

_test4_path = os.path.join(CURRENT_DIR, "test4.py")
_spec = importlib.util.spec_from_file_location("debug_test4", _test4_path)
_test4 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_test4)

_resolve_train_config_path = _test4._resolve_train_config_path
_resolve_checkpoint_path = _test4._resolve_checkpoint_path
_resolve_config_root = _test4._resolve_config_root
SPECTRAL_BAND_CENTERS_NM = _test4.SPECTRAL_BAND_CENTERS_NM


def _normalize_prototype_rows(logits: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """与 SpectralPi / SpectralGraphDictionaryPrior 一致：逐 prototype 去均值 + L2 归一化。"""
    x = logits.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def extract_spectral_prototypes(model: torch.nn.Module) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    inner = get_model(model)
    scem = getattr(inner, "scem_module", None)
    if scem is None:
        raise RuntimeError("Model has no scem_module.")

    posterior = scem.posterior
    pi_head = posterior.pi_head
    spectral_type = str(getattr(pi_head, "spectral_type", "pi"))

    meta: Dict[str, Any] = {
        "spectral_type": spectral_type,
        "num_bands": len(SPECTRAL_BAND_CENTERS_NM),
        "band_centers_nm": SPECTRAL_BAND_CENTERS_NM,
    }

    if spectral_type == "pi":
        W = pi_head.spec_pi.spectral_db_logits.detach().cpu().numpy()
        meta["source"] = "pi_head.spec_pi.spectral_db_logits"
    elif spectral_type == "manifold":
        W = pi_head.spec_manifold.prototypes.detach().cpu().numpy()
        meta["source"] = "spec_manifold.prototypes"
    elif spectral_type == "graph_dictionary":
        W = pi_head.spec_graph_dictionary.prototypes.detach().cpu().numpy()
        meta["source"] = "spec_graph_dictionary.prototypes"
    else:
        raise ValueError(f"Unsupported spectral_type: {spectral_type}")

    meta["shape_kc"] = list(W.shape)
    return W, spectral_type, meta


def _plot_all_curves(
    W: np.ndarray,
    x_nm: np.ndarray,
    title: str,
    out_path: str,
    ylabel: str,
) -> None:
    k_num, _ = W.shape
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap("tab20", max(k_num, 1))
    for k in range(k_num):
        ax.plot(x_nm, W[k], color=cmap(k % 20), linewidth=1.2, alpha=0.85, label=f"p{k}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if k_num <= 16:
        ax.legend(ncol=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_heatmap(W: np.ndarray, x_nm: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, W.shape[0] * 0.15)))
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks(range(len(x_nm)))
    ax.set_xticklabels([f"{v:.0f}" for v in x_nm], rotation=45, ha="right")
    ax.set_yticks(range(W.shape[0]))
    ax.set_yticklabels([f"{i}" for i in range(W.shape[0])])
    ax.set_xlabel("Band center (nm)")
    ax.set_ylabel("Prototype id")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_grid_subplots(W: np.ndarray, x_nm: np.ndarray, title: str, out_path: str) -> None:
    k_num = W.shape[0]
    cols = 8
    rows = int(np.ceil(k_num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.8), sharex=True)
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        if k < k_num:
            ax.plot(x_nm, W[k], "C0", linewidth=1.0)
            ax.set_title(f"{k}", fontsize=8)
        ax.tick_params(labelsize=6)
        if r == rows - 1:
            ax.set_xlabel("nm", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_prototype_curve_figures(
    W: np.ndarray,
    out_dir: str,
    spectral_type: str,
    meta: Dict[str, Any],
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    x_nm = np.array(SPECTRAL_BAND_CENTERS_NM, dtype=np.float64)
    W_norm = _normalize_prototype_rows(W)

    np.save(os.path.join(out_dir, "spectral_dict_logits.npy"), W)
    np.save(os.path.join(out_dir, "spectral_dict_normalized.npy"), W_norm)

    paths = {
        "curves_raw": os.path.join(out_dir, f"prototype_curves_raw_{spectral_type}.png"),
        "curves_norm": os.path.join(out_dir, f"prototype_curves_norm_{spectral_type}.png"),
        "heatmap_raw": os.path.join(out_dir, f"prototype_heatmap_raw_{spectral_type}.png"),
        "heatmap_norm": os.path.join(out_dir, f"prototype_heatmap_norm_{spectral_type}.png"),
        "grid_raw": os.path.join(out_dir, f"prototype_grid_raw_{spectral_type}.png"),
        "grid_norm": os.path.join(out_dir, f"prototype_grid_norm_{spectral_type}.png"),
    }

    _plot_all_curves(
        W, x_nm,
        f"SCEM prototypes (raw logits, K={W.shape[0]}, {spectral_type})",
        paths["curves_raw"],
        "logit",
    )
    _plot_all_curves(
        W_norm, x_nm,
        f"SCEM prototypes (zero-mean L2 norm, K={W.shape[0]}, {spectral_type})",
        paths["curves_norm"],
        "normalized weight",
    )
    _plot_heatmap(W, x_nm, f"Prototype heatmap (raw, {spectral_type})", paths["heatmap_raw"])
    _plot_heatmap(W_norm, x_nm, f"Prototype heatmap (normalized, {spectral_type})", paths["heatmap_norm"])
    _plot_grid_subplots(W, x_nm, f"Per-prototype curves (raw, {spectral_type})", paths["grid_raw"])
    _plot_grid_subplots(W_norm, x_nm, f"Per-prototype curves (norm, {spectral_type})", paths["grid_norm"])

    meta_path = os.path.join(out_dir, "prototype_curves_meta.json")
    meta_out = {**meta, "files": paths}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    paths["meta"] = meta_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SCEM spectral prototype curves [K,8].")
    parser.add_argument("--train-config", type=str, default="20260511-2.yaml")
    parser.add_argument("--checkpoint", type=str, default="last.pth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_cfg_path = _resolve_train_config_path(args.train_config)
    train_config = load_yaml_with_inheritance(path=train_cfg_path)

    config_root = _resolve_config_root(train_cfg_path, args.output_dir)
    out_dir = os.path.join(config_root, "scem_proto", "prototype_curves")
    os.makedirs(out_dir, exist_ok=True)

    train_config["MEMOTR_VERSION"] = "20260511_figure"
    model = build_model(config=train_config)
    model.to(torch.device(args.device))
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    print(f"[Info] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)

    W, spectral_type, meta = extract_spectral_prototypes(model)
    print(f"[Info] prototypes shape={W.shape}, type={spectral_type}, source={meta['source']}")

    paths = save_prototype_curve_figures(W, out_dir, spectral_type, meta)
    for k, p in paths.items():
        print(f"[Info] saved {k}: {p}")


if __name__ == "__main__":
    main()
