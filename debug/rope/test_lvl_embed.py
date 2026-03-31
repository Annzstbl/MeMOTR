
import os
import re
import torch
import matplotlib.pyplot as plt

def load_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj:  # lightning / custom
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):  # some trainers
            return obj["model"]
        return obj  # maybe already a state_dict
    raise ValueError("Unsupported checkpoint format")

def find_lvl_embed_key(state_dict: dict):
    """
    尝试匹配常见命名：lvl_embed / level_embed / level_embedding / lvl_embedding 等
    形状通常是 [L, D] 或 [1, L, D] 或 [L, D, 1]（极少见）
    """
    patterns = [
        r"\blvl[_\.]?emb(ed|edding)?\b",
        r"\blevel[_\.]?emb(ed|edding)?\b",
        r"\blevels?[_\.]?emb(ed|edding)?\b",
    ]
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        name = k.lower()
        if any(re.search(p, name) for p in patterns):
            # 优先考虑 2D/3D 且维度里有“L×D”的张量
            if v.ndim in (2, 3):
                candidates.append((k, tuple(v.shape)))
    if not candidates:
        raise KeyError("找不到 lvl_embed / level_embed 相关权重。请检查键名。")
    # 简单启发式：优先选维度包含 L×D 的（例如 [L,D] 或 [1,L,D]）
    candidates.sort(key=lambda x: (-(2 in x[1] or 3 in x[1]), x[0]))
    print("候选 lvl_embed keys:")
    for k, shp in candidates:
        print(f"  {k}: {shp}")
    return candidates[0][0]

def to_LD(t: torch.Tensor):
    """
    统一到 [L, D] 形状
    """
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        # 常见: [1, L, D] 或 [L, D, 1]
        if t.shape[0] == 1:
            return t[0]
        if t.shape[-1] == 1:
            return t[..., 0]
    raise ValueError(f"无法识别的 lvl_embed 形状: {tuple(t.shape)}。请手动 reshape。")

def cosine_sim_matrix(E: torch.Tensor):
    """
    E: [L, D]  ->  S: [L, L]  (cosine similarity)
    """
    E = E.float()
    E = torch.nn.functional.normalize(E, dim=1)  # 行向量单位化
    return E @ E.T

def analyze_and_print(S: torch.Tensor):
    L = S.size(0)
    # 自身相似度置 NaN 便于统计
    S_no_diag = S.clone()
    S_no_diag.fill_diagonal_(float('nan'))

    # 每个 level 的最相似“其他 level”
    nn_idx = torch.nanargmax(S_no_diag, dim=1)
    nn_val = S_no_diag[torch.arange(L), nn_idx]

    print("\n=== 基本统计 ===")
    print(f"L (levels) = {L}")
    print(f"对角均值(应≈1): {torch.diag(S).mean().item():.6f}")
    off = S_no_diag[~torch.isnan(S_no_diag)]
    print(f"非对角均值: {off.mean().item():.6f}, 中位数: {off.median().item():.6f}, "
          f"最小: {off.min().item():.6f}, 最大: {off.max().item():.6f}")

    print("\n=== 每个 level 的最近邻(排除自身) ===")
    for i in range(L):
        print(f"lvl {i:3d} -> {int(nn_idx[i]):3d} | cos={float(nn_val[i]): .6f}")

def plot_heatmap(S: torch.Tensor, title="Cosine similarity of lvl_embed", save_path=None):
    plt.figure(figsize=(6,5))
    plt.imshow(S.cpu().numpy(), aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("level")
    plt.ylabel("level")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"相似度热力图已保存: {save_path}")
    else:
        plt.show()

def check_lvl_cosine_from_ckpt(ckpt_path: str, heatmap_path: str | None = None):
    sd = load_state_dict(ckpt_path)
    key = find_lvl_embed_key(sd)
    E = to_LD(sd[key])              # [L, D]
    S = cosine_sim_matrix(E)        # [L, L]
    print(f"\n使用的键: {key}, 形状: {tuple(E.shape)}")
    analyze_and_print(S)
    if heatmap_path is None:
        # 默认与 ckpt 同目录
        base, _ = os.path.splitext(ckpt_path)
        heatmap_path = base + "_lvl_cosine.png"
    plot_heatmap(S, save_path=heatmap_path)

# ----------------- 示例 -----------------
check_lvl_cosine_from_ckpt("/data4/litianhao/hsmot/memotr/spectralemb/12_2_global_resumeQueryUpdater/checkpoint_24.pth")
