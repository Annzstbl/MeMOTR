import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# =========================
# 1. 路径与基本配置
# =========================
# json_path = "/data1/users/litianhao01/hsmot/MeMOTR/debug/20260526/20260511-2/track/data31-1/id0/data31-1__trackId0__timeline_meta.json"
json_path = "/data1/users/litianhao01/hsmot/MeMOTR/debug/20260526/20260511-2/track/data33-1/id4/data33-1__trackId4__timeline_meta.json"

save_path = "/data1/users/litianhao01/hsmot/MeMOTR/debug/20260526/20260511-2/track/data33-1/id4_spectral"
os.makedirs(save_path, exist_ok=True)
save_fig1 = os.path.join(save_path, "fig1_band_identity.png")
save_fig2 = os.path.join(save_path, "fig2_mean_profile.png")
save_fig3 = os.path.join(save_path, "fig3_temporal_spectral_map.png")
save_fig4 = os.path.join(save_path, "fig4_band_variability.png")

# 8个谱段中心波长
wavelengths = np.array([422.5, 487.5, 550.0, 602.5, 660.0, 725.0, 785.0, 887.2])
band_labels_nm = [f"{w:.1f}" for w in wavelengths]
band_ids = np.arange(8)

# 谱段身份颜色
# 前5个是可见光近似颜色；后3个NIR统一采用红色相关伪彩
band_colors = [
    "#4B3BFF",  # 422.5 nm: violet-blue
    "#0072FF",  # 487.5 nm: blue
    "#1FA64A",  # 550.0 nm: green
    "#FFA31A",  # 602.5 nm: orange
    "#D7191C",  # 660.0 nm: visible red
    "#FF6B4A",  # 725.0 nm: NIR pseudo-color
    "#C5161D",  # 785.0 nm: NIR pseudo-color
    "#6E0000",  # 887.2 nm: NIR pseudo-color
]

# 如果json中有多个目标，可以筛选；不筛选就设为None
target_global_q_idx = None
target_track_local_idx = 4

# 归一化方式：
# "sigmoid" -> 逐元素 sigmoid，映射到 (0, 1)
# "none"    -> 不归一化
norm_mode = "sigmoid"

# 阴影放大系数（用于 mean ± shade_scale * std）
shade_scale = 1.8

# sigmoid 响应显示范围
response_ylim = (0.2, 0.8)


# =========================
# 2. 读取数据并提取 pooled_spectral[0]
# =========================
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = []
for item in data:
    if target_global_q_idx is not None and item.get("global_q_idx") != target_global_q_idx:
        continue
    if target_track_local_idx is not None and item.get("track_local_idx") != target_track_local_idx:
        continue
    filtered.append(item)

records = []
for item in filtered:
    pooled = item.get("pooled_spectral", None)
    if pooled is None or not isinstance(pooled, list) or len(pooled) == 0:
        continue

    first_spec = pooled[0]  # 只取第一个元素
    if not isinstance(first_spec, list) or len(first_spec) != 8:
        continue

    frame_id = item.get("frame", len(records))
    records.append((frame_id, np.array(first_spec, dtype=float)))

if len(records) == 0:
    raise ValueError("没有找到可用的 pooled_spectral[0] 数据，请检查筛选条件。")

records.sort(key=lambda x: x[0])

frames = np.array([r[0] for r in records], dtype=int)
X_raw = np.stack([r[1] for r in records], axis=0)   # [T, 8]


# =========================
# 3. 归一化
# =========================
def sigmoid(X):
    X = np.asarray(X, dtype=float)
    return np.where(
        X >= 0,
        1.0 / (1.0 + np.exp(-X)),
        np.exp(X) / (1.0 + np.exp(X)),
    )

if norm_mode == "none":
    X = X_raw.copy()
elif norm_mode == "sigmoid":
    X = sigmoid(X_raw)
else:
    raise ValueError(f"Unsupported norm_mode: {norm_mode}")

mean_spec = X.mean(axis=0)
std_spec = X.std(axis=0)
shade_spec = shade_scale * std_spec


# =========================
# 通用绘图参数
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Figure 1: 谱段身份条
# =========================
fig1, ax1 = plt.subplots(figsize=(8, 1.8))

strip = np.arange(8).reshape(1, -1)
cmap_strip = ListedColormap(band_colors)
ax1.imshow(strip, aspect="auto", cmap=cmap_strip, interpolation="nearest")

ax1.set_yticks([])
ax1.set_xticks(np.arange(8))
ax1.set_xticklabels([f"B{i}\n{w:.1f}" for i, w in enumerate(wavelengths)], fontsize=10)
ax1.set_xlabel("Spectral band / center wavelength (nm)", fontsize=11)

# 可见光与NIR分界（band 4 和 band 5 之间）
ax1.axvline(4.5, color="white", linestyle="--", linewidth=1.5)
ax1.text(2.2, -0.75, "Visible", ha="center", va="center", fontsize=10)
ax1.text(6.3, -0.75, "NIR pseudo-color", ha="center", va="center", fontsize=10)

for spine in ax1.spines.values():
    spine.set_visible(False)

ax1.set_title("Band identity", fontsize=12, pad=6)

plt.tight_layout()
plt.savefig(save_fig1, dpi=300, bbox_inches="tight")
plt.show()

# =========================
# Figure 2: 平均谱线（阴影更宽，无底格，无虚线）
# =========================
fig2, ax2 = plt.subplots(figsize=(8, 4.5))

xpos = np.arange(8)

ax2.plot(xpos, mean_spec, linewidth=2.2, marker="o")
ax2.fill_between(
    xpos,
    mean_spec - shade_spec,
    mean_spec + shade_spec,
    alpha=0.30
)

# 每个点单独着色
for i in range(8):
    ax2.scatter(
        i, mean_spec[i],
        s=75,
        color=band_colors[i],
        edgecolors="black",
        linewidths=0.6,
        zorder=3
    )

ax2.set_xticks(xpos)
ax2.set_xticklabels([f"{w:.1f}" for w in wavelengths], fontsize=10)
ylabel = "Mean response" if norm_mode == "none" else "Mean sigmoid response"
ax2.set_ylabel(ylabel, fontsize=11)
ax2.set_xlabel("Wavelength (nm)", fontsize=11)
ax2.set_title(f"Average spectral profile over time (mean ± {shade_scale:.1f}×std)", fontsize=12)
if norm_mode != "none":
    ax2.set_ylim(*response_ylim)

# 去掉底格
ax2.grid(False)

# 可选：让图更干净
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(save_fig2, dpi=300, bbox_inches="tight")
plt.show()


# =========================
# Figure 3: 时间-谱段图
# 横坐标: frame
# 纵坐标: 0,1,2,3,4,5,6,7
# 从上到下依次是 0 -> 7
# =========================
fig3, ax3 = plt.subplots(figsize=(10, 4.8))

# 变成 [8, T]，纵轴是band，横轴是frame
heatmap_data = X.T

imshow_kwargs = dict(
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
    origin="upper",  # 确保最上面是band 0
)
if norm_mode != "none":
    imshow_kwargs["vmin"], imshow_kwargs["vmax"] = response_ylim
im = ax3.imshow(heatmap_data, **imshow_kwargs)

# x轴用frame
num_xticks = min(10, len(frames))
xtick_idx = np.linspace(0, len(frames) - 1, num_xticks, dtype=int)
ax3.set_xticks(xtick_idx)
ax3.set_xticklabels(frames[xtick_idx], fontsize=10)

# y轴 0~7，从上到下显示
ax3.set_yticks(np.arange(8))
ax3.set_yticklabels([str(i) for i in range(8)], fontsize=10)

ax3.set_xlabel("Frame", fontsize=11)
ax3.set_ylabel("Spectral band index", fontsize=11)
ax3.set_title("Temporal spectral signature map", fontsize=12)

cbar = fig3.colorbar(im, ax=ax3, fraction=0.035, pad=0.02)
if norm_mode == "none":
    cbar.set_label("Raw pooled spectral value", fontsize=10)
else:
    cbar.set_label("Sigmoid response", fontsize=10)

plt.tight_layout()
plt.savefig(save_fig3, dpi=300, bbox_inches="tight")
plt.show()


# =========================
# Figure 4: 谱段时间波动
# =========================
fig4, ax4 = plt.subplots(figsize=(8, 4.5))

bars = ax4.bar(np.arange(8), std_spec, width=0.7)
for i, b in enumerate(bars):
    b.set_color(band_colors[i])
    b.set_edgecolor("black")
    b.set_linewidth(0.6)

ax4.axvline(4.5, color="gray", linestyle="--", linewidth=1.2)
ax4.set_xticks(np.arange(8))
ax4.set_xticklabels([f"{w:.1f}" for w in wavelengths], fontsize=10)
ax4.set_ylabel("Temporal std", fontsize=11)
ax4.set_xlabel("Wavelength (nm)", fontsize=11)
ax4.set_title("Band-wise temporal variability (lower = more stable)", fontsize=12)
ax4.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(save_fig4, dpi=300, bbox_inches="tight")
plt.show()


print("Saved figures:")
print("  ", save_fig1)
print("  ", save_fig2)
print("  ", save_fig3)
print("  ", save_fig4)