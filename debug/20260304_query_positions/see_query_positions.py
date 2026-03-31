"""
加载模型后查看 decoder 中 detection query 的位置和大小（ref_pts / det_anchor）。
实现风格与 debug/debug.py 一致：同一套 config/checkpoint 加载方式，输出 query 的归一化 (cx, cy, w, h[, angle]) 及统计与可视化。

python debug/20260304_query_positions/see_query_positions.py \
  --train-config /data4/litianhao/hsmot/memotr/spectralemb/22_7_2_only_detr_edgeswap_lrdrop_newtransforms_noscem_noiou/train/config.yaml \
  --checkpoint /data4/litianhao/hsmot/memotr/spectralemb/22_7_2_only_detr_edgeswap_lrdrop_newtransforms_noscem_noiou/checkpoint_39.pth \
  --output-dir /data4/litianhao/hsmot/memotr/spectralemb/22_7_2_only_detr_edgeswap_lrdrop_newtransforms_noscem_noiou/vis_query_positions \
  --save-csv \
  --vis \
  --vis-size 900,1200 \
  --vis-max-queries 500

"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# 与 debug.py 一致：支持从项目根或本文件所在目录运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.utils import load_checkpoint, get_model
from utils.utils import load_yaml_with_inheritance


def get_det_query_ref_points(model_core) -> torch.Tensor:
    """
    获取 detection query 的参考点（位置+大小）。
    - use_dab: det_anchor (N_det, 5)，未 sigmoid，格式 [cx, cy, w, h, angle]
    - 非 DAB: transformer.reference_points(det_query_embed) 得到 (N_det, 2)，仅中心
    Returns:
        ref: (N_det, 2) 或 (N_det, 5)，值域在 sigmoid 后为 [0,1]
    """
    if getattr(model_core, "use_dab", False):
        return model_core.det_anchor.detach()  # (N_det, 5)
    ref = model_core.get_det_reference_points()  # (N_det, 2)
    return ref.detach()


def ref_points_to_numpy(ref: torch.Tensor, apply_sigmoid: bool = True) -> np.ndarray:
    """(N, 2) 或 (N, 5) -> numpy，可选 sigmoid 到 [0,1]。"""
    if apply_sigmoid:
        ref = ref.sigmoid()
    return ref.detach().cpu().float().numpy()


def print_and_save_query_positions(
    ref: torch.Tensor,
    output_dir: str,
    save_csv: bool = True,
    name_prefix: str = "det_query_ref",
) -> None:
    """打印 query 位置/大小统计，并可选保存 CSV。"""
    arr = ref_points_to_numpy(ref, apply_sigmoid=True)
    n, dim = arr.shape

    dim_names = ["cx", "cy", "w", "h", "angle"] if dim >= 5 else ["cx", "cy"]
    print(f"[Query ref] shape=({n}, {dim}), after sigmoid in [0,1]")
    for i, name in enumerate(dim_names[:dim]):
        col = arr[:, i]
        print(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")

    if save_csv and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe = name_prefix.replace(".", "_")
        path = os.path.join(output_dir, f"{safe}_shape{n}x{dim}.csv")
        df = pd.DataFrame(arr, columns=dim_names[:dim])
        df.to_csv(path, index=False)
        print(f"  saved: {path}")


def draw_queries_on_canvas(
    ref: torch.Tensor,
    height: int,
    width: int,
    out_path: str,
    max_queries: int = 500,
    line_width: int = 1,
) -> None:
    """
    在画布上绘制 query 框（仅当 ref 为 5 维时绘制旋转框，否则只画中心点）。
    ref: (N, 5) 或 (N, 2)，sigmoid 后的归一化坐标。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[Warn] matplotlib not available, skip drawing.")
        return

    arr = ref_points_to_numpy(ref, apply_sigmoid=True)
    n, dim = arr.shape
    n = min(n, max_queries)
    arr = arr[:n]

    fig, ax = plt.subplots(1, 1, figsize=(width / 80, height / 80))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    if dim >= 5:
        # 旋转框: cx, cy, w, h, angle (弧度)
        cx = arr[:, 0] * width
        cy = arr[:, 1] * height
        w = arr[:, 2] * width
        h = arr[:, 3] * height
        angle_rad = arr[:, 4] * (2 * np.pi)  # 假设 angle 归一化到 [0,1] 对应 [0, 2pi] #TODO 这里对应关系错误，应该是le135
        for i in range(n):
            rect = patches.Rectangle(
                (-w[i] / 2, -h[i] / 2),
                w[i],
                h[i],
                linewidth=line_width,
                edgecolor="b",
                facecolor="none",
            )
            t = patches.transforms.Affine2D().rotate(angle_rad[i]).translate(cx[i], cy[i]) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
    else:
        # 仅中心点
        cx = arr[:, 0] * width
        cy = arr[:, 1] * height
        ax.scatter(cx, cy, s=2, c="b", alpha=0.6)

    ax.axis("off")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()
    print(f"[Vis] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load model and inspect detection query positions/sizes (ref_pts / det_anchor)."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        help="Path to train config yaml (same as debug.py).",
        default="/data/users/wangying01/lth/hsmot/MeMOTR/configs_hsmot_spectral_embed_99/18_train_half_hsmot8ch_spectralembConv_fconv10lr_scem_priormap.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to .pth checkpoint.",
        default="/data4/litianhao/hsmot/memotr/spectralemb/18_half_priormap_99/checkpoint_15.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/users/wangying01/lth/hsmot/MeMOTR/debug/20260304_query_positions/outputs",
        help="Directory for CSV and optional visualization.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=True,
        help="Save query ref points to CSV.",
    )
    parser.add_argument(
        "--no-save-csv",
        action="store_false",
        dest="save_csv",
        help="Disable saving CSV.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        default=False,
        help="Draw query boxes on a canvas and save as PNG.",
    )
    parser.add_argument(
        "--vis-size",
        type=str,
        default="640,640",
        help="Canvas size for visualization: height,width (e.g. 640,640).",
    )
    parser.add_argument(
        "--vis-max-queries",
        type=int,
        default=500,
        help="Max number of queries to draw (default 500).",
    )

    args = parser.parse_args()

    train_cfg_path = args.train_config
    if not os.path.isfile(train_cfg_path):
        raise FileNotFoundError(f"Config not found: {train_cfg_path}")
    train_config = load_yaml_with_inheritance(path=train_cfg_path)

    print(f"[Info] Building model from: {train_cfg_path}")
    model = build_model(config=train_config)
    device = torch.device(args.device)
    model.to(device)

    print(f"[Info] Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model=model, path=args.checkpoint)

    inner = get_model(model)
    use_dab = getattr(inner, "use_dab", False)
    n_det = getattr(inner, "n_det_queries", 0)
    print(f"[Info] use_dab={use_dab}, n_det_queries={n_det}")

    ref = get_det_query_ref_points(inner)
    if ref.dim() == 2 and ref.shape[0] > 0:
        print("\n===== Detection query reference points (position & size) =====")
        print_and_save_query_positions(
            ref,
            output_dir=args.output_dir,
            save_csv=args.save_csv,
            name_prefix="det_query_ref",
        )
        if args.vis:
            hw = [int(x) for x in args.vis_size.replace(" ", "").split(",")]
            if len(hw) >= 2:
                h, w = hw[0], hw[1]
            else:
                h = w = 640
            out_path = os.path.join(args.output_dir, "query_positions_vis.png")
            draw_queries_on_canvas(
                ref,
                height=h,
                width=w,
                out_path=out_path,
                max_queries=args.vis_max_queries,
            )
    else:
        print("[Warn] No detection query ref points (empty or wrong shape).")


if __name__ == "__main__":
    main()
