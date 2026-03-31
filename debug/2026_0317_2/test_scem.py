"""
查看 scem 模块的输入输出。实现风格与 debug/debug.py 一致：同一套 config/checkpoint 加载，
对 scem_module 注册 forward hook 同时捕获输入与输出，并打印/保存 CSV/热力图。

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import sys
import argparse
import json
from typing import List, Dict, Any, Tuple
import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# 与 debug.py 一致：支持从项目根或本文件所在目录运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DEBUG_DIR not in sys.path:
    sys.path.insert(0, DEBUG_DIR)

# 复用 debug.py 中的工具与前向逻辑（debug 目录下 debug.py）
import debug as db

from models import build_model
from models.utils import load_checkpoint, get_model
from utils.utils import load_yaml_with_inheritance
from utils.nested_tensor import tensor_list_to_nested_tensor, tensor_list_to_nested_tensor_with_shared_shapes
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset


def _tensors_from_obj(obj, prefix: str, captured: Dict[str, torch.Tensor]) -> None:
    """递归把 obj 中的 Tensor 写入 captured，键为 prefix + 下标。"""
    if isinstance(obj, torch.Tensor):
        captured[prefix] = obj.detach().clone()
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            _tensors_from_obj(x, f"{prefix}.{i}", captured)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _tensors_from_obj(v, f"{prefix}.{k}", captured)
    # 其他类型忽略


def _parse_csv_patterns(raw: str) -> List[str]:
    return [s.strip().lower() for s in str(raw).split(",") if s.strip()]


def _infer_hook_vis_mode(name_lower: str, tensor: torch.Tensor, args) -> str:
    """
    Decide visualization mode for a captured hook tensor.
    Priority: mask > attn > value > disc > allch > feature > None
    """
    if any(p in name_lower for p in args.vis_mask_patterns) or tensor.dtype == torch.bool:
        return "mask"
    if any(p in name_lower for p in args.vis_attn_patterns):
        return "attn"
    if any(p in name_lower for p in args.vis_value_patterns):
        return "value"
    if any(p in name_lower for p in args.vis_disc_patterns):
        return "disc"
    if any(p in name_lower for p in args.vis_allch_patterns):
        return "allch"
    if any(p in name_lower for p in args.vis_feature_patterns):
        return "feature"
    # Fallback is now treated as feature mode by default.
    return "None"


def run_forward_once_scem(
    model: nn.Module,
    seq_dir: str,
    npy2rgb: bool,
    hook_module_names: List[str],
    capture_inputs: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], np.ndarray]:
    """
    跑一次前向，对指定模块注册 hook，同时捕获输入与输出（便于查看 scem 的输入输出）。
    captured 的 key 形如: "scem_module.in.0.0", "scem_module.in.1", "scem_module.out.0", "scem_module.out.1"
    """
    device = next(model.parameters()).device
    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {seq_dir}")
    image, ori_image = dataset[0][0]
    padded_img_shape = (image.shape[1], image.shape[2], image.shape[0])
    frame = tensor_list_to_nested_tensor_with_shared_shapes([image.squeeze(0)], effective_img_shape=ori_image.shape, padded_img_shape=padded_img_shape).to(device)

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    if hook_module_names:
        modules = db.find_modules_by_names(get_model(model), hook_module_names)
        if len(modules) == 0:
            print(f"[Warn] No modules matched for hooks: {hook_module_names}")
        for nm, m in modules.items():
            def _make_hook(key: str, capture_inp: bool):
                def _hook(_mod, _inp, _out):
                    try:
                        if capture_inp and _inp:
                            _tensors_from_obj(_inp, f"{key}.in", captured)
                        # 输出
                        _tensors_from_obj(_out, f"{key}.out", captured)
                    except Exception as e:
                        print(f"[HookError] {key}: {e}")
                return _hook
            handles.append(m.register_forward_hook(_make_hook(nm, capture_inputs)))

    model.eval()
    TrackInstances.set_static_properties(
        use_spectral_decoder=False,
        decoder_spectral_weights_dim=0,
    )
    with torch.no_grad():
        tracks = db.build_tracks_for_infer(model, device)
        outputs = model(frame=frame, tracks=tracks, debug=True)

    for h in handles:
        h.remove()

    if torch.is_tensor(ori_image):
        ori_image = ori_image.detach().cpu().numpy()
    if ori_image.dtype != np.uint8:
        ori_image = np.clip(ori_image, 0, 255).astype(np.uint8)

    return outputs, captured, ori_image


def main():

    parser = argparse.ArgumentParser(
        description="Inspect scem module inputs and outputs (same config/checkpoint as debug.py)."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="/data/users/wangying01/lth/hsmot/MeMOTR/debug/2026_0317/20260317_detr_debug.yaml",
        help="Path to train config yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data4/litianhao/hsmot/memotr/spectralemb/20260317_detr_2/checkpoint_39.pth",
        help="Path to .pth checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/users/wangying01/lth/hsmot/data",
        help="DATA_ROOT for dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hsmot_8ch",
        help="Dataset name (e.g. hsmot_8ch).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split folder (val/test).",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=",".join(
            [
                "data39-1", 
                "data48-1", 
                "data46-12", 
                "data36-13", 
                "data30-10"
                ]
        ),
        help="Sequence folder name, or 'all' for all sequences.",
    )
    parser.add_argument(
        "--npy2rgb",
        action="store_true",
        help="Use npy2rgb in SeqDataset.",
    )

    parser.add_argument(
        "--no-capture-inputs",
        action="store_true",
        help="Do not capture hook module inputs, only outputs.",
    )
    parser.add_argument(
        "--list-modules",
        default=True,
        action="store_true",
        help="List all modules and exit (optional filter: scem).",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=False,
        help="Save tensors to CSV.",
    )
    parser.add_argument(
        "--save-heatmaps",
        action="store_true",
        default=True,
        help="Save 2D heatmaps for 4D tensors (e.g. 1xCxHxW).",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default="",
        help="If set, dump shapes/dtypes of outputs and hooks to this JSON.",
    )
    parser.add_argument(
        "--list-params",
        type=str,
        default=True,
        help="Comma-separated substrings to match parameter/buffer names; only prints matched names (no save). Use to see which params can be saved.",
    )
    parser.add_argument(
        "--hook-params",
        type=str,
        default=",".join(
            [
                "scem_module.posterior.pi_head.spec_pi.spectral_db_logits",
                "scem_module.posterior.pi_head.out",
            ]
        ),
        help="Comma-separated substrings to match parameter/buffer names; matched params are printed and saved as CSV/heatmap.",
    )
    parser.add_argument(
        "--hook-modules",
        type=str,
        default=",".join(
            [
                "scem_module",
                # "scem_module.feature_fusion",
                # "scem_module.posterior.bg_head",
                # "scem_module.posterior.bg_head.mu_b_head",
                # "scem_module.posterior.bg_head.logsig_b_head",
                "scem_module.posterior.pi_head",
                # "scem_module.posterior.pi_head.space_to_spec_gate",
                # "scem_module.posterior.pi_head.spec_proj",
                # "scem_module.posterior.pi_head.local_fuse",
                "scem_module.posterior.pi_head.spec_pi",
                # "transformer.encoder.layers.0.self_attn",
                # "transformer.encoder.layers.5.self_attn",
            ]
        ),
        help=(
            "Comma-separated module paths to hook "
            "(default includes SCEM feature_fusion / BGHead / PIHead)."
        ),
    )
    parser.add_argument(
        "--vis-mask-patterns",
        type=str,
        default=",".join(
            [
                "mask",
            ]
        ),
        help="Comma-separated hook-name patterns mapped to mask visualization.",
    )
    parser.add_argument(
        "--vis-attn-patterns",
        type=str,
        default=",".join(
            [
                # "transformer.encoder.layers.0.self_attn",
                # "transformer.encoder.layers.5.self_attn",
            ]
        ),
        help="Comma-separated hook-name patterns mapped to cross-attention tiled visualization.",
    )
    parser.add_argument(
        "--vis-disc-patterns",
        type=str,
        default=",".join(
            [
                # "posterior.bg_head.out",
                # "posterior.pi_head.out",
                # "scem_module.out",
                "pi_head.out.0"
            ]
        ),
        help="Comma-separated hook-name patterns mapped to discriminative heatmap visualization.",
    )
    parser.add_argument(
        "--vis-value-patterns",
        type=str,
        default=",".join(
            [
                "scem_module.out.4.0",
            ]
        ),
        help="Comma-separated hook-name patterns mapped to value-annotated heatmap (1D/2D).",
    )
    parser.add_argument(
        "--vis-feature-patterns",
        type=str,
        default=",".join(
                [
                    "scem_module.posterior.pi_head.space_to_spec_gate.in",
                    "scem_module.posterior.pi_head.space_to_spec_gate.out",
                    "spec_pi.in",
                    "pi_head.out.1"
                ]),
        help="Comma-separated hook-name patterns mapped to feature energy + PCA visualization.",
    )
    parser.add_argument(
        "--vis-allch-patterns",
        type=str,
        default=",".join(
            [
                "spec_pi.out",
                "scem_module.out.4.2",
            ]
        ),
        help="Comma-separated hook-name patterns mapped to all-channel raw visualization.",
    )

    args = parser.parse_args()
    args.vis_mask_patterns = _parse_csv_patterns(args.vis_mask_patterns)
    args.vis_attn_patterns = _parse_csv_patterns(args.vis_attn_patterns)
    args.vis_disc_patterns = _parse_csv_patterns(args.vis_disc_patterns)
    args.vis_value_patterns = _parse_csv_patterns(args.vis_value_patterns)
    args.vis_feature_patterns = _parse_csv_patterns(args.vis_feature_patterns)
    args.vis_allch_patterns = _parse_csv_patterns(args.vis_allch_patterns)
    csv_output_dir = os.path.join(CURRENT_DIR, "csvs")
    heatmap_dir = os.path.join(CURRENT_DIR, "heatmaps")

    train_cfg_path = args.train_config
    if not os.path.isfile(train_cfg_path):
        raise FileNotFoundError(f"Config not found: {train_cfg_path}")
    train_config = load_yaml_with_inheritance(path=train_cfg_path)


    print(f"[Info] Building model from: {train_cfg_path}")
    # 设置GPU
    model = build_model(config=train_config)



    print(f"[Info] Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model=model, path=args.checkpoint)

    inner = get_model(model)
    if not getattr(inner, "scem_module", None):
        print("[Warn] Model has no scem_module; hook may not match.")

    if args.list_modules or args.list_params:
        if args.list_modules:
            print("\n===== Available modules (for hook) =====")
            all_mods = db.list_all_modules(inner)
            for name in all_mods:
                # if "scem" in name.lower():
                print(f"  {name}")
            print(f"\nTotal listed: {len([n for n in all_mods if 'scem' in n.lower()])} scem-related, {len(all_mods)} total.")
        if args.list_params:
            print("\n===== Available params (for hook) =====")
            all_named = db.collect_named_params_and_buffers(inner)
            for name in all_named:
                if "scem" in name.lower():
                    print(f"  {name}")
            print(f"\nTotal listed: {len([n for n in all_named if 'scem' in n.lower()])} scem-related, {len(all_named)} total.")
        # return

    # hook_params: 对匹配到的参数打印并保存为 CSV / heatmap
    hook_param_patterns = [s.strip() for s in args.hook_params.split(",") if s.strip()]
    if hook_param_patterns:
        matched_param_names = db.match_keys(list(all_named.keys()), hook_param_patterns)
        if matched_param_names:
            param_csv_dir = os.path.join(csv_output_dir, "params")
            param_map_dir = os.path.join(heatmap_dir, "params")
            os.makedirs(param_csv_dir, exist_ok=True)
            os.makedirs(param_map_dir, exist_ok=True)
            print("\n===== Hook params (打印并保存) =====")
            for name in matched_param_names:
                t = all_named[name]
                print(f"  {name}: shape={list(t.shape)}, dtype={t.dtype}, numel={t.numel()}")
                safe_name = name.replace(".", "_").replace("/", "_")
                if args.save_csv:
                    db.save_tensor_to_csv(safe_name, t, param_csv_dir)
                if args.save_heatmaps:
                    with torch.no_grad():
                        t_np = t.detach().cpu().numpy()

                    # 针对 spectral_db_logits 的特殊可视化：N×8
                    if "spectral_db_logits" in name and t_np.ndim == 2 and t_np.shape[1] == 8:
                        N = t_np.shape[0]
                        vmin, vmax = t_np.min(), t_np.max()

                        # 1) heatmap（N×8）
                        db.save_heatmap(
                            t_np,
                            os.path.join(param_map_dir, f"{safe_name}__heatmap.png"),
                            vmin=vmin,
                            vmax=vmax,
                            add_colorbar=True,
                        )

                        # 2) 每个 [1,8] 画一条曲线，共 N 条
                        plt.figure(figsize=(6, 4))
                        x = np.arange(8)
                        for i in range(N):
                            plt.plot(x, t_np[i], alpha=0.4)
                        plt.xlabel("spectral band")
                        plt.ylabel("value")
                        plt.title(f"{safe_name} line curves (N={N})")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(param_map_dir, f"{safe_name}__lines.png"), dpi=300)
                        plt.close()

                        # 3) PCA 降到二维并散点图显示
                        X = t_np.astype(np.float32)
                        X_mean = X.mean(axis=0, keepdims=True)
                        Xc = X - X_mean
                        # 协方差矩阵 + SVD 求前两主成分
                        cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
                        U, S, Vt = np.linalg.svd(cov)
                        W = U[:, :2]              # [8,2]
                        X2 = Xc @ W               # [N,2]

                        plt.figure(figsize=(4, 4))
                        plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.7)
                        plt.xlabel("PC1")
                        plt.ylabel("PC2")
                        plt.title(f"{safe_name} PCA-2D (N={N})")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(param_map_dir, f"{safe_name}__pca2.png"), dpi=300)
                        plt.close()

                    else:
                        # 默认可视化逻辑
                        if t_np.ndim == 1:
                            db.save_heatmap(
                                t_np.reshape(1, -1, 1),
                                os.path.join(param_map_dir, f"{safe_name}.png"),
                                vmin=t_np.min(),
                                vmax=t_np.max(),
                                add_colorbar=True,
                            )
                        elif t_np.ndim == 2:
                            db.save_heatmap(
                                t_np,
                                os.path.join(param_map_dir, f"{safe_name}.png"),
                                vmin=t_np.min(),
                                vmax=t_np.max(),
                                add_colorbar=True,
                            )
                        elif t_np.ndim == 4 and t_np.shape[0] == 1:
                            for c in range(min(16, t_np.shape[1])):
                                db.save_heatmap(
                                    t_np[0, c],
                                    os.path.join(param_map_dir, f"{safe_name}_ch{c}.png"),
                                    vmin=t_np.min(),
                                    vmax=t_np.max(),
                                    add_colorbar=True,
                                )
                        elif t_np.ndim == 3 and t_np.shape[0] == 1:
                            db.save_heatmap(
                                t_np[0],
                                os.path.join(param_map_dir, f"{safe_name}.png"),
                                vmin=t_np.min(),
                                vmax=t_np.max(),
                                add_colorbar=True,
                            )
            print(f"  Saved to CSV dir: {param_csv_dir}, heatmap dir: {param_map_dir}")
        else:
            print(f"[Warn] No params/buffers matched for hook-params: {hook_param_patterns}")

    # 解析序列目录
    base_name = args.dataset_name.replace("_8ch", "") if "hsmot" in args.dataset_name else args.dataset_name
    npy_root = os.path.join(args.data_root, base_name, args.split, "npy")
    rgb_root = os.path.join(args.data_root, base_name, args.split, "rgb")

    if args.seq.lower() == "all":
        if not os.path.isdir(npy_root):
            raise FileNotFoundError(f"npy root not found: {npy_root}")
        seq_list = sorted([d for d in os.listdir(npy_root) if os.path.isdir(os.path.join(npy_root, d))])
    else:
        seq_list = [s.strip() for s in args.seq.split(",") if s.strip()]

    hook_names = [s.strip() for s in args.hook_modules.split(",") if s.strip()]
    all_meta: Dict[str, Any] = {"per_seq": []}

    for seq in seq_list:
        seq_dir = os.path.join(npy_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[Warn] Skip missing seq dir: {seq_dir}")
            continue

        print(f"\n[Info] Running forward once (scem io) seq={seq}")
        outputs, captured, _ = run_forward_once_scem(
            model=model,
            seq_dir=seq_dir,
            npy2rgb=args.npy2rgb,
            hook_module_names=hook_names,
            capture_inputs=not args.no_capture_inputs,
        )

        csv_out_dir = os.path.join(csv_output_dir, seq)
        map_out_dir = os.path.join(heatmap_dir, seq)
        os.makedirs(csv_out_dir, exist_ok=True)
        os.makedirs(map_out_dir, exist_ok=True)

        seq_meta: Dict[str, Any] = {"seq": seq, "outputs": {}, "hooks": {}}

        # 1) 模型输出中的 scem 相关键
        heatmap_outputs = ["scem_gamma", "scem_log_mix"]
        print("\n===== Model outputs (scem-related) =====")
        if isinstance(outputs, dict):
            for k in list(outputs.keys()):
                if "scem" not in k.lower():
                    continue
                v = outputs[k]
                if isinstance(v, torch.Tensor):
                    print(f"  out[{k}]: shape={list(v.shape)}, dtype={v.dtype}, device={v.device}")
                    seq_meta["outputs"][k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                    if k in heatmap_outputs:
                        v_np = v.detach().cpu().numpy()
                        for c in range(min(8, v_np.shape[1])):
                            arr2d = v_np[0, c]
                            db.save_heatmap(
                                arr2d,
                                os.path.join(map_out_dir, f"{seq}__out_{k}__ch{c}.png"),
                                vmin=v_np.min(),
                                vmax=v_np.max(),
                                add_colorbar=True,
                            )
                    if args.save_csv:
                        db.save_tensor_to_csv(f"{seq}__out_{k}", v, csv_out_dir)
                else:
                    print(f"  out[{k}]: type={type(v).__name__}")
                    seq_meta["outputs"][k] = {"type": type(v).__name__}
        else:
            print(f"  outputs type: {type(outputs).__name__}")
            seq_meta["outputs"]["__type__"] = type(outputs).__name__

        # 2) Hook 捕获的 scem 输入/输出
        if captured:
            print("\n===== Hook captured (scem input / output) =====")
            for k, t in captured.items():
                seq_meta["hooks"][k] = {"shape": list(t.shape), "dtype": str(t.dtype)}

                safe_name = k.replace(".", "_")
                name_lower = k.lower()
                t_det = t.detach()
                vis_mode = _infer_hook_vis_mode(name_lower=name_lower, tensor=t_det, args=args)
                print(f"  {k}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}, vis_mode={vis_mode}")

                # 0) mask 专门处理：bool 类型或名称中包含 "mask" 的张量
                if vis_mode == "mask":
                    tt = t_det
                    if tt.dim() == 3:
                        tt = tt.unsqueeze(0)  # [1,H,W]
                    if tt.dim() == 4:
                        arr2d = tt[0, 0].float().cpu().numpy()
                    elif tt.dim() == 3:
                        arr2d = tt[0].float().cpu().numpy()
                    else:
                        arr2d = tt.float().cpu().numpy()

                    if args.save_heatmaps:
                        out_path = os.path.join(map_out_dir, f"{seq}__{safe_name}__mask.png")
                        # 0/1 → 灰度 mask，可视化有效/无效区域
                        db.save_scalar_heatmap(arr2d, out_path, vmin=0.0, vmax=1.0, cmap="gray")

                    if args.save_csv:
                        # 将 bool 转成 float 保存，便于后续分析
                        db.save_tensor_to_csv(f"{seq}__hook_{safe_name}", tt.float(), csv_out_dir)
                    continue

                # 0.5) cross_attention_prior 的注意力权重特殊可视化：
                # 只处理第二个输出 (attention weights)，shape [1, 24225, 8]
                # 其中 24225 = 120*152 + 60*76 + 30*38 + 15*19。
                # 对每个尺度，将 [h,w,8] 按通道平铺成 [2h, 4w]（2 行 × 4 列）的单张 heatmap。
                if vis_mode == "attn":
                    tt = t_det
                    if tt.dim() != 3 or tt.shape[0] != 1 or tt.shape[2] != 8:
                        # 形状不符合预期则跳过专门处理，走后面的通用逻辑
                        pass
                    else:
                        att = tt[0].cpu().numpy()  # [24225, 8]
                        num_tokens, num_ch = att.shape
                        if num_tokens == 24225 and num_ch == 8 and args.save_heatmaps:
                            scales = [(120, 152), (60, 76), (30, 38), (15, 19)]
                            vmin, vmax = float(att.min()), float(att.max())
                            offset = 0
                            for lvl, (hh, ww) in enumerate(scales):
                                n = hh * ww
                                group = att[offset:offset + n, :]  # [n,8]
                                offset += n
                                group = group.reshape(hh, ww, 8)   # [H,W,8]

                                # 按 2×4 网格平铺 8 个通道：输出 [2h,4w]
                                tiled = np.zeros((2 * hh, 4 * ww), dtype=group.dtype)
                                for ch in range(8):
                                    r = ch // 4
                                    c = ch % 4
                                    patch = group[:, :, ch]
                                    tiled[r * hh:(r + 1) * hh,c * ww:(c + 1) * ww] = patch

                                out_path = os.path.join(
                                    map_out_dir,
                                    f"{seq}__{safe_name}__att_lvl{lvl}.png",
                                )
                                db.save_scalar_heatmap(tiled, out_path, vmin=vmin, vmax=vmax, cmap="jet")
                        # attention 已按专门规则可视化，仍然按原样保存 CSV
                        if args.save_csv:
                            db.save_tensor_to_csv(f"{seq}__hook_{safe_name}", t_det, csv_out_dir)
                        continue

                # 1) 非 mask 的张量，先保存原始 CSV，便于数值分析
                if args.save_csv:
                    db.save_tensor_to_csv(f"{seq}__hook_{safe_name}", t_det, csv_out_dir)

                # 2) 判别型量：直接 heatmap
                if vis_mode == "value" and args.save_heatmaps:
                    tt = t_det
                    if tt.dim() == 2:
                        arr = tt.cpu().numpy()
                    elif tt.dim() == 1:
                        arr = tt.cpu().numpy()
                    elif tt.dim() == 3 and tt.shape[0] == 1:
                        arr = tt[0].cpu().numpy()
                    elif tt.dim() == 4 and tt.shape[0] == 1 and tt.shape[1] == 1:
                        arr = tt[0, 0].cpu().numpy()
                    else:
                        print(f"[Warn] value mode expects 1D/2D (or squeezable), got {list(tt.shape)} for {k}")
                        continue
                    vmin, vmax = float(np.min(arr)), float(np.max(arr))
                    out_path = os.path.join(map_out_dir, f"{seq}__{safe_name}__value.png")
                    db.save_value_heatmap(arr, out_path, vmin=vmin, vmax=vmax, cmap="jet", fmt=".3f")
                    continue

                # 3) 判别型量：直接 heatmap
                if vis_mode == "disc":
                    tt = t_det
                    if tt.dim() == 3:
                        tt = tt.unsqueeze(0)
                    if tt.dim() == 4 and args.save_heatmaps:
                        arr2d = tt[0, 0].cpu().numpy()
                        vmin, vmax = float(arr2d.min()), float(arr2d.max())
                        out_path = os.path.join(
                            map_out_dir, f"{seq}__{safe_name}__disc.png"
                        )
                        db.save_scalar_heatmap(arr2d, out_path, vmin=vmin, vmax=vmax, cmap="jet")
                    continue
                  

                # 4) allch
                if vis_mode == "allch" and args.save_heatmaps:
                    tt = t_det
                    if tt.dim() == 4:
                        tt = tt[0]  # [C,H,W]
                    if tt.dim() == 3:
                        t_np = tt.cpu().numpy()
                        for c in range(t_np.shape[0]):
                            arr2d = t_np[c]
                            vmin, vmax = float(arr2d.min()), float(arr2d.max())
                            out_path = os.path.join(
                                map_out_dir, f"{seq}__{safe_name}__allch{c}.png"
                            )
                            # Keep raw channel values; only color-map for display.
                            db.save_scalar_heatmap(arr2d, out_path, vmin=vmin, vmax=vmax, cmap="jet")
                    elif tt.dim() == 2:
                        arr2d = tt.cpu().numpy()
                        vmin, vmax = float(arr2d.min()), float(arr2d.max())
                        out_path = os.path.join(
                            map_out_dir, f"{seq}__{safe_name}__allch0.png"
                        )
                        db.save_scalar_heatmap(arr2d, out_path, vmin=vmin, vmax=vmax, cmap="jet")
                    continue

                # 5) 特征型量：feature energy + PCA
                if vis_mode == "feature" and t_det.dim() >= 3 and args.save_heatmaps:
                    try:
                        tt = t_det
                        if tt.dim() == 3:
                            tt = tt.unsqueeze(0)
                        fe_path = os.path.join(map_out_dir, f"{seq}__{safe_name}__featE.png")
                        db.save_feature_energy_heatmap(tt, fe_path)
                        pca_path = os.path.join(map_out_dir, f"{seq}__{safe_name}__pca.png")
                        db.save_feature_pca_rgb(tt, pca_path)
                    except Exception as e:
                        print(f"[Warn] feature visualization failed: {k}, shape={list(t_det.shape)}, err={e}")
                    continue
                    


        else:
            if hook_names:
                print("[Warn] No hook tensors captured (check module path, e.g. scem_module).")

        all_meta["per_seq"].append(seq_meta)

    if args.dump_json:
        try:
            with open(args.dump_json, "w") as f:
                json.dump(all_meta, f, indent=2)
            print(f"[Info] Dumped metadata to {args.dump_json}")
        except Exception as e:
            print(f"[Warn] Failed to dump json: {e}")


if __name__ == "__main__":
    main()
