import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import shutil

# Allow running from project root or this file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.utils import load_checkpoint, get_model
from utils.utils import yaml_to_dict
from utils.nested_tensor import tensor_list_to_nested_tensor
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset
import matplotlib.pyplot as plt
import cv2

def find_modules_by_names(root_module: nn.Module, dotted_names: List[str]) -> Dict[str, nn.Module]:
    """根据精确的点号路径查找模块，不做模糊匹配或递归类名匹配。
    例如："transformer.decoder.layers.0.self_attn"。
    """
    found: Dict[str, nn.Module] = {}

    def _get_by_path(module: nn.Module, dotted_path: str) -> nn.Module:
        cur = module
        for part in dotted_path.split("."):
            if not hasattr(cur, part):
                raise AttributeError(f"Module {cur.__class__.__name__} has no attribute '{part}' in path '{dotted_path}'")
            cur = getattr(cur, part)
        return cur

    for name in dotted_names:
        if not name:
            continue
        try:
            target = _get_by_path(root_module, name)
            if isinstance(target, nn.Module):
                found[name] = target
        except Exception as e:
            print(f"[Warn] Could not resolve module path '{name}': {e}")
    return found


def collect_named_params_and_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters(recurse=True):
        tensors[n] = p.data
    for n, b in model.named_buffers(recurse=True):
        tensors[n] = b.data
    return tensors


def list_all_modules(model: nn.Module, max_depth: int = 5) -> List[str]:
    """列出模型中所有可用的模块，用于hook"""
    module_names = []
    
    def _collect_modules(module: nn.Module, current_path: str = "", depth: int = 0):
        if depth > max_depth:
            return
        
        # 添加当前模块
        if current_path:
            module_names.append(current_path)
        
        # 递归搜索子模块
        for name, child in module.named_children():
            child_path = current_path + "." + name if current_path else name
            _collect_modules(child, child_path, depth + 1)
    
    _collect_modules(model)
    return module_names


def match_keys(keys: List[str], patterns: List[str]) -> List[str]:
    if not patterns:
        return []
    matched = []
    for k in keys:
        for pat in patterns:
            if pat in k:
                matched.append(k)
                break
    return matched


def save_tensor_to_csv(name: str, t: torch.Tensor, output_dir: str = ".") -> str:
    """Save tensor to CSV file with shape information in filename."""
    if t.numel() == 0:
        return ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with shape info
    shape_str = "_".join([str(s) for s in t.shape])
    safe_name = name.replace(".", "_").replace("/", "_")
    filename = f"{safe_name}_{shape_str}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Convert to numpy and save
    with torch.no_grad():
        t_np = t.detach().cpu().squeeze().numpy()
        
        if t_np.ndim == 1:
            # 1D tensor: save as single column
            df = pd.DataFrame(t_np, columns=[name])
        elif t_np.ndim == 2:
            # 2D tensor: save as is
            df = pd.DataFrame(t_np)
        else:
            # Higher dimensional: flatten and save with index indicating original position
            flat = t_np.flatten()
            df = pd.DataFrame(flat, columns=[f"{name}_flattened"])
            # Add shape info as comment in the first row
            df.loc[-1] = [f"# Original shape: {t.shape}"]
            df.index = df.index + 1
            df = df.sort_index()
        
        df.to_csv(filepath, index=False)
    
    return filepath

def pretty_print_tensor(name: str, t: torch.Tensor, max_numel: int = 20, save_csv: bool = False, output_dir: str = ".") -> None:
    device = str(t.device)
    shape = list(t.shape)
    dtype = str(t.dtype)
    numel = t.numel()
    print(f"[Tensor] {name}: shape={shape}, dtype={dtype}, device={device}, numel={numel}")
    if numel == 0:
        return
    
    # Save to CSV if requested
    if save_csv:
        csv_path = save_tensor_to_csv(name, t, output_dir)
        if csv_path:
            print(f"  saved to: {csv_path}")
    
    with torch.no_grad():
        flat = t.detach().view(-1)
        head = flat[:max_numel].cpu()
        vals = ", ".join([f"{v.item():.6g}" for v in head])
        suffix = " ..." if numel > max_numel else ""
        print(f"  values: [{vals}]{suffix}")


def build_tracks_for_infer(model: nn.Module, device: torch.device) -> List[TrackInstances]:
    inner = get_model(model)
    decoder_spectral_weights_dim = getattr(inner, "decoder_spectral_clusters", 1) * 8
    tracks = [TrackInstances(hidden_dim=inner.hidden_dim,
                             num_classes=inner.num_classes,
                             use_dab=getattr(inner, "use_dab", False))]
    return tracks

# >>> NEW: 2D CSV saver to avoid flatten for heatmaps
def save_2d_to_csv(filepath: str, arr2d: np.ndarray) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(arr2d).to_csv(filepath, index=False)
    
def run_forward_once(model: nn.Module,
                     seq_dir: str,
                     npy2rgb: bool,
                     hook_module_names: List[str],
                     args) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], np.ndarray]:
    device = next(model.parameters()).device

    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {seq_dir}")
    image, ori_image = dataset[0][0]  # ((image, ori_image), info)
    frame = tensor_list_to_nested_tensor([image]).to(device)

    captured: Dict[str, torch.Tensor] = {}
    handles = []
    if hook_module_names:
        modules = find_modules_by_names(get_model(model), hook_module_names)
        if len(modules) == 0:
            print(f"[Warn] No modules matched for hooks (exact paths expected): {hook_module_names}")
        for nm, m in modules.items():
            def _make_hook(key: str):
                def _hook(_mod, _inp, _out):
                    try:
                        if isinstance(_out, (list, tuple)):
                            for item in _out:
                                if isinstance(item, torch.Tensor):
                                    captured[key] = item.detach().clone()
                                    break
                        elif isinstance(_out, dict):
                            for vv in _out.values():
                                if isinstance(vv, torch.Tensor):
                                    captured[key] = vv.detach().clone()
                                    break
                        elif isinstance(_out, torch.Tensor):
                            captured[key] = _out.detach().clone()
                    except Exception as e:
                        print(f"[HookError] {key}: {e}")
                return _hook
            handles.append(m.register_forward_hook(_make_hook(nm)))

    model.eval()
    TrackInstances.set_static_properties(use_spectral_decoder=get_model(model).use_spectral_decoder, decoder_spectral_weights_dim=get_model(model).decoder_spectral_clusters * 8)
    with torch.no_grad():
        tracks = build_tracks_for_infer(model, device)
        outputs = model(frame=frame, tracks=tracks, debug=True)

    for h in handles:
        h.remove()

    # Ensure ori_image is numpy uint8 HxWx3
    if torch.is_tensor(ori_image):
        ori_image = ori_image.detach().cpu().numpy()
    if ori_image.dtype != np.uint8:
        ori_image = np.clip(ori_image, 0, 255).astype(np.uint8)

    return outputs, captured, ori_image
def save_heatmap(arr2d: np.ndarray, filepath: str, vmin: float = 0.0, vmax: float = 1.0, add_colorbar: bool = False) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    arr = np.clip(arr2d, vmin, vmax)  # enforce 0..1 mapping
    plt.figure()
    plt.imshow(arr, vmin=vmin, vmax=vmax)
    if add_colorbar:
        plt.colorbar()
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Debug utility: load checkpoint, inspect params, forward once and capture outputs.")

    parser.add_argument("--train-config", type=str,
                        help="Path to train/config.yaml or any config yaml compatible with build_model.",
                        default='/data/users/litianhao/hsmot_code/MeMOTR/configs_hsmot_spectral_embed_197/02_train_hsmot8ch_spectralembConv_fconv10lr.yaml')
    parser.add_argument("--checkpoint", type=str, help="Path to .pth checkpoint to load.",
    default='/data4/litianhao/hsmot/memotr/spectralemb/02_v2_fconv10lr_2gpu_git_reset/checkpoint_16.pth'
    )

    # Parameter inspection
    parser.add_argument("--show-params", type=str, default="",
                        help="Comma-separated substrings to match parameter/buffer names to display.")
    parser.add_argument("--print-values", action="store_true", default=True,
                        help="Print actual values (head) of tensors; otherwise only shape/dtype.")
    parser.add_argument("--save-csv", action="store_true", default=True,
                        help="Save tensors to CSV files.")
    parser.add_argument("--csv-output-dir", type=str, default="/data/users/litianhao/hsmot_code/MeMOTR/debug/encoder_tensor_csvs",
                        help="Directory to save CSV files (default: ./tensor_csvs).")

    # >>> NEW: heatmap saving options
    parser.add_argument("--save-heatmaps", action="store_true", default=True,
                        help="Whether to save 2D heatmaps of hook outputs (expects 1x8xMxN).")
    parser.add_argument("--heatmap-dir", type=str, default="/data/users/litianhao/hsmot_code/MeMOTR/debug/encoder_tensor_maps",
                        help="Directory to save heatmaps.")

    # Forward data source (reuse submit-style dataset layout)
    parser.add_argument("--data-root", type=str, help="DATA_ROOT from training.", default='/data/users/litianhao/hsmot_code/data')
    parser.add_argument("--dataset-name", type=str,
                        help="Dataset name (e.g., hsmot_8ch). Will be normalized as in submit_engine if needed.", default='hsmot_8ch')
    parser.add_argument("--split", type=str, help="Split folder, e.g., val or test.", default='train')
    parser.add_argument("--seq", type=str, help="Sequence folder name to read first frame from.", default='all')
    parser.add_argument("--npy2rgb", action="store_true", help="Whether to convert npy to rgb as in SeqDataset.")

    # Hook targets
    parser.add_argument("--hook-modules", type=str, default="backbone.spectral_embedding.conv_list.0.3, backbone.spectral_embedding.conv_list.1.3, backbone.spectral_embedding.conv_list.2.3, backbone.spectral_embedding.conv_list.3.3",
                        help="Comma-separated dotted module names to register forward hooks on (under model core).")
    parser.add_argument("--list-modules", action="store_true", default=True,
                        help="List all available modules in the model for hooking.")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device, e.g., cuda:0 or cpu.")

    # Dump outputs
    parser.add_argument("--dump-json", type=str, default="",
                        help="If set, dump shallow metadata (shapes/dtypes) of results to this json file.")

    args = parser.parse_args()

    train_cfg_path = args.train_config
    assert os.path.isfile(train_cfg_path), f"Config not found: {train_cfg_path}"
    train_config = yaml_to_dict(path=train_cfg_path)

    print(f"[Info] Building model from: {train_cfg_path}")
    model = build_model(config=train_config)

    device = torch.device(args.device)
    model.to(device)

    print(f"[Info] Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model=model, path=args.checkpoint)

    # List all modules if requested
    if args.list_modules:
        print("\n===== Available Modules for Hooking =====")
        all_modules = list_all_modules(get_model(model))
        for i, module_name in enumerate(all_modules):
            print(f"{i+1:3d}. {module_name}")
        print(f"\nTotal: {len(all_modules)} modules found")
        if not args.hook_modules and not args.show_params:
            print("Use --hook-modules to specify which modules to hook, or --show-params to inspect parameters.")
            return

    # Show params
    show_param_patterns = [s.strip() for s in args.show_params.split(",") if s.strip()]
    all_named = collect_named_params_and_buffers(get_model(model))
    matched_names = match_keys(list(all_named.keys()), show_param_patterns)

    if matched_names:
        print("\n===== Matched Parameters/Buffers =====")
        for n in matched_names:
            t = all_named[n]
            if args.print_values:
                pretty_print_tensor(n, t, save_csv=args.save_csv, output_dir=args.csv_output_dir)
            else:
                print(f"{n}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}, numel={t.numel()}")
                if args.save_csv:
                    csv_path = save_tensor_to_csv(n, t, args.csv_output_dir)
                    if csv_path:
                        print(f"  saved to: {csv_path}")
    else:
        if show_param_patterns:
            print(f"[Warn] No params/buffers matched patterns: {show_param_patterns}")

    # Prepare seq_dir
    dataset_name = args.dataset_name
    base_name = dataset_name.replace("_8ch", "") if "hsmot" in dataset_name else dataset_name
    npy_root = os.path.join(args.data_root, base_name, args.split, 'npy')
    rgb_root = os.path.join(args.data_root, base_name, args.split, 'rgb')

    if args.seq.lower() == "all":
        assert os.path.isdir(npy_root), f"npy root not found: {npy_root}"
        seq_list = sorted([d for d in os.listdir(npy_root) if os.path.isdir(os.path.join(npy_root, d))])
    else:
        seq_list = [s.strip() for s in args.seq.split(",") if s.strip()]

    # seq_dir = os.path.join(args.data_root, base_name, args.split, 'npy', args.seq)
    # assert os.path.isdir(seq_dir), f"Sequence dir not found: {seq_dir}"

    hook_names = [s.strip() for s in args.hook_modules.split(",") if s.strip()]

    all_meta: Dict[str, Any] = {"per_seq": []}

    for seq in seq_list:
        seq_dir = os.path.join(npy_root, seq)
        assert os.path.isdir(seq_dir), f"Sequence dir not found: {seq_dir}"

        print(f"\n[Info] Running forward once... seq={seq}")
        outputs, captured, _ = run_forward_once(model=model,
                                                        seq_dir=seq_dir,
                                                        npy2rgb=args.npy2rgb,
                                                        hook_module_names=hook_names,
                                                        args=args)

        # Per-seq output dirs
        csv_out_dir = os.path.join(args.csv_output_dir, seq)
        map_out_dir = os.path.join(args.heatmap_dir, seq)
        os.makedirs(csv_out_dir, exist_ok=True)
        os.makedirs(map_out_dir, exist_ok=True)

        # Save the RGB reference image (first frame) next to heatmaps
        try:
            rgb_path = os.path.join(map_out_dir, f"{seq}__rgb_firstframe.png")
            rgb_img = os.path.join(rgb_root, seq, '000001.png')
            # just copy the rgb_img to rgb_path
            shutil.copy(rgb_img, rgb_path)
            print(f"[Info] Saved RGB image: {rgb_path}")
            # ori_image = cv2.imread(rgb_img)
            # # Ensure RGB correct order (H,W,3)
            # if ori_image.ndim == 3 and ori_image.shape[-1] in (3, 4):
            #     cv2.imwrite(rgb_path, ori_image[..., :3])
            #     print(f"[Info] Saved RGB image: {rgb_path}")
            # else:
            #     # fallback via matplotlib
            #     plt.figure()
            #     if ori_image.ndim == 2:
            #         plt.imshow(ori_image, cmap="gray")
            #     else:
            #         plt.imshow(ori_image)
            #     plt.axis('off')
            #     plt.tight_layout(pad=0)
            #     plt.savefig(rgb_path, bbox_inches="tight", pad_inches=0, dpi=200)
            #     plt.close()
            #     print(f"[Info] Saved RGB image via plt: {rgb_path}")
        except Exception as e:
            print(f"[Warn] Failed to save RGB image for seq {seq}: {e}")

        # Print model outputs (shallow)
        print("\n===== Model Outputs (shallow) =====")
        seq_meta: Dict[str, Any] = {"seq": seq, "outputs": {}, "hooks": {}}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"out[{k}]: shape={{list}} dtype={{dtype}} device={{dev}}".format(
                        list=list(v.shape), dtype=v.dtype, dev=v.device))
                    seq_meta["outputs"][k] = {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
                else:
                    try:
                        desc = str(type(v))
                    except Exception:
                        desc = "unknown"
                    print(f"out[{k}]: type={desc}")
                    seq_meta["outputs"][k] = {"type": desc}
        else:
            desc = str(type(outputs))
            print(f"outputs type: {desc}")
            seq_meta["outputs"]["__type__"] = desc

        # Handle captured hooks
        if captured:
            print("\n===== Captured Hook Tensors =====")
            for k, t in captured.items():
                print(f"hook[{k}]: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}")
                seq_meta["hooks"][k] = {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}

                # Save per-hook tensors
                # safe_hook = _safe_name(k)
                safe_hook = k

                # If tensor is 1x8xMxN -> split channels
                if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 8:
                    with torch.no_grad():
                        t_np = t.detach().cpu().numpy()  # (1,8,M,N)
                        for c in range(8):
                            arr2d = t_np[0, c]
                            # CSV (2D)
                            csv_name = f"{seq}__{safe_hook}__ch{c}.csv"
                            csv_path = os.path.join(csv_out_dir, csv_name)
                            # save_2d_to_csv(csv_path, arr2d)
                            # Heatmap PNG (0..1 clamp)
                            if args.save_heatmaps:
                                png_name = f"{seq}__{safe_hook}__ch{c}.png"
                                png_path = os.path.join(map_out_dir, png_name)
                                save_heatmap(arr2d, png_path, vmin=0.0, vmax=1.0, add_colorbar=False)
                else:
                    # Fallback: save as original util (may flatten)
                    if args.save_csv:
                        csv_path = save_tensor_to_csv(f"{seq}__hook_{safe_hook}", t, csv_out_dir)
                        if csv_path:
                            print(f"  saved to: {csv_path}")

        else:
            if hook_names:
                print("[Warn] No hook outputs captured. The modules may not produce tensor outputs or names mismatch.")

        all_meta["per_seq"].append(seq_meta)

    # Dump combined metadata json if requested
    if args.dump_json:
        try:
            with open(args.dump_json, "w") as f:
                json.dump(all_meta, f, indent=2)
            print(f"[Info] Dumped metadata json to {args.dump_json}")
        except Exception as e:
            print(f"[Warn] Failed to dump json: {e}")



if __name__ == "__main__":
    main()
