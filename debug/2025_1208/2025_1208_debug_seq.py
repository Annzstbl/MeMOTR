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
from utils.nested_tensor import NestedTensor

# Allow running from project root or this file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.utils import load_checkpoint, get_model
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict
from utils.nested_tensor import tensor_list_to_nested_tensor
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset
import matplotlib.pyplot as plt
import cv2
from utils.utils import load_yaml_with_inheritance
from utils.GMC import compute_gmc_sequence
from sklearn.decomposition import PCA
from hsmot.datasets.pipelines.channel import rotate_norm_boxes_to_boxes, HeatmapFromRotateGt
from hsmot.mmlab.hs_mmrotate import obb2poly

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
    
def hook_apply_posterior_enhance(model: nn.Module, captured_list: List[Dict], frame_idx_ref: List[int]):
    """
    通过monkey patching来hook apply_posterior_enhance的结果
    """
    inner_model = get_model(model)
    if not hasattr(inner_model, 'scem_module') or inner_model.scem_module is None:
        return None
    
    from models.SCEM import SCEM
    original_apply = SCEM.apply_posterior_enhance
    
    def wrapped_apply_posterior_enhance(features, masks, scem_out, alpha: float = 0.5):
        result = original_apply(features, masks, scem_out, alpha)
        # 保存结果
        frame_idx = frame_idx_ref[0]
        while len(captured_list) <= frame_idx:
            captured_list.append({})
        # 保存每个尺度的增强特征
        for i, feat in enumerate(result):
            key = f"scem_module.apply_posterior_enhance_lvl{i}"
            captured_list[frame_idx][key] = feat.detach().clone()
        return result
    
    # 替换方法
    SCEM.apply_posterior_enhance = staticmethod(wrapped_apply_posterior_enhance)
    return original_apply

def restore_apply_posterior_enhance(original_apply):
    """恢复原始的apply_posterior_enhance方法"""
    from models.SCEM import SCEM
    SCEM.apply_posterior_enhance = original_apply

# Hook函数注册表：字符串 -> (hook_function, restore_function)
HOOK_FUNCTIONS_REGISTRY = {
    "apply_posterior_enhance": (hook_apply_posterior_enhance, restore_apply_posterior_enhance),
    # 可以在这里添加更多的hook函数
    # "other_function": (hook_other_function, restore_other_function),
}

def run_forward_sequential(model: nn.Module,
                           seq_dir: str,
                           npy2rgb: bool,
                           hook_module_names: List[str],
                           num_frames: int,
                           start_frame: int,
                           args) -> List[Tuple[Dict[str, Any], Dict[str, torch.Tensor], np.ndarray]]:
    """
    序贯处理多帧，保持tracks状态。
    返回每帧的 (outputs, captured, ori_image) 列表。
    
    Args:
        start_frame: 起始帧号（1-indexed，即从1开始计数）
    """
    device = next(model.parameters()).device

    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {seq_dir}")
    
    # 转换为0-indexed
    start_idx = max(0, start_frame - 1)
    if start_idx >= len(dataset):
        raise RuntimeError(f"Start frame {start_frame} exceeds dataset length {len(dataset)}")
    
    # 计算实际可处理的帧数
    available_frames = len(dataset) - start_idx
    num_frames = min(num_frames, available_frames)
    
    print(f"[Info] Processing {num_frames} frames from sequence (starting from frame {start_frame}, total: {len(dataset)})")

    # 使用共享的字典来存储每帧的captured结果
    all_captured: List[Dict[str, torch.Tensor]] = []
    current_frame_idx = [0]  # 使用列表以便在hook中修改

    # 注册hooks
    def extract_tensors_recursive(obj, base_key: str, frame_idx: int, captured_list: List[Dict]):
        """
        递归提取嵌套结构中的所有tensors，并在key中添加结构前缀
        
        Args:
            obj: 要解析的对象（可能是Tensor, list, tuple, dict等）
            base_key: 基础key名称
            frame_idx: 当前帧索引
            captured_list: 保存结果的列表
        """
        # 确保captured_list有足够的空间
        while len(captured_list) <= frame_idx:
            captured_list.append({})
        
        if isinstance(obj, torch.Tensor):
            # 直接是Tensor，保存
            captured_list[frame_idx][base_key] = obj.detach().clone()
        elif isinstance(obj, (list, tuple)):
            # 是list或tuple，递归处理每个元素
            struct_type = "list" if isinstance(obj, list) else "tuple"
            for idx, item in enumerate(obj):
                item_key = f"{base_key}__{struct_type}{idx}"
                extract_tensors_recursive(item, item_key, frame_idx, captured_list)
        elif isinstance(obj, dict):
            # 是dict，递归处理每个值
            for k, v in obj.items():
                # 使用dict的key作为标识，确保key是字符串
                dict_key_str = str(k).replace(".", "_").replace("/", "_")
                item_key = f"{base_key}__dict_{dict_key_str}"
                extract_tensors_recursive(v, item_key, frame_idx, captured_list)
        elif isinstance(obj, NestedTensor):
            captured_list[frame_idx][base_key] = obj.decompose()[0].detach().clone()
            base_key_mask = base_key + "__mask"
            captured_list[frame_idx][base_key_mask] = obj.decompose()[1].detach().clone()
        else:
            # 其他类型，打印记录
            obj_type = type(obj).__name__
            obj_str = str(obj)[:100] if len(str(obj)) > 100 else str(obj)  # 限制长度避免过长
            print(f"[HookInfo] Frame {frame_idx}, Key '{base_key}': Unsupported type '{obj_type}', value: {obj_str}")
    
    handles = []
    if hook_module_names:
        modules = find_modules_by_names(get_model(model), hook_module_names)
        if len(modules) == 0:
            print(f"[Warn] No modules matched for hooks (exact paths expected): {hook_module_names}")
        for nm, m in modules.items():
            def _make_hook(key: str, frame_idx_ref, captured_list):
                def _hook(_mod, _inp, _out):
                    try:
                        frame_idx = frame_idx_ref[0]
                        # 递归提取所有tensors
                        extract_tensors_recursive(_out, key, frame_idx, captured_list)
                    except Exception as e:
                        print(f"[HookError] {key}: {e}")
                return _hook
            hook_func = _make_hook(nm, current_frame_idx, all_captured)
            handles.append(m.register_forward_hook(hook_func))
    
    # Hook函数（根据字符串参数注册）
    hook_function_restores = {}  # 保存每个hook函数的恢复函数
    hook_function_names = [s.strip() for s in args.hook_functions.split(",") if s.strip()]
    for func_name in hook_function_names:
        if func_name not in HOOK_FUNCTIONS_REGISTRY:
            available = ", ".join(HOOK_FUNCTIONS_REGISTRY.keys())
            raise ValueError(f"Unknown hook function '{func_name}'. Available functions: {available}")
        
        hook_func, restore_func = HOOK_FUNCTIONS_REGISTRY[func_name]
        original = hook_func(model, all_captured, current_frame_idx)
        if original is not None:
            hook_function_restores[func_name] = (restore_func, original)
            print(f"[Info] Hooked function '{func_name}' to capture results")

    model.eval()
    inner_model = get_model(model)
    TrackInstances.set_static_properties(use_spectral_decoder=inner_model.decoder_spectral, decoder_spectral_weights_dim=inner_model.decoder_spectral_clusters * 8)
    
    # 创建 RuntimeTracker（参考 submit_engine.py）
    use_dab = getattr(inner_model, "use_dab", False)
    decoder_spectral = getattr(inner_model, "decoder_spectral", True)
    tracker = RuntimeTracker(
        det_score_thresh=0.5,
        track_score_thresh=0.5,
        miss_tolerance=30,
        use_motion=False,
        motion_min_length=3,
        motion_max_length=5,
        visualize=False,
        use_dab=use_dab,
        decoder_spectral=decoder_spectral
    )
    
    # 初始化 tracks
    tracks = [TrackInstances(
        hidden_dim=inner_model.hidden_dim,
        num_classes=inner_model.num_classes,
        use_dab=use_dab
    ).to(device)]
    
    # 检查是否需要使用 prior_map 和 GMC
    use_prior_map = False
    if hasattr(inner_model, 'scem_module'):
        if inner_model.scem_module.prior_mode is not None:
            use_prior_map = True
    
    results = []
    prev_frame = None
    
    with torch.no_grad():
        for i in range(num_frames):
            # 计算实际的数据集索引（考虑起始帧）
            dataset_idx = start_idx + i
            # 计算显示的帧号（1-indexed，从start_frame开始）
            frame_num = start_frame + i
            
            print(f"[Info] Processing frame {frame_num} (dataset index {dataset_idx})")
            
            # 更新当前帧索引（用于hook，相对于处理帧的索引）
            current_frame_idx[0] = i
            
            # 获取当前帧
            image, ori_image = dataset[dataset_idx][0]  # ((image, ori_image), info)
            frame = tensor_list_to_nested_tensor([image]).to(device)
            
            # 计算 GMC（如果需要）
            gmc = None
            if use_prior_map:
                if prev_frame is not None:
                    gmc = compute_gmc_sequence(images=[prev_frame[0], frame.tensors[0]], method='sparseOptFlow', downscale=1)[-1]  # [2, 3]
                else:
                    gmc = np.eye(2, 3, dtype=np.float32)
                prev_frame = frame.tensors.detach().clone()
                gmc = torch.tensor(gmc, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 2, 3]
            
            # 前向传播
            if gmc is not None:
                res = model(frame=frame, tracks=tracks, gmc=gmc, debug=True)
            else:
                res = model(frame=frame, tracks=tracks, debug=True)
            
            # 使用 tracker 更新 tracks（参考 submit_engine.py）
            previous_tracks, new_tracks = tracker.update(
                model_outputs=res,
                tracks=tracks
            )
            tracks: List[TrackInstances] = inner_model.postprocess_single_frame(previous_tracks, new_tracks, None)
            
            # #TODO 重新初始化tracks
            # tracks = [TrackInstances(
            #     hidden_dim=inner_model.hidden_dim,
            #     num_classes=inner_model.num_classes,
            #     use_dab=use_dab
            # ).to(device)]

            # 收集当前帧的captured结果
            captured: Dict[str, torch.Tensor] = {}
            if i < len(all_captured):
                captured = all_captured[i]
            
            # 确保ori_image是numpy uint8 HxWx3
            if torch.is_tensor(ori_image):
                ori_image = ori_image.detach().cpu().numpy()
            if ori_image.dtype != np.uint8:
                ori_image = np.clip(ori_image, 0, 255).astype(np.uint8)
            
            # 保存 model 的输出（res）而不是原始的 outputs
            results.append((res, captured, ori_image, image.shape))

    for h in handles:
        h.remove()
    
    # 恢复所有hook的函数
    for func_name, (restore_func, original) in hook_function_restores.items():
        restore_func(original)
        print(f"[Info] Restored function '{func_name}'")

    return results

def save_heatmap(arr: np.ndarray, filepath: str, vmin: float = None, vmax: float = None, 
                 add_colorbar: bool = False, max_channels: int = 8, use_pca: bool = False) -> None:
    """
    保存heatmap图像，支持多通道和PCA降维
    
    Args:
        arr: 输入数组，可以是2D (H, W) 或 3D (H, W, C) 或 4D (1, C, H, W)
        filepath: 保存路径，如果是多通道且不分通道绘制，直接使用；否则会自动添加通道后缀
        vmin: 最小值，如果为None则自动计算
        vmax: 最大值，如果为None则自动计算
        add_colorbar: 是否添加颜色条
        max_channels: 最大通道数（当分通道绘制时）
        use_pca: 是否使用PCA降维（当通道数>3时）
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 处理不同维度的输入
    if arr.ndim == 4:
        # (1, C, H, W) -> (C, H, W)
        arr = arr[0]
    elif arr.ndim == 2:
        # (H, W) -> (1, H, W) 添加通道维度
        arr = arr[np.newaxis, :, :]
    
    # 现在arr应该是 (C, H, W) 或 (H, W) 如果是2D则已转换为(1, H, W)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    
    C, H, W = arr.shape
    
    # 计算vmin和vmax
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    
    # 如果通道数 <= 3，直接绘制RGB图像
    if C <= 3:
        if C == 1:
            # 单通道：使用colormap渲染
            arr2d = arr[0]  # (H, W)
            arr2d_normalized = np.clip((arr2d - vmin) / (vmax - vmin + 1e-8), 0, 1)
            
            plt.figure()
            plt.imshow(arr2d_normalized, vmin=0, vmax=1, cmap='viridis')
            if add_colorbar:
                plt.colorbar()
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=200)
            plt.close()
        elif C == 2:
            # 2通道：前两个通道 + 零通道
            rgb = np.stack([arr[0], arr[1], np.zeros((H, W))], axis=0)
            # 转换为 (H, W, 3) 并归一化
            rgb = rgb.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            rgb = np.clip((rgb - vmin) / (vmax - vmin + 1e-8), 0, 1)
            
            plt.figure()
            plt.imshow(rgb, vmin=0, vmax=1)
            if add_colorbar:
                plt.colorbar()
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=200)
            plt.close()
        else:  # C == 3
            # 3通道：直接使用RGB
            rgb = arr
            # 转换为 (H, W, 3) 并归一化
            rgb = rgb.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            rgb = np.clip((rgb - vmin) / (vmax - vmin + 1e-8), 0, 1)
            
            plt.figure()
            plt.imshow(rgb, vmin=0, vmax=1)
            if add_colorbar:
                plt.colorbar()
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=200)
            plt.close()
    
    # 如果通道数 > 3
    elif C > 3:
        if use_pca:
            # 使用PCA降维到3通道
            arr = np.clip(arr, vmin, vmax)
            # 将 (C, H, W) 重塑为 (H*W, C)
            arr_flat = arr.transpose(1, 2, 0).reshape(-1, C)  # (H*W, C)
            
            # PCA降维到3维
            pca = PCA(n_components=3)
            arr_pca = pca.fit_transform(arr_flat)  # (H*W, 3)
            
            # 重塑回 (H, W, 3)
            rgb = arr_pca.reshape(H, W, 3)
            
            # 归一化到0-1
            rgb_min = rgb.min()
            rgb_max = rgb.max()
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
            
            plt.figure()
            plt.imshow(rgb, vmin=0, vmax=1)
            if add_colorbar:
                plt.colorbar()
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=200)
            plt.close()
        else:
            # 分通道绘制
            base_path = filepath.rsplit('.', 1)[0]  # 去掉扩展名
            ext = filepath.rsplit('.', 1)[1] if '.' in filepath else 'png'
            
            for c in range(min(max_channels, C)):
                arr2d = arr[c]
                arr2d_clipped = np.clip((arr2d - vmin) / (vmax - vmin + 1e-8), 0, 1)
                
                ch_filepath = f"{base_path}__ch{c}.{ext}"
                plt.figure()
                plt.imshow(arr2d_clipped, vmin=0, vmax=1, cmap='viridis')
                if add_colorbar:
                    plt.colorbar()
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(ch_filepath, bbox_inches="tight", pad_inches=0, dpi=200)
                plt.close()


def main():
    parser = argparse.ArgumentParser(description="Debug utility: load checkpoint, inspect params, forward once and capture outputs.")

    parser.add_argument("--train-config", type=str,
                        help="Path to train/config.yaml or any config yaml compatible with build_model.",
                        default='/data/users/wangying01/lth/hsmot/MeMOTR/debug/2025_1208/18_merge6.yaml')
    parser.add_argument("--checkpoint", type=str, help="Path to .pth checkpoint to load.",
    default='/data4/litianhao/hsmot/memotr/spectralemb/18_half_priormap_99/checkpoint_15.pth'
    )

    # Parameter inspection
    parser.add_argument("--show-params", type=str, default="",
                        help="Comma-separated substrings to match parameter/buffer names to display.")
    parser.add_argument("--print-values", action="store_true", default=True,
                        help="Print actual values (head) of tensors; otherwise only shape/dtype.")
    parser.add_argument("--save-csv", action="store_true", default=False,
                        help="Save tensors to CSV files.")
    parser.add_argument("--csv-output-dir", type=str, default="csvs",
                        help="Directory to save CSV files (default: ./tensor_csvs).")

    # >>> NEW: heatmap saving options
    parser.add_argument("--save-heatmaps", action="store_true", default=True,
                        help="Whether to save 2D heatmaps of hook outputs (expects 1x8xMxN).")
    parser.add_argument("--heatmap-dir", type=str, default="maps",
                        help="Directory to save heatmaps.")
    parser.add_argument("--max-channels", type=int, default=2,
                        help="Maximum number of channels to save for feature maps/heatmaps (default: 8).")
    parser.add_argument("--heatmap-use-pca", action="store_true", default=True,
                        help="Use PCA to reduce multi-channel (>3) features to 3 channels for visualization. If False, split channels.")
    parser.add_argument("--heatmap-outputs", type=str, default="scem_gamma,scem_log_mix,scores",
                        help="Comma-separated list of model output keys to save as heatmaps (default: 'scem_gamma,scem_log_mix').")
    
    # Project name for organizing outputs
    parser.add_argument("--output-dir", type=str, default="/data/users/wangying01/lth/hsmot/MeMOTR/debug",
                        help="Directory to save outputs.")
    parser.add_argument("--project-name", type=str, default="2025_1208/train",
                        help="Project name to organize outputs. Results will be saved under {output_dir}/{project_name}/ (default: empty, saves directly under output_dir).")

    # Forward data source (reuse submit-style dataset layout)
    parser.add_argument("--data-root", type=str, help="DATA_ROOT from training.", default='/data/users/wangying01/lth/hsmot/data')
    parser.add_argument("--dataset-name", type=str,
                        help="Dataset name (e.g., hsmot_8ch). Will be normalized as in submit_engine if needed.", default='hsmot_8ch')
    parser.add_argument("--split", type=str, help="Split folder, e.g., val or test.", default='train')
    parser.add_argument("--seq", type=str, help="Sequence folder name to read first frame from.", default='data36-7')
    parser.add_argument("--npy2rgb", action="store_true", help="Whether to convert npy to rgb as in SeqDataset.")
    parser.add_argument("--start-frame", type=int, default=1,
                        help="Starting frame number (1-indexed). Process frames starting from this number (default: 1).")
    #TODO
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Number of consecutive frames to process sequentially (default: 1).")

    # Hook targets
    parser.add_argument("--hook-modules", type=str, default="",
    # parser.add_argument("--hook-modules", type=str, default="backbone.backbone.stem_conv",
    # parser.add_argument("--hook-modules", type=str, default="backbone.backbone.stem_conv.conv3d, backbone.backbone.stem_conv.se_sig",
    # parser.add_argument("--hook-modules", type=str, default="feature_projs.0, feature_projs.1, feature_projs.2, feature_projs.3",
    # parser.add_argument("--hook-modules", type=str, default="transformer.encoder, backbone",
    # parser.add_argument("--hook-modules", type=str, default="scem_module.posterior.pi_head, scem_module.posterior.gate_head, scem_module.posterior.mu_b_head, scem_module.posterior.logsig_b_head, transformer.encoder",
    # scem_module.posterior.mu_f_head, scem_module.posterior.logsig_f_head,",
    # parser.add_argument("--hook-modules", type=str, default="scem_module,",
                        help="Comma-separated dotted module names to register forward hooks on (under model core).")
    parser.add_argument("--hook-functions", type=str, default="",
    # parser.add_argument("--hook-functions", type=str, default="apply_posterior_enhance",
                        help="Comma-separated function names to hook (e.g., 'apply_posterior_enhance'). Available functions: " + ", ".join(HOOK_FUNCTIONS_REGISTRY.keys()))
    parser.add_argument("--list-modules", action="store_true", default=True,
                        help="List all available modules in the model for hooking.")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device, e.g., cuda:0 or cpu.")

    # Dump outputs
    parser.add_argument("--dump-json", type=str, default="",
                        help="If set, dump shallow metadata (shapes/dtypes) of results to this json file.")

    args = parser.parse_args()

    # output dir
    output_dir = os.path.join(args.output_dir, args.project_name)
    os.makedirs(output_dir, exist_ok=True)
    csv_output_dir = os.path.join(output_dir, args.csv_output_dir)
    os.makedirs(csv_output_dir, exist_ok=True)
    heatmap_dir = os.path.join(output_dir, args.heatmap_dir)
    os.makedirs(heatmap_dir, exist_ok=True)

    train_cfg_path = args.train_config
    assert os.path.isfile(train_cfg_path), f"Config not found: {train_cfg_path}"
    train_config = load_yaml_with_inheritance(path=train_cfg_path)

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
            # return

    # Show params
    show_param_patterns = [s.strip() for s in args.show_params.split(",") if s.strip()]
    all_named = collect_named_params_and_buffers(get_model(model))
    matched_names = match_keys(list(all_named.keys()), show_param_patterns)


    if matched_names:
        # Prepare output dir for params (with project name if specified)
        param_output_dir = csv_output_dir
        if args.project_name:
            param_output_dir = csv_output_dir
            os.makedirs(param_output_dir, exist_ok=True)
        
        print("\n===== Matched Parameters/Buffers =====")
        for n in matched_names:
            t = all_named[n]
            if args.print_values:
                pretty_print_tensor(n, t, save_csv=args.save_csv, output_dir=param_output_dir)
            else:
                print(f"{n}: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}, numel={t.numel()}")
                if args.save_csv:
                    csv_path = save_tensor_to_csv(n, t, param_output_dir)
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

        print(f"\n[Info] Running forward sequential ({args.num_frames} frames starting from frame {args.start_frame})... seq={seq}")
        frame_results = run_forward_sequential(model=model,
                                               seq_dir=seq_dir,
                                               npy2rgb=args.npy2rgb,
                                               hook_module_names=hook_names,
                                               num_frames=args.num_frames,
                                               start_frame=args.start_frame,
                                               args=args)

        # Per-seq output dirs
        csv_out_dir = os.path.join(csv_output_dir, seq)
        map_out_dir = os.path.join(heatmap_dir, seq)
        os.makedirs(csv_out_dir, exist_ok=True)
        os.makedirs(map_out_dir, exist_ok=True)

        seq_meta: Dict[str, Any] = {"seq": seq, "frames": []}

        # 处理每一帧的结果
        for frame_idx, (outputs, captured, ori_image, img_shapes) in enumerate(frame_results):
            # 计算实际帧号（从start_frame开始）
            frame_num = args.start_frame + frame_idx
            frame_str = f"f{frame_num:06d}"  # 格式化为 f000001, f000002, ...
            
            print(f"\n===== Frame {frame_num} (processing {frame_idx + 1}/{len(frame_results)}) =====")
            
            # Save the RGB reference image for this frame
            try:
                rgb_path = os.path.join(map_out_dir, f"{seq}__rgb_{frame_str}.png")
                # 使用ori_image直接保存
                if ori_image.ndim == 3 and ori_image.shape[-1] == 3:
                    cv2.imwrite(rgb_path, cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR))
                else:
                    # 提取235通道
                    cv2.imwrite(rgb_path, ori_image[:, :, [4,2,1]])
                    
                print(f"[Info] Saved RGB image: {rgb_path}")
            except Exception as e:
                print(f"[Warn] Failed to save RGB image for frame {frame_num}: {e}")

            # Parse heatmap outputs from arguments
            heatmap_outputs = [s.strip() for s in args.heatmap_outputs.split(",") if s.strip()]
            # Print model outputs (shallow)
            print(f"\n===== Model Outputs (Frame {frame_num}) =====")
            frame_meta: Dict[str, Any] = {"frame": frame_num, "outputs": {}, "hooks": {}}
            
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"out[{k}]: shape={{list}} dtype={{dtype}} device={{dev}}".format(
                            list=list(v.shape), dtype=v.dtype, dev=v.device))
                        frame_meta["outputs"][k] = {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
                        if k in heatmap_outputs:
                            v_np = v.detach().cpu().numpy()
                            save_heatmap(v_np, os.path.join(map_out_dir, f"{seq}__heatmap_{k}__{frame_str}.png"), 
                                       vmin=v_np.min(), vmax=v_np.max(), add_colorbar=True,
                                       max_channels=args.max_channels, use_pca=args.heatmap_use_pca)
                    else:
                        try:
                            desc = str(type(v))
                        except Exception:
                            desc = "unknown"
                        print(f"out[{k}]: type={desc}")
                        frame_meta["outputs"][k] = {"type": desc}
            else:
                desc = str(type(outputs))
                print(f"outputs type: {desc}")
                frame_meta["outputs"]["__type__"] = desc

            # 可视化 scores & pred_bboxes（旋转框），画在原图上
            if 'scores' in heatmap_outputs:
                try:
                    if isinstance(outputs, dict) and ('scores' in outputs) and ('pred_bboxes' in outputs):
                        scores_tensor = outputs['pred_logits'].sigmoid()      # 形状: [B, N, C] 或 [B, N]
                        bboxes_tensor = outputs['pred_bboxes'] # 形状: [B, N, 5] (le135旋转框)

                        # 只处理 batch=1 的情况
                        if scores_tensor.dim() == 3:
                            # 对类别维取最大值作为该query的score: [B, N]
                            scores_max = torch.max(scores_tensor, dim=-1).values
                        else:
                            scores_max = scores_tensor

                        scores_np = scores_max[0].detach().cpu().numpy()    # (N,)
                        bboxes_np = bboxes_tensor[0].detach().cpu().numpy() # (N,5)

                        N = bboxes_np.shape[0]
                        # 第一部分为检测query，后面为track query，默认分界为300
                        det_query_num = 300
                        if N < det_query_num:
                            det_query_num = N

                        # 将归一化旋转框转换为像素坐标 (xywha 和 多边形)
                        H, W = img_shapes[1], img_shapes[2]
                        boxes_tensor = torch.from_numpy(bboxes_np).to(torch.float32)  # (N,5) 归一化
                        boxes_xywha = rotate_norm_boxes_to_boxes(boxes_tensor, (H, W), version='le135')  # (N,5) 像素坐标
                        boxes_xyxyxyxy = obb2poly(boxes_xywha)  # (N, 8)

                        # 复制一份原图用于绘制
                        vis_img = ori_image.copy()
                        if vis_img.dtype != np.uint8:
                            vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
                        # 如果是8通道，提取RGB通道
                        if vis_img.shape[2] == 8:
                            vis_img = np.ascontiguousarray(vis_img[:, :, [4, 2, 1]])
                            

                        # # 1) 先根据scores和旋转框生成heatmap，并叠加到图像上
                        # try:
                        #     device = boxes_xywha.device
                        #     # 构造单框heatmap并按score加权累加
                        #     heatmap = torch.zeros((H, W), dtype=torch.float32, device=device)
                        #     for i in range(N):
                        #         score_i = float(scores_np[i])
                        #         if score_i <= 0:
                        #             continue
                        #         box_i = boxes_xywha[i:i+1]  # (1,5)
                        #         hm_i = HeatmapFromRotateGt._heatmap_from_rotate_gt_xywha_fast(
                        #             box_i, (H, W), version='le135',
                        #             mode='fixed_peak', peak=score_i*255,
                        #             reduce='sum', chunk_boxes=1, k=5.0
                        #         )  # (H,W)
                        #         # 这里取最大值
                        #         heatmap = torch.maximum(heatmap, hm_i)


                        #     # 截断heat_color
                        #     heatmap = heatmap[:vis_img.shape[0], :vis_img.shape[1]]
                        #     heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)

                        #     # 叠加到原图上 (RGB)
                        #     alpha = 0.8
                        #     alpha_img = 0.3
                        #     vis_img = np.ascontiguousarray((alpha * heatmap + alpha_img * vis_img).astype(np.uint8))
                        # except Exception as e_hm:
                        #     print(f"[Warn] Failed to generate heatmap overlay for frame {frame_num}: {e_hm}")

                        # 2) 再画统一粗细的旋转框
                        for i, (poly, score) in enumerate(zip(boxes_xyxyxyxy, scores_np)):
                            poly_np = poly.detach().cpu().numpy() if isinstance(poly, torch.Tensor) else np.asarray(poly)
                            pts = poly_np.reshape(-1, 2).astype(np.float32)
                            pts_int = pts.reshape(-1, 1, 2).astype(np.int32)

                            # 根据是否为前300个query使用不同颜色
                            if i < det_query_num:
                                if score > 0.5:
                                    color = (0, 255, 0)
                                    thickness = 2
                                else:
                                    color = (0, 125, 0)   # 绿色：检测query  
                                    thickness = 1
                                # 线粗度统一
                                
                            else:
                                if score > 0.5:
                                    color = (255, 0, 0)
                                    thickness = 2
                                else:
                                    color = (125, 0, 0)
                                    thickness = 1

                            cv2.polylines(vis_img, [pts_int], isClosed=True, color=color, thickness=thickness)

                        det_vis_path = os.path.join(map_out_dir, f"{seq}__detections_{frame_str}.png")
                        # ori_image 是RGB，这里转换为BGR保存
                        cv2.imwrite(det_vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                        print(f"[Info] Saved detection visualization: {det_vis_path}")
                except Exception as e:
                    print(f"[Warn] Failed to draw detections for frame {frame_num}: {e}")

            # Handle captured hooks
            if captured:
                print(f"\n===== Captured Hook Tensors (Frame {frame_num}) =====")
                for k, t in captured.items():
                    print(f"hook[{k}]: shape={list(t.shape)}, dtype={t.dtype}, device={t.device}")
                    frame_meta["hooks"][k] = {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}

                    # Save per-hook tensors with frame identifier
                    safe_hook = k.replace(".", "_").replace("/", "_")
                    hook_base = safe_hook
                    hook_frame_str = frame_str

                    # If tensor is 1xCxMxN -> split channels
                    if t.dim() == 4 and t.shape[0] == 1:
                        with torch.no_grad():
                            t_np = t.detach().cpu().numpy()  # (1,C,M,N)
                            vmin = t_np.min()
                            vmax = t_np.max()
                            # CSV (2D) - 仍然分通道保存CSV
                            if args.save_csv:
                                for c in range(min(args.max_channels, t.shape[1])):
                                    arr2d = t_np[0, c]
                                    csv_name = f"{seq}__{hook_base}__{hook_frame_str}__ch{c}.csv"
                                    csv_path = os.path.join(csv_out_dir, csv_name)
                                    save_2d_to_csv(csv_path, arr2d)
                            # Heatmap PNG - 使用新的save_heatmap函数
                            if args.save_heatmaps:
                                png_path = os.path.join(map_out_dir, f"{seq}__{hook_base}__{hook_frame_str}.png")
                                save_heatmap(t_np, png_path, vmin=vmin, vmax=vmax, add_colorbar=True,
                                           max_channels=args.max_channels, use_pca=args.heatmap_use_pca)
                    elif t.dim() == 3 and k == "transformer.encoder":
                        # t = 1 * 24225 * 256
                        # lvl1 : 120*152
                        # lvl2 : 60*76
                        # lvl3 : 30*38
                        # lvl4 : 15*19
                        # 120*152 + 60*76 + 30*38 + 15*19 = 24225
                        t_np = t.detach().cpu().numpy()
                        lvl1 = t_np[:, :120*152, :].reshape(1, 120, 152, 256).transpose(0,3,1,2)
                        lvl2 = t_np[:, 120*152:120*152+60*76, :].reshape(1, 60, 76, 256).transpose(0,3,1,2)
                        lvl3 = t_np[:, 120*152+60*76:120*152+60*76+30*38, :].reshape(1, 30, 38, 256).transpose(0,3,1,2)
                        lvl4 = t_np[:, 120*152+60*76+30*38:120*152+60*76+30*38+15*19, :].reshape(1, 15, 19, 256).transpose(0,3,1,2)
                        lvls = [lvl1, lvl2, lvl3, lvl4]
                        vmin = t_np.min()
                        vmax = t_np.max()
                        for l, lvl in enumerate(lvls):
                            png_name = f"{seq}__{hook_base}_lvl{l+1}_{hook_frame_str}.png"
                            save_heatmap(lvl, os.path.join(map_out_dir, png_name), vmin=vmin, vmax=vmax, 
                                       add_colorbar=True, max_channels=args.max_channels, use_pca=args.heatmap_use_pca)
                    elif k == "backbone.backbone.stem_conv.conv3d":
                        t_np = t.detach().cpu().numpy()
                        # (1, 64, 8, H, W)
                        # 第三维度的每个通道分别存储
                        for c in range(t.shape[2]):
                            arr2d = t_np[0, c, :]
                            std = arr2d.std()
                            png_name = f"{seq}__{hook_base}_ch{c}__{hook_frame_str}.png"
                            save_heatmap(arr2d, os.path.join(map_out_dir, png_name), vmin=-3*std, vmax=3*std, add_colorbar=True,
                                       max_channels=args.max_channels, use_pca=args.heatmap_use_pca)
                    else:

                        # Fallback: save as original util (may flatten)
                        if args.save_csv:
                            csv_path = save_tensor_to_csv(f"{seq}__hook_{hook_base}__{hook_frame_str}", t, csv_out_dir)
                            if csv_path:
                                print(f"  saved to: {csv_path}")

            else:
                if hook_names:
                    print(f"[Warn] No hook outputs captured for frame {frame_num}. The modules may not produce tensor outputs or names mismatch.")

            seq_meta["frames"].append(frame_meta)

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
