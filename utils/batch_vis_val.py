#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量可视化多个实验文件夹的验证指标

用法示例:
    # 处理单个文件夹
    python MeMOTR/utils/batch_vis_val.py --paths /path/to/exp1
    
    # 处理多个文件夹
    python MeMOTR/utils/batch_vis_val.py --paths /path/to/exp1 /path/to/exp2 /path/to/exp3
    
    # 使用通配符
    python MeMOTR/utils/batch_vis_val.py --paths /path/to/exp*
    
    # 从文件读取路径列表
    python MeMOTR/utils/batch_vis_val.py --path_file paths.txt
    
    # 并行处理（使用4个进程）
    python MeMOTR/utils/batch_vis_val.py --paths /path/to/exp* --parallel --n_jobs 4
"""
import argparse
import os
import glob
import sys
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.vis_val import visualize_validation_metrics


def process_single_path(val_root_path: str, fig_path: Optional[str] = None) -> Dict:
    """
    处理单个路径的验证指标可视化
    
    Args:
        val_root_path: 验证数据根路径
        fig_path: 图片保存路径，如果为None则使用val_root_path/fig
    
    Returns:
        包含处理结果的字典
    """
    result = {
        'path': val_root_path,
        'success': False,
        'error': None,
        'best_result': None
    }
    
    try:
        if not os.path.exists(val_root_path):
            result['error'] = f"Path does not exist: {val_root_path}"
            return result
        
        # 检查是否有epoch文件夹
        epoch_folders = glob.glob(os.path.join(val_root_path, "epoch*"))
        if len(epoch_folders) == 0:
            result['error'] = f"No epoch folders found in: {val_root_path}"
            return result
        
        print(f"\n{'='*60}")
        print(f"Processing: {val_root_path}")
        print(f"{'='*60}")
        
        best_result = visualize_validation_metrics(val_root_path, fig_path)
        
        result['success'] = True
        result['best_result'] = best_result
        
        if best_result:
            print(f"✓ Success! Best epoch: {best_result['epoch']}, "
                  f"Combined Score: {best_result['combined_score']:.4f}")
        else:
            print(f"✓ Success! (No best result found)")
            
    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"✗ Error processing {val_root_path}: {str(e)}")
    
    return result


def load_paths_from_file(path_file: str) -> List[str]:
    """
    从文件中读取路径列表（每行一个路径）
    
    Args:
        path_file: 包含路径的文件路径
    
    Returns:
        路径列表
    """
    paths = []
    with open(path_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 忽略空行和注释
                paths.append(line)
    return paths


def expand_paths(paths: List[str]) -> List[str]:
    """
    展开路径列表，处理通配符
    
    Args:
        paths: 路径列表（可能包含通配符）
    
    Returns:
        展开后的路径列表
    """
    expanded_paths = []
    for path in paths:
        # 使用glob展开通配符
        matched = glob.glob(path)
        if matched:
            expanded_paths.extend(matched)
        else:
            # 如果没有匹配，保留原路径（可能是无效路径，后续会报错）
            expanded_paths.append(path)
    return expanded_paths


def process_sequential(paths: List[str], fig_paths: Optional[List[str]] = None) -> List[Dict]:
    """
    顺序处理多个路径
    
    Args:
        paths: 路径列表
        fig_paths: 对应的图片保存路径列表，如果为None则使用默认路径
    
    Returns:
        处理结果列表
    """
    results = []
    total = len(paths)
    
    for idx, path in enumerate(paths, 1):
        print(f"\n[{idx}/{total}] Processing: {path}")
        fig_path = fig_paths[idx-1] if fig_paths and idx-1 < len(fig_paths) else None
        result = process_single_path(path, fig_path)
        results.append(result)
    
    return results


def process_parallel(paths: List[str], n_jobs: int = 4, 
                     fig_paths: Optional[List[str]] = None) -> List[Dict]:
    """
    并行处理多个路径
    
    Args:
        paths: 路径列表
        n_jobs: 并行进程数
        fig_paths: 对应的图片保存路径列表，如果为None则使用默认路径
    
    Returns:
        处理结果列表
    """
    results = []
    total = len(paths)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # 提交所有任务
        future_to_path = {}
        for idx, path in enumerate(paths):
            fig_path = fig_paths[idx] if fig_paths and idx < len(fig_paths) else None
            future = executor.submit(process_single_path, path, fig_path)
            future_to_path[future] = (idx, path)
        
        # 收集结果（按完成顺序）
        completed = 0
        path_results = {}
        for future in as_completed(future_to_path):
            idx, path = future_to_path[future]
            completed += 1
            try:
                result = future.result()
                path_results[idx] = result
                print(f"\n[{completed}/{total}] Completed: {path} "
                      f"{'✓' if result['success'] else '✗'}")
            except Exception as e:
                path_results[idx] = {
                    'path': path,
                    'success': False,
                    'error': str(e),
                    'best_result': None
                }
                print(f"\n[{completed}/{total}] Failed: {path} ✗")
        
        # 按原始顺序返回结果
        results = [path_results[i] for i in range(total)]
    
    return results


def print_summary(results: List[Dict]):
    """
    打印处理结果汇总
    
    Args:
        results: 处理结果列表
    """
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    
    if success_count > 0:
        print(f"\n{'='*60}")
        print("BEST RESULTS")
        print(f"{'='*60}")
        print(f"{'Path':<50} {'Epoch':<8} {'Combined Score':<15}")
        print("-" * 75)
        
        for result in results:
            if result['success'] and result['best_result']:
                br = result['best_result']
                path_short = result['path'][-47:] if len(result['path']) > 47 else result['path']
                print(f"{path_short:<50} {br['epoch']:<8} {br['combined_score']:<15.4f}")
    
    if fail_count > 0:
        print(f"\n{'='*60}")
        print("FAILED PATHS")
        print(f"{'='*60}")
        for result in results:
            if not result['success']:
                print(f"{result['path']}")
                print(f"  Error: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='批量可视化多个实验文件夹的验证指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--paths',
        type=str,
        nargs='+',
        help='一个或多个实验文件夹路径（支持通配符）'
    )
    
    parser.add_argument(
        '--path_file',
        type=str,
        help='包含路径列表的文件（每行一个路径，支持#注释）'
    )
    
    parser.add_argument(
        '--fig_paths',
        type=str,
        nargs='+',
        help='对应的图片保存路径列表（可选，数量需与paths一致）'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='使用并行处理'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=4,
        help='并行处理的进程数（默认: 4）'
    )
    
    args = parser.parse_args()
    
    # 获取路径列表
    paths = []
    if args.path_file:
        if not os.path.exists(args.path_file):
            print(f"Error: Path file does not exist: {args.path_file}")
            return
        paths.extend(load_paths_from_file(args.path_file))
    
    if args.paths:
        paths.extend(args.paths)
    
    if not paths:
        parser.print_help()
        print("\nError: No paths provided. Use --paths or --path_file to specify paths.")
        return
    
    # 展开通配符
    paths = expand_paths(paths)
    
    if not paths:
        print("Error: No valid paths found after expansion.")
        return
    
    # 去重并排序
    paths = sorted(list(set(paths)))
    
    print(f"Found {len(paths)} path(s) to process:")
    for i, path in enumerate(paths, 1):
        print(f"  {i}. {path}")
    
    # 处理路径
    if args.parallel and len(paths) > 1:
        print(f"\nUsing parallel processing with {args.n_jobs} workers...")
        results = process_parallel(paths, args.n_jobs, args.fig_paths)
    else:
        print("\nUsing sequential processing...")
        results = process_sequential(paths, args.fig_paths)
    
    # 打印汇总
    print_summary(results)


if __name__ == "__main__":
    main()

