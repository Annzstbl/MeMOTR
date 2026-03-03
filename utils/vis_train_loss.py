#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化训练日志中的损失

用法示例:
    # 处理单个日志文件
    python MeMOTR/utils/vis_train_loss.py --log_file /path/to/log.txt --output_dir /path/to/output
    
    # 处理实验目录（自动查找log.txt）
    python MeMOTR/utils/vis_train_loss.py --exp_dir /path/to/experiment --output_dir /path/to/output
"""
import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from collections import defaultdict
from pathlib import Path
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn风格
sns.set_style("whitegrid")
sns.set_palette("husl")


def parse_log_file(log_file):
    """
    解析训练日志文件，提取所有损失值
    只处理最后一次训练的结果，跳过Epoch汇总行
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        tuple: (data字典, train_count训练次数)
    """
    # 第一步：找到所有训练开始的位置
    train_start_positions = []
    train_start_times = []
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # 查找训练开始标记：训练开始 Start Time: 2025-12-15 18:55:08
            # 匹配多种可能的格式
            if re.search(r'训练开始\s*Start Time:', line, re.IGNORECASE) or \
               re.search(r'Start Time:\s*\d{4}-\d{2}-\d{2}', line):
                train_start_positions.append(idx)
                # 提取时间信息：Start Time: 2025-12-15 18:55:08
                time_match = re.search(r'Start Time:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if time_match:
                    train_start_times.append(time_match.group(1))
                else:
                    # 如果没有完整时间，尝试只提取日期
                    date_match = re.search(r'Start Time:\s*(\d{4}-\d{2}-\d{2})', line)
                    if date_match:
                        train_start_times.append(date_match.group(1))
                    else:
                        train_start_times.append(f"Line {idx+1}")
    
    train_count = len(train_start_positions)
    
    # 确定要处理的起始行（最后一次训练）
    if train_count > 0:
        start_line_idx = train_start_positions[-1]
        print(f"Found {train_count} training session(s), using the last one (started at: {train_start_times[-1] if train_start_times else 'unknown'})")
    else:
        start_line_idx = 0
        print("No training start markers found, processing entire log file")
    
    # 第一步：扫描所有行，记录每个epoch的最大iteration数
    epoch_max_iter = {}  # {epoch: max_iteration}
    for line in lines[start_line_idx:]:
        line = line.strip()
        main_loss_match = re.search(r'--\[Epoch=(\d+),\s*Iter=(\d+)', line)
        if main_loss_match:
            epoch = int(main_loss_match.group(1))
            local_iter = int(main_loss_match.group(2))
            if epoch not in epoch_max_iter:
                epoch_max_iter[epoch] = local_iter
            else:
                epoch_max_iter[epoch] = max(epoch_max_iter[epoch], local_iter)
    
    # 计算每个epoch的起始全局iteration偏移
    epoch_iter_offsets = {}
    for epoch in sorted(epoch_max_iter.keys()):
        if epoch == 0:
            epoch_iter_offsets[epoch] = 0
        else:
            # 之前所有epoch的总iteration数 = sum(每个epoch的最大iter + 1)
            offset = sum(epoch_max_iter.get(e, 0) + 1 for e in range(epoch))
            epoch_iter_offsets[epoch] = offset
    
    data = {
        'iterations': [],  # 全局iteration（跨epoch累计）
        'local_iterations': [],  # 每个epoch内的iteration
        'epochs': [],
        'total_loss': [],
        'main_losses': defaultdict(list),  # 主损失（layer5）
        'layer_losses': defaultdict(lambda: defaultdict(list)),  # 各layer损失
        'class_losses': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),  # 按类别损失
        'scem_losses': defaultdict(list),  # SCEM相关损失
    }
    
    # 第二步：解析数据，使用预先计算的offset
    for line in lines[start_line_idx:]:
        line = line.strip()
        if not line:
            continue
        
        # 跳过Epoch结束的汇总行：--[Epoch: X, Total Time: ...]
        if re.match(r'--\[Epoch:\s*\d+,\s*Total Time:', line):
            continue
        
        # 跳过Epoch设置行：--Epoch=X Settings: ...
        if re.match(r'--Epoch=\d+\s+Settings:', line):
            continue
        
        # 跳过学习率行：[Epoch X] lr=...
        if re.match(r'\[Epoch\s+\d+\]\s+lr=', line):
            continue
        
        # 解析主损失行：--[Epoch=39, Iter=414, ...]
        main_loss_match = re.search(r'--\[Epoch=(\d+),\s*Iter=(\d+)', line)
        if main_loss_match:
            epoch = int(main_loss_match.group(1))
            local_iter = int(main_loss_match.group(2))
            
            # 计算全局iteration
            global_iter = epoch_iter_offsets.get(epoch, 0) + local_iter
            
            data['iterations'].append(global_iter)
            data['local_iterations'].append(local_iter)
            data['epochs'].append(epoch)
            
            # 提取总损失
            total_loss_match = re.search(r'loss\s*=\s*([\d.]+)', line)
            if total_loss_match:
                data['total_loss'].append(float(total_loss_match.group(1)))
            else:
                data['total_loss'].append(None)
            
            # 提取主损失（frame0_xxx_loss，不带layer的）
            # frame0_box_l1_loss = 0.0204 ( 0.0571,  0.0568)
            main_loss_patterns = [
                (r'frame\d+_box_l1_loss\s*=\s*([\d.]+)', 'box_l1_loss'),
                (r'frame\d+_box_giou_loss\s*=\s*([\d.]+)', 'box_giou_loss'),
                (r'frame\d+_label_focal_loss\s*=\s*([\d.]+)', 'label_focal_loss'),
                (r'frame\d+_scem_bce_loss\s*=\s*([\d.]+)', 'scem_bce_loss'),
                (r'frame\d+_scem_nll_loss\s*=\s*([\d.]+)', 'scem_nll_loss'),
                (r'frame\d+_scem_dice_loss\s*=\s*([\d.]+)', 'scem_dice_loss'),
            ]
            
            for pattern, key in main_loss_patterns:
                match = re.search(pattern, line)
                if match:
                    if 'scem' in key:
                        data['scem_losses'][key].append(float(match.group(1)))
                    else:
                        data['main_losses'][key].append(float(match.group(1)))
        
        # 解析detail_loss行
        if line.startswith('detail_loss'):
            # 提取所有损失项：frame0_box_l1_loss:0.0204
            detail_items = re.findall(r'(\S+):([\d.]+)', line)
            
            for item_name, value in detail_items:
                value = float(value)
                
                # 解析layer信息
                layer_match = re.search(r'aux_layer(\d+)', item_name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    layer_name = f'layer{layer_idx}'
                else:
                    layer_idx = 5  # 不带layer的等价于layer5
                    layer_name = 'layer5'
                
                # 解析损失类型和类别
                if '_class_' in item_name:
                    # 按类别损失：frame0_box_l1_loss_class_0
                    class_match = re.search(r'_class_(\d+)', item_name)
                    if class_match:
                        class_idx = int(class_match.group(1))
                        if 'box_l1_loss' in item_name:
                            loss_type = 'l1'
                        elif 'box_giou_loss' in item_name:
                            loss_type = 'giou'
                        else:
                            continue
                        data['class_losses'][layer_name][loss_type][class_idx].append(value)
                else:
                    # 普通损失：frame0_box_l1_loss, frame0_aux_layer0_box_l1_loss
                    if 'box_l1_loss' in item_name:
                        data['layer_losses'][layer_name]['l1'].append(value)
                    elif 'box_giou_loss' in item_name:
                        data['layer_losses'][layer_name]['giou'].append(value)
                    elif 'label_focal_loss' in item_name:
                        data['layer_losses'][layer_name]['focal'].append(value)
    
    # 确保所有列表长度一致
    n_iters = len(data['iterations'])
    for key in data['main_losses']:
        while len(data['main_losses'][key]) < n_iters:
            data['main_losses'][key].append(None)
    
    for layer_name in data['layer_losses']:
        for loss_type in data['layer_losses'][layer_name]:
            while len(data['layer_losses'][layer_name][loss_type]) < n_iters:
                data['layer_losses'][layer_name][loss_type].append(None)
    
    for layer_name in data['class_losses']:
        for loss_type in data['class_losses'][layer_name]:
            for class_idx in data['class_losses'][layer_name][loss_type]:
                while len(data['class_losses'][layer_name][loss_type][class_idx]) < n_iters:
                    data['class_losses'][layer_name][loss_type][class_idx].append(None)
    
    for key in data['scem_losses']:
        while len(data['scem_losses'][key]) < n_iters:
            data['scem_losses'][key].append(None)
    
    return data, train_count


def moving_average(values, window_size=50):
    """
    计算移动平均（窗口平滑）
    
    Args:
        values: 数值列表
        window_size: 窗口大小
    
    Returns:
        平滑后的数值列表
    """
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(values), i + window_size // 2 + 1)
        window = values[start_idx:end_idx]
        smoothed.append(np.mean(window))
    
    return smoothed


def exponential_smoothing(values, alpha=0.1):
    """
    计算指数平滑
    
    Args:
        values: 数值列表
        alpha: 平滑系数（0-1），越小越平滑
    
    Returns:
        平滑后的数值列表
    """
    if not values:
        return values
    
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    
    return smoothed


def add_epoch_separators(ax, data):
    """
    在图表上添加epoch分隔线和标签
    
    Args:
        ax: matplotlib axes对象
        data: 包含iterations和epochs的数据字典
    """
    if not data['iterations'] or not data['epochs']:
        return
    
    iterations = data['iterations']
    epochs = data['epochs']
    
    # 找到每个epoch的起始和结束位置
    epoch_boundaries = {}
    for i, (global_iter, epoch) in enumerate(zip(iterations, epochs)):
        if epoch not in epoch_boundaries:
            epoch_boundaries[epoch] = {'start': global_iter, 'start_idx': i}
        epoch_boundaries[epoch]['end'] = global_iter
        epoch_boundaries[epoch]['end_idx'] = i
    
    # 绘制epoch分隔线
    y_min, y_max = ax.get_ylim()
    for epoch in sorted(epoch_boundaries.keys())[1:]:  # 跳过第一个epoch
        boundary = epoch_boundaries[epoch]
        x_pos = boundary['start']
        ax.axvline(x=x_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        # 添加epoch标签
        ax.text(x_pos, y_max * 0.98, f'Epoch {epoch}', 
               rotation=90, verticalalignment='top', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def plot_total_loss(data, output_dir):
    """绘制总损失曲线（包含平滑曲线）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    iterations = data['iterations']
    total_loss = [v for v in data['total_loss'] if v is not None]
    valid_iters = [iterations[i] for i, v in enumerate(data['total_loss']) if v is not None]
    
    if total_loss:
        # 原始曲线
        ax.plot(valid_iters, total_loss, linewidth=1.5, label='Total Loss (raw)', 
               color='#1f77b4', alpha=0.5)
        
        # 移动平均
        ma_values = moving_average(total_loss, window_size=50)
        ax.plot(valid_iters, ma_values, linewidth=2, label='Moving Average (window=50)', 
               color='#ff7f0e', linestyle='-')
        
        # 指数平滑
        es_values = exponential_smoothing(total_loss, alpha=0.1)
        ax.plot(valid_iters, es_values, linewidth=2, label='Exponential Smoothing (α=0.1)', 
               color='#2ca02c', linestyle='--')
    
    ax.set_xlabel('Global Iteration (across epochs)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Total Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # 添加epoch分隔线
    add_epoch_separators(ax, data)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: total_loss.png")


def plot_main_losses(data, output_dir):
    """绘制主损失（layer5）曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    iterations = data['iterations']
    loss_types = ['box_l1_loss', 'box_giou_loss', 'label_focal_loss']
    
    for idx, loss_type in enumerate(loss_types):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        if loss_type in data['main_losses']:
            values = [v for v in data['main_losses'][loss_type] if v is not None]
            valid_iters = [iterations[i] for i, v in enumerate(data['main_losses'][loss_type]) if v is not None]
            
            if values:
                # 原始曲线
                ax.plot(valid_iters, values, linewidth=1.5, label=f'{loss_type} (raw)', 
                       alpha=0.5, marker='o', markersize=2)
                
                # 移动平均
                ma_values = moving_average(values, window_size=50)
                ax.plot(valid_iters, ma_values, linewidth=2, label='MA (w=50)', linestyle='-')
                
                # 指数平滑
                es_values = exponential_smoothing(values, alpha=0.1)
                ax.plot(valid_iters, es_values, linewidth=2, label='ES (α=0.1)', linestyle='--')
                
                ax.set_xlabel('Global Iteration', fontsize=10)
                ax.set_ylabel('Loss', fontsize=10)
                ax.set_title(f'Main Loss: {loss_type}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
                add_epoch_separators(ax, data)
    
    # SCEM损失
    if data['scem_losses']:
        ax = axes[3]
        for loss_type, values in data['scem_losses'].items():
            valid_values = [v for v in values if v is not None]
            valid_iters = [iterations[i] for i, v in enumerate(values) if v is not None]
            if valid_values:
                # 原始曲线
                ax.plot(valid_iters, valid_values, linewidth=1.5, label=f'{loss_type} (raw)', 
                       alpha=0.5, marker='o', markersize=2)
                
                # 移动平均
                ma_values = moving_average(valid_values, window_size=50)
                ax.plot(valid_iters, ma_values, linewidth=2, label=f'{loss_type} MA', linestyle='-')
                
                # 指数平滑
                es_values = exponential_smoothing(valid_values, alpha=0.1)
                ax.plot(valid_iters, es_values, linewidth=2, label=f'{loss_type} ES', linestyle='--')
        
        if ax.lines:  # 如果有数据
            ax.set_xlabel('Global Iteration', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title('SCEM Losses', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            add_epoch_separators(ax, data)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_losses.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: main_losses.png")


def plot_layer_comparison(data, output_dir):
    """绘制各layer损失对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    loss_types = ['l1', 'giou', 'focal']
    loss_labels = ['L1 Loss', 'GIoU Loss', 'Focal Loss']
    
    iterations = data['iterations']
    
    for idx, (loss_type, loss_label) in enumerate(zip(loss_types, loss_labels)):
        ax = axes[idx]
        
        for layer_name in sorted(data['layer_losses'].keys()):
            if loss_type in data['layer_losses'][layer_name]:
                values = [v for v in data['layer_losses'][layer_name][loss_type] if v is not None]
                valid_iters = [iterations[i] for i, v in enumerate(data['layer_losses'][layer_name][loss_type]) if v is not None]
                
                if values:
                    # 原始曲线
                    ax.plot(valid_iters, values, linewidth=1.5, label=f'{layer_name} (raw)', 
                           alpha=0.4, marker='o', markersize=1)
                    
                    # 移动平均
                    ma_values = moving_average(values, window_size=50)
                    ax.plot(valid_iters, ma_values, linewidth=2, label=f'{layer_name} MA', linestyle='-')
                    
                    # 指数平滑
                    es_values = exponential_smoothing(values, alpha=0.1)
                    ax.plot(valid_iters, es_values, linewidth=2, label=f'{layer_name} ES', linestyle='--')
        
        ax.set_xlabel('Global Iteration', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'{loss_label} by Layer', fontsize=12, fontweight='bold')
        # 只显示平滑曲线的图例
        handles, labels = ax.get_legend_handles_labels()
        filtered_handles = [h for h, l in zip(handles, labels) if '(raw)' not in l]
        filtered_labels = [l for l in labels if '(raw)' not in l]
        ax.legend(filtered_handles, filtered_labels, fontsize=8, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        add_epoch_separators(ax, data)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: layer_comparison.png")


def plot_class_losses_curves(data, output_dir):
    """绘制按类别损失的曲线图"""
    # 收集所有出现的类别
    all_classes = set()
    for layer_name in data['class_losses']:
        for loss_type in data['class_losses'][layer_name]:
            all_classes.update(data['class_losses'][layer_name][loss_type].keys())
    all_classes = sorted(list(all_classes))
    
    if not all_classes:
        print("No class loss data found, skipping curves.")
        return
    
    iterations = data['iterations']
    
    # 为每个layer和loss类型创建曲线图
    for loss_type in ['l1', 'giou']:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        layer_names = sorted(data['class_losses'].keys())
        
        for idx, layer_name in enumerate(layer_names[:6]):  # 最多6个layer
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            if loss_type in data['class_losses'][layer_name]:
                for class_idx in all_classes:
                    if class_idx in data['class_losses'][layer_name][loss_type]:
                        values = data['class_losses'][layer_name][loss_type][class_idx]
                        valid_values = [v for v in values if v is not None]
                        valid_iters = [iterations[i] for i, v in enumerate(values) if v is not None]
                        
                        if valid_values:
                            # 原始曲线
                            ax.plot(valid_iters, valid_values, linewidth=1, 
                                   label=f'Class {class_idx} (raw)', marker='o', markersize=1, alpha=0.4)
                            
                            # 移动平均
                            ma_values = moving_average(valid_values, window_size=50)
                            ax.plot(valid_iters, ma_values, linewidth=2, 
                                   label=f'Class {class_idx} MA', linestyle='-')
                            
                            # 指数平滑
                            es_values = exponential_smoothing(valid_values, alpha=0.1)
                            ax.plot(valid_iters, es_values, linewidth=2, 
                                   label=f'Class {class_idx} ES', linestyle='--')
                
                ax.set_xlabel('Global Iteration', fontsize=10)
                ax.set_ylabel('Loss', fontsize=10)
                ax.set_title(f'{layer_name} - {loss_type.upper()} Loss by Class', fontsize=11, fontweight='bold')
                # 只显示平滑曲线的图例，原始曲线作为背景
                handles, labels = ax.get_legend_handles_labels()
                # 过滤掉raw标签
                filtered_handles = [h for h, l in zip(handles, labels) if '(raw)' not in l]
                filtered_labels = [l for l in labels if '(raw)' not in l]
                ax.legend(filtered_handles, filtered_labels, fontsize=7, ncol=2, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
                add_epoch_separators(ax, data)
        
        # 隐藏多余的子图
        for idx in range(len(layer_names), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_losses_{loss_type}_curves.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: class_losses_{loss_type}_curves.png")


def plot_layer5_class_losses(data, output_dir):
    """绘制layer5按类别区分的l1和giou损失，每个类别一个子图，每个子图同时显示l1和giou"""
    # 只处理layer5的数据
    layer_name = 'layer5'
    
    if layer_name not in data['class_losses']:
        print("No layer5 class loss data found, skipping layer5 class losses plot.")
        return
    
    # 收集所有出现的类别
    all_classes = set()
    for loss_type in data['class_losses'][layer_name]:
        all_classes.update(data['class_losses'][layer_name][loss_type].keys())
    all_classes = sorted(list(all_classes))
    
    if not all_classes:
        print("No class loss data found for layer5, skipping.")
        return
    
    iterations = data['iterations']
    
    # 计算子图布局：每行3个，根据类别数量计算行数
    n_classes = len(all_classes)
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols  # 向上取整
    
    # 为L1和GIoU分别创建图
    for loss_type, loss_label, color in [('l1', 'L1 Loss', '#1f77b4'), ('giou', 'GIoU Loss', '#ff7f0e')]:
        if loss_type not in data['class_losses'][layer_name]:
            continue
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        # 统一处理axes，确保是数组格式
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, class_idx in enumerate(all_classes):
            ax = axes[idx]
            
            # 绘制损失
            if class_idx in data['class_losses'][layer_name][loss_type]:
                values = data['class_losses'][layer_name][loss_type][class_idx]
                valid_values = [v for v in values if v is not None]
                valid_iters = [iterations[i] for i, v in enumerate(values) if v is not None]
                
                if valid_values:
                    # 原始曲线
                    ax.plot(valid_iters, valid_values, linewidth=1.5, 
                           label=f'{loss_label} (raw)', color=color, alpha=0.4, marker='o', markersize=1)
                    
                    # 移动平均
                    ma_values = moving_average(valid_values, window_size=50)
                    ax.plot(valid_iters, ma_values, linewidth=2, 
                           label=f'{loss_label} MA', color=color, linestyle='-')
                    
                    # 指数平滑
                    es_values = exponential_smoothing(valid_values, alpha=0.1)
                    ax.plot(valid_iters, es_values, linewidth=2, 
                           label=f'{loss_label} ES', color=color, linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Global Iteration', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'Layer5 - Class {class_idx} - {loss_label}', fontsize=11, fontweight='bold')
            # 只显示平滑曲线的图例
            handles, labels = ax.get_legend_handles_labels()
            filtered_handles = [h for h, l in zip(handles, labels) if '(raw)' not in l]
            filtered_labels = [l for l in labels if '(raw)' not in l]
            ax.legend(filtered_handles, filtered_labels, fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            add_epoch_separators(ax, data)
        
        # 隐藏多余的子图
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'layer5_class_{loss_type}_losses.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: layer5_class_{loss_type}_losses.png")


def visualize_train_loss(log_file, output_dir):
    """
    可视化训练损失
    
    Args:
        log_file: 日志文件路径
        output_dir: 输出目录
    """
    print(f"Parsing log file: {log_file}")
    data, train_count = parse_log_file(log_file)
    
    print(f"Found {len(data['iterations'])} iterations in the last training session")
    print(f"Main losses: {list(data['main_losses'].keys())}")
    print(f"Layers: {list(data['layer_losses'].keys())}")
    print(f"SCEM losses: {list(data['scem_losses'].keys())}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制各种图表
    print("\nGenerating visualizations...")
    plot_total_loss(data, output_dir)
    plot_main_losses(data, output_dir)
    plot_layer_comparison(data, output_dir)
    plot_class_losses_curves(data, output_dir)
    plot_layer5_class_losses(data, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='可视化训练日志中的损失',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='日志文件路径（log.txt）'
    )
    
    parser.add_argument(
        '--exp_dir',
        type=str,
        default=None,
        help='实验目录（会自动查找log.txt）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认：log_file所在目录/fig_loss）'
    )
    
    args = parser.parse_args()
    
    # 确定日志文件路径
    if args.log_file:
        log_file = args.log_file
    elif args.exp_dir:
        log_file = os.path.join(args.exp_dir, 'log.txt')
    else:
        parser.print_help()
        print("\nError: 必须提供 --log_file 或 --exp_dir")
        return
    
    if not os.path.exists(log_file):
        print(f"Error: 日志文件不存在: {log_file}")
        return
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        log_dir = os.path.dirname(log_file)
        output_dir = os.path.join(log_dir, 'fig_loss')
    
    visualize_train_loss(log_file, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()

