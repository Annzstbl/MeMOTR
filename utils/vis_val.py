import argparse
import os
import pandas as pd
import glob 
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def visualize_validation_metrics(val_root_path, fig_path=None):
    """
    可视化验证指标并找到最佳组合分数
    
    Args:
        val_root_path: 包含epoch文件夹的根路径
        fig_path: 保存图片的路径，如果为None则使用val_root_path/fig
    
    Returns:
        dict: 包含最佳epoch信息的字典，如果未找到则返回None
    """
    if fig_path is None:
        fig_path = os.path.join(val_root_path, 'fig')
    
    # 1. 加载验证数据
    val_folder_list = glob.glob(os.path.join(val_root_path, "epoch*"))
    val_folder_list.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1]))
    
    epoch_list = [int(val_path.split("/")[-1].split("_")[-1]) for val_path in val_folder_list]
    val_matrix = []
    
    for val_path in val_folder_list:
        val_file = os.path.join(val_path, 'test', 'eval', 'all_cls_summary.csv')
        if os.path.exists(val_file):
            val_matrix.append(pd.read_csv(val_file))
        else:
            print(f"Warning: {val_file} not found, skipping...")
    
    if len(val_matrix) == 0:
        print("Error: No validation data found!")
        return None
    
    print(f"Found {len(epoch_list)} epochs: {epoch_list}")
    
    # 2. 提取指标
    HOTA_matrix = defaultdict(list)
    DetA_matrix = defaultdict(list)
    AssA_matrix = defaultdict(list)
    MOTA_matrix = defaultdict(list)
    IDF1_matrix = defaultdict(list)
    
    valid_cls = ['car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle', 
                 'awning-bike', 'cls_comb_cls_av', 'cls_comb_det_av']
    
    for val_df, epoch in zip(val_matrix, epoch_list):
        for row in val_df.iterrows():
            row_data = row[1]
            cls = row_data.cls
            if cls not in valid_cls:
                continue
            HOTA_matrix[cls].append(row_data.HOTA)
            DetA_matrix[cls].append(row_data.DetA)
            AssA_matrix[cls].append(row_data.AssA)
            MOTA_matrix[cls].append(row_data.MOTA)
            IDF1_matrix[cls].append(row_data.IDF1)
    
    # 3. 找到最佳组合分数并保存
    best_result = None
    if 'cls_comb_cls_av' in HOTA_matrix and 'cls_comb_det_av' in HOTA_matrix:
        cls_av_values = HOTA_matrix['cls_comb_cls_av']
        det_av_values = HOTA_matrix['cls_comb_det_av']
        
        combined_scores = [cls_av + det_av for cls_av, det_av in zip(cls_av_values, det_av_values)]
        max_idx = np.argmax(combined_scores)
        max_epoch = epoch_list[max_idx]
        max_cls_av = cls_av_values[max_idx]
        max_det_av = det_av_values[max_idx]
        max_combined = combined_scores[max_idx]
        
        filename = f'epoch{max_epoch:02d}-cls{max_cls_av:.2f}-det{max_det_av:.2f}.txt'
        filepath = os.path.join(val_root_path, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"Best Combined Score (cls_comb_cls_av + cls_comb_det_av)\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Epoch: {max_epoch}\n")
            f.write(f"cls_comb_cls_av: {max_cls_av:.4f}\n")
            f.write(f"cls_comb_det_av: {max_det_av:.4f}\n")
            f.write(f"Combined Score: {max_combined:.4f}\n")
        
        print(f"Best combined score found at epoch {max_epoch}:")
        print(f"  cls_comb_cls_av: {max_cls_av:.4f}")
        print(f"  cls_comb_det_av: {max_det_av:.4f}")
        print(f"  Combined: {max_combined:.4f}")
        print(f"  Saved to: {filepath}")
        
        best_result = {
            'epoch': max_epoch,
            'cls_comb_cls_av': max_cls_av,
            'cls_comb_det_av': max_det_av,
            'combined_score': max_combined,
            'filepath': filepath
        }
    else:
        print("Warning: cls_comb_cls_av or cls_comb_det_av not found in data")
    
    # 4. 绘制指标图
    os.makedirs(fig_path, exist_ok=True)
    
    for cls in HOTA_matrix.keys():
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_list, HOTA_matrix[cls], label='HOTA', marker='o')
        for i, h in enumerate(HOTA_matrix[cls]):
            plt.text(epoch_list[i], h, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
        plt.plot(epoch_list, DetA_matrix[cls], label='DetA', marker='s')
        plt.plot(epoch_list, AssA_matrix[cls], label='AssA', marker='^')
        plt.plot(epoch_list, MOTA_matrix[cls], label='MOTA', marker='d')
        plt.plot(epoch_list, IDF1_matrix[cls], label='IDF1', marker='v')
        plt.legend(loc='lower right')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title(f'{cls} Metrics')
        plt.xticks(epoch_list)
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_path, f'{cls}.png'), dpi=150)
        plt.close()
        print(f"Saved plot for {cls} to {os.path.join(fig_path, f'{cls}.png')}")
    
    return best_result


def main():
    parser = argparse.ArgumentParser(description='Visualize validation metrics across epochs')
    parser.add_argument('--val_root_path', type=str, required=True,
                        help='Root path containing epoch folders with validation results')
    parser.add_argument('--fig_path', type=str, default=None,
                        help='Path to save figures (default: val_root_path/fig)')
    
    args = parser.parse_args()
    
    visualize_validation_metrics(args.val_root_path, args.fig_path)
    print("Done!")


if __name__ == "__main__":
    main()
