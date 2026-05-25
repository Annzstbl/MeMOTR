#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将MOT跟踪结果可视化为视频

用法示例:
    # 处理单个txt文件
    python MeMOTR/utils/batch_vis_result.py \
        --txt_path /data4/litianhao/hsmot/memotr/spectralemb/18_half_priormap_99/epoch_15/test/tracker/xxxx.txt \
        --img_dir /data/users/litianhao/data/HSMOT/rgb/xxxx \
        --output_dir /data4/litianhao/hsmot/memotr/spectralemb/18_half_priormap_99/epoch_15/vis_result
    
    # 批量处理目录下所有txt文件
    python MeMOTR/utils/batch_vis_result.py \
        --txt_dir /data4/litianhao/hsmot/memotr/spectralemb/18_half_priormap_99/epoch_15/test/tracker \
        --img_root /data/users/litianhao/data/HSMOT/rgb \
        --output_dir /data4/litianhao/hsmot/memotr/spectralemb/18_half_priormap_99/epoch_15/vis_result
"""
import argparse
import os
import cv2
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import glob


# 生成100个颜色的列表（BGR格式，用于OpenCV）
def generate_color_palette(n=100):
    """生成n个不同的颜色，循环使用"""
    colors = []
    np.random.seed(42)  # 固定随机种子，保证颜色一致
    for i in range(n):
        # 使用HSV空间生成更均匀分布的颜色
        hue = int(180 * i / n)
        saturation = 200 + (i % 3) * 20
        value = 200 + (i % 3) * 20
        color_hsv = np.uint8([[[hue, saturation, value]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color_bgr)))
    return colors


COLOR_PALETTE = generate_color_palette(100)


def get_color_by_id(track_id):
    """根据track_id获取颜色，循环使用颜色表"""
    if track_id == None:
        return (255, 0, 0)
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]


def _parse_rect_line(parts: list[str]):
    """解析正框行，支持 tracking(10列) 与 detection(9列) 两种格式。"""
    frame = int(float(parts[0]))
    if len(parts) >= 10:
        track_id = int(float(parts[1]))
        x, y, w, h = map(float, parts[2:6])
        score = float(parts[6])
        cls = int(float(parts[7]))
    else:
        track_id = None
        x, y, w, h = map(float, parts[1:5])
        score = float(parts[5])
        cls = int(float(parts[6]))
    return frame, track_id, x, y, w, h, score, cls


def _parse_rotated_line(parts: list[str]):
    frame = int(float(parts[0]))
    track_id = int(float(parts[1]))
    x1, y1 = float(parts[2]), float(parts[3])
    x2, y2 = float(parts[4]), float(parts[5])
    x3, y3 = float(parts[6]), float(parts[7])
    x4, y4 = float(parts[8]), float(parts[9])
    score = float(parts[10])
    cls = int(float(parts[11]))
    return frame, track_id, x1, y1, x2, y2, x3, y3, x4, y4, score, cls


def parse_txt_file(txt_path):
    """
    解析跟踪结果txt文件，自动识别正框/旋转框格式。

    正框 tracking: frame, id, x, y, w, h, score, cls, -1, -1
    正框 detection: frame, x, y, w, h, score, cls, -1, -1
    旋转框: frame, id, x1..y4, score, cls, -1

    Returns:
        dict: {frame_id: [track_record, ...]}
        bbox_format: "rect" | "rotated"
    """
    track_dict = defaultdict(list)
    bbox_format = None

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 9:
                continue

            try:
                if len(parts) >= 13:
                    record = _parse_rotated_line(parts)[1:]
                    frame = int(float(parts[0]))
                    bbox_format = bbox_format or "rotated"
                else:
                    frame, track_id, x, y, w, h, score, cls = _parse_rect_line(parts)
                    record = (track_id, x, y, w, h, score, cls)
                    bbox_format = bbox_format or "rect"

                track_dict[frame].append(record)
            except (ValueError, IndexError) as e:
                print(f"Warning: 跳过无效行: {line}, 错误: {e}")
                continue

    return track_dict, bbox_format or "rect"


def draw_rect_bbox(img, track_id: int | None, x, y, w, h, score, cls=None, thickness=2, font_scale=0.6, color=None):
    """在图像上绘制正框 (xywh)。"""
    if color is None:
        color = get_color_by_id(track_id)

    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    label_text = f"id={track_id} {score:.2f}" if track_id is not None else f"{score:.2f}"
    if cls is not None:
        label_text = f"{label_text} cls={cls}"
    text_pos = (x1, y1 - 5)
    if text_pos[1] < 0:
        text_pos = (x1, y1 + 20)

    if font_scale > 0:
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            img,
            (text_pos[0], text_pos[1] - text_height - baseline),
            (text_pos[0] + text_width, text_pos[1] + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img, label_text, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )


def draw_rotated_bbox(img, track_id: int|None, x1, y1, x2, y2, x3, y3, x4, y4, score, thickness=2, font_scale=0.6, color=None):
    """
    在图像上绘制旋转框
    
    Args:
        img: 图像数组
        track_id: 跟踪ID
        x1, y1, x2, y2, x3, y3, x4, y4: 旋转框的4个顶点坐标
        score: 置信度分数
        thickness: 线条粗细
        font_scale: 字体大小
    """
    # 获取颜色
    if color is None:
        color = get_color_by_id(track_id)
    
    # 将坐标转换为整数
    pts = np.array([
        [int(x1), int(y1)],
        [int(x2), int(y2)],
        [int(x3), int(y3)],
        [int(x4), int(y4)]
    ], dtype=np.int32)
    
    # 绘制旋转框的4条边
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, thickness)
    cv2.line(img, tuple(pts[1]), tuple(pts[2]), color, thickness)
    cv2.line(img, tuple(pts[2]), tuple(pts[3]), color, thickness)
    cv2.line(img, tuple(pts[3]), tuple(pts[0]), color, thickness)
    
    # 绘制ID和分数文本
    label_text = f'{track_id}|{score:.2f}' if track_id is not None else f'{score:.2f}'
    text_pos = (int(x1), int(y1) - 5)
    if text_pos[1] < 0:
        text_pos = (int(x1), int(y1) + 20)
    
    if font_scale > 0:
        # 绘制文本背景（提高可读性）
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            img,
            (text_pos[0], text_pos[1] - text_height - baseline),
            (text_pos[0] + text_width, text_pos[1] + baseline),
            (0, 0, 0),
            -1
        )
        
        # 绘制文本
        cv2.putText(
            img, label_text, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA
        )


def visualize_tracking_result(txt_path, img_dir, output_path, fps=20):
    """
    将跟踪结果可视化为视频
    
    Args:
        txt_path: 跟踪结果txt文件路径
        img_dir: 图片目录路径
        output_path: 输出视频路径
        fps: 视频帧率
    """
    print(f"处理文件: {txt_path}")
    print(f"图片目录: {img_dir}")
    print(f"输出视频: {output_path}")
    
    # 解析跟踪结果
    track_dict, bbox_format = parse_txt_file(txt_path)
    txt_frame_ids = sorted(track_dict.keys()) if track_dict else []
    print(f"检测框格式: {bbox_format}")

    # 获取图片目录中的所有图片文件，只从图像序列确定帧范围
    img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
    img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    if not img_files:
        print(f"错误: 图片目录中没有找到图片文件: {img_dir}")
        return False

    # MOT txt 帧号从 1 开始，按排序后的图像列表建立 1-based 映射
    min_frame = 1
    max_frame = len(img_files)
    frame_to_img_file = {i + 1: img_files[i] for i in range(len(img_files))}
    print(f"图像序列帧范围: {min_frame} - {max_frame}，共 {len(img_files)} 帧")
    
    # 生成所有需要处理的帧ID列表（只从图像序列确定）
    all_frame_ids = list(range(min_frame, max_frame + 1))
    print(f"将处理 {len(all_frame_ids)} 帧（其中 {len(txt_frame_ids)} 帧有跟踪结果，{len(all_frame_ids) - len(txt_frame_ids)} 帧没有跟踪结果）")
    
    # 读取第一张图片以获取尺寸
    first_frame_id = min_frame
    if first_frame_id in frame_to_img_file:
        first_img_path = frame_to_img_file[first_frame_id]
    else:
        # 如果映射中没有，使用第一张图片文件
        first_img_path = img_files[0]
    
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"错误: 无法读取图片: {first_img_path}")
        return False
    
    height, width = first_img.shape[:2]
    print(f"图片尺寸: {width}x{height}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件: {output_path}")
        return False
    
    # 处理每一帧（包括没有跟踪结果的帧）
    processed_frames = 0
    for frame_id in tqdm(all_frame_ids, desc="生成视频帧"):
        # 查找对应的图片文件（使用之前建立的映射）
        img_path = None
        
        if frame_id in frame_to_img_file:
            # 如果帧号在映射中，直接使用
            img_path = frame_to_img_file[frame_id]
        else:
            # 如果不在映射中，尝试按帧号查找（标准命名格式）
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.bmp', '.BMP']:
                candidate = os.path.join(img_dir, f"{frame_id:06d}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
        
        if img_path is None:
            # 如果还是找不到，尝试按索引读取（作为后备方案）
            frame_idx = frame_id - min_frame
            if 0 <= frame_idx < len(img_files):
                img_path = img_files[frame_idx]
            else:
                print(f"警告: 找不到第 {frame_id} 帧的图片，跳过")
                continue
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片: {img_path}，跳过")
            continue
        
        # 如果图片尺寸不一致，调整大小
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # 绘制该帧的所有跟踪框（如果该帧有跟踪结果）
        if frame_id in track_dict:
            for track_data in track_dict[frame_id]:
                if bbox_format == "rect":
                    track_id, x, y, w, h, score, cls = track_data
                    draw_rect_bbox(img, track_id, x, y, w, h, score, cls)
                else:
                    track_id, x1, y1, x2, y2, x3, y3, x4, y4, score, cls = track_data
                    draw_rotated_bbox(img, track_id, x1, y1, x2, y2, x3, y3, x4, y4, score)
        
        # 在左上角添加帧序号
        frame_text = f"Frame: {frame_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_color = (255, 255, 255)  # 白色文字
        bg_color = (0, 0, 0)  # 黑色背景
        
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            frame_text, font, font_scale, thickness
        )
        
        # 绘制文本背景（提高可读性）
        padding = 5
        cv2.rectangle(
            img,
            (10, 10),
            (10 + text_width + padding * 2, 10 + text_height + baseline + padding * 2),
            bg_color,
            -1
        )
        
        # 绘制文本
        cv2.putText(
            img, frame_text, (10 + padding, 10 + text_height + padding),
            font, font_scale, text_color, thickness, cv2.LINE_AA
        )
        
        # 写入视频
        video_writer.write(img)
        processed_frames += 1
    
    # 释放视频写入器
    video_writer.release()
    
    print(f"成功生成视频: {output_path}")
    print(f"处理了 {processed_frames} 帧")
    
    return True


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def has_image_files(dir_path):
    """检查目录下是否包含图像文件"""
    if not os.path.isdir(dir_path):
        return False
    for name in os.listdir(dir_path):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            return True
    return False


def process_single_file(txt_path, img_root, output_dir):
    """
    处理单个txt文件
    
    Args:
        txt_path: txt文件路径
        img_root: 图片根目录
        output_dir: 输出目录
    """
    # 从txt文件名获取序列名（假设文件名是 xxxx.txt）
    seq_name = os.path.splitext(os.path.basename(txt_path))[0]
    
    # 构建图片目录路径；当前目录无图像时，尝试 vt_tiny 的 00 子目录
    img_dir = os.path.join(img_root, seq_name)
    if not has_image_files(img_dir):
        fallback_dir = os.path.join(img_dir, "00")
        if has_image_files(fallback_dir):
            img_dir = fallback_dir

    # 构建输出视频路径
    output_path = os.path.join(output_dir, f"{seq_name}.mp4")
    
    # 检查图片目录是否存在
    if not os.path.exists(img_dir):
        print(f"警告: 图片目录不存在: {img_dir}，跳过")
        return False
    
    # 可视化
    return visualize_tracking_result(txt_path, img_dir, output_path)


def process_directory(txt_dir, img_root, output_dir):
    """
    批量处理目录下所有txt文件
    
    Args:
        txt_dir: txt文件目录
        img_root: 图片根目录
        output_dir: 输出目录
    """
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    
    if not txt_files:
        print(f"警告: 在 {txt_dir} 中没有找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    success_count = 0
    for txt_path in tqdm(txt_files, desc="处理文件"):
        if process_single_file(txt_path, img_root, output_dir):
            success_count += 1
    
    print(f"\n处理完成: {success_count}/{len(txt_files)} 个文件成功")


def main():
    parser = argparse.ArgumentParser(
        description='将MOT跟踪结果可视化为视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--txt_path',
        type=str,
        help='单个跟踪结果txt文件路径',
    )
    
    parser.add_argument(
        '--txt_dir',
        type=str,
        help='包含多个txt文件的目录路径（批量处理）'
    )
    
    parser.add_argument(
        '--img_dir',
        type=str,
        help='单个序列的图片目录路径（与--txt_path配合使用）'
    )
    
    parser.add_argument(
        '--img_root',
        type=str,
        help='图片根目录路径（与--txt_dir配合使用，会自动查找对应序列名的子目录）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出视频保存目录'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='输出视频帧率（默认: 20）'
    )
    
    args = parser.parse_args()
    
    # 检查参数
    if args.txt_path and args.txt_dir:
        print("错误: --txt_path 和 --txt_dir 不能同时指定")
        return
    
    if not args.txt_path and not args.txt_dir:
        print("错误: 必须指定 --txt_path 或 --txt_dir")
        parser.print_help()
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理单个文件
    if args.txt_path:
        if not args.img_dir:
            print("错误: 使用 --txt_path 时必须指定 --img_dir")
            return
        
        if not os.path.exists(args.txt_path):
            print(f"错误: txt文件不存在: {args.txt_path}")
            return
        
        if not os.path.exists(args.img_dir):
            print(f"错误: 图片目录不存在: {args.img_dir}")
            return
        
        seq_name = os.path.splitext(os.path.basename(args.txt_path))[0]
        output_path = os.path.join(args.output_dir, f"{seq_name}.mp4")
        visualize_tracking_result(args.txt_path, args.img_dir, output_path, args.fps)
    
    # 批量处理目录
    elif args.txt_dir:
        if not args.img_root:
            print("错误: 使用 --txt_dir 时必须指定 --img_root")
            return
        
        if not os.path.exists(args.txt_dir):
            print(f"错误: txt目录不存在: {args.txt_dir}")
            return
        
        if not os.path.exists(args.img_root):
            print(f"错误: 图片根目录不存在: {args.img_root}")
            return
        
        process_directory(args.txt_dir, args.img_root, args.output_dir)


if __name__ == "__main__":
    main()

