#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将MOT检测结果可视化为视频

检测结果格式与跟踪结果的区别：没有 track_id 字段，其它一致。
可视化时根据得分 0.1、0.5 为检测框赋予不同颜色：
  - score >= 0.5: 绿色（高置信度）
  - 0.1 <= score < 0.5: 黄色（中置信度）
  - score < 0.1: 红色（低置信度）

用法示例:
    # 处理单个txt文件
    python MeMOTR/utils/batch_vis_detr.py \
        --txt_path /path/to/det/xxxx.txt \
        --img_dir /path/to/rgb/xxxx \
        --output_dir /path/to/vis_det

    python MeMOTR/utils/batch_vis_detr.py --txt_path /data4/litianhao/hsmot/memotr/spectralemb/22_7_only_detr_edgeswap_lrdrop_newtransforms_noscem_noiou/epoch_39/test/det/data30-3_det.txt --img_dir ./data/HSMOT/rgb/data30-3  --output_dir /data4/litianhao/hsmot/memotr/spectralemb/22_7_only_detr_edgeswap_lrdrop_newtransforms_noscem_noiou/epoch_39/test/vis_det

    # 批量处理目录下所有txt文件
    python MeMOTR/utils/batch_vis_detr.py \
        --txt_dir /path/to/det \
        --img_root /path/to/rgb \
        --output_dir /path/to/vis_det

    python MeMOTR/utils/batch_vis_detr.py --txt_dir /data4/litianhao/hsmot/memotr/spectralemb/22_7_only_detr_edgeswap_lrdrop_newtransforms_noscem_noiou/epoch_39/test/det --img_root ./data/HSMOT/rgb --output_dir /data4/litianhao/hsmot/memotr/spectralemb/22_7_only_detr_edgeswap_lrdrop_newtransforms_noscem_noiou/epoch_39/test/vis_det
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


# 按得分区间使用的颜色（BGR）
# score >= 0.5: 绿色（高置信度）
# 0.1 <= score < 0.5: 黄色（中置信度）
# score < 0.1: 红色（低置信度）
COLOR_HIGH = (0, 255, 0)    # BGR green
COLOR_MID = (0, 255, 255)   # BGR yellow
COLOR_LOW = (0, 0, 255)     # BGR red


def get_color_by_score(score):
    """根据得分返回颜色"""
    if score >= 0.5:
        return COLOR_HIGH
    if score >= 0.1:
        return COLOR_MID
    return COLOR_LOW

def get_thickness_by_score(score):
    """根据得分返回线条粗细"""
    if score >= 0.5:
        return 2
    if score >= 0.1:
        return 1
    return 1

def parse_txt_file(txt_path):
    """
    解析检测结果txt文件（无 track_id）

    格式: frame(from1), xyxyxyxy, score, cls, -1
    即: frame, x1, y1, x2, y2, x3, y3, x4, y4, score, cls, -1

    Returns:
        dict: {frame_id: [(x1, y1, x2, y2, x3, y3, x4, y4, score, cls), ...]}
    """
    det_dict = defaultdict(list)

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            # frame, x1, y1, x2, y2, x3, y3, x4, y4, score, cls, -1 → 12 列
            if len(parts) < 12:
                continue

            try:
                frame = int(float(parts[0]))
                x1, y1 = float(parts[1]), float(parts[2])
                x2, y2 = float(parts[3]), float(parts[4])
                x3, y3 = float(parts[5]), float(parts[6])
                x4, y4 = float(parts[7]), float(parts[8])
                score = float(parts[9])
                cls = int(float(parts[10]))

                det_dict[frame].append((x1, y1, x2, y2, x3, y3, x4, y4, score, cls))
            except (ValueError, IndexError) as e:
                print(f"Warning: 跳过无效行: {line}, 错误: {e}")
                continue

    return det_dict


def draw_rotated_bbox(img, x1, y1, x2, y2, x3, y3, x4, y4, score, thickness=None, font_scale=0, color=None):
    """
    在图像上绘制旋转框（检测结果，无 track_id）

    Args:
        img: 图像数组
        x1, y1, x2, y2, x3, y3, x4, y4: 旋转框的4个顶点坐标
        score: 置信度分数（用于决定颜色）
        thickness: 线条粗细
        font_scale: 字体大小
        color: 若指定则使用该颜色，否则按 score 计算
    """
    if color is None:
        color = get_color_by_score(score)
    if thickness is None:
        thickness = get_thickness_by_score(score)

    pts = np.array([
        [int(x1), int(y1)],
        [int(x2), int(y2)],
        [int(x3), int(y3)],
        [int(x4), int(y4)]
    ], dtype=np.int32)

    cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, thickness)
    cv2.line(img, tuple(pts[1]), tuple(pts[2]), color, thickness)
    cv2.line(img, tuple(pts[2]), tuple(pts[3]), color, thickness)
    cv2.line(img, tuple(pts[3]), tuple(pts[0]), color, thickness)

    label_text = f'{score:.2f}'
    text_pos = (int(x1), int(y1) - 5)
    if text_pos[1] < 0:
        text_pos = (int(x1), int(y1) + 20)

    if font_scale > 0:
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
        cv2.putText(
            img, label_text, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA
        )


def visualize_detection_result(txt_path, img_dir, output_path, fps=5):
    """
    将检测结果可视化为视频

    Args:
        txt_path: 检测结果txt文件路径
        img_dir: 图片目录路径
        output_path: 输出视频路径
        fps: 视频帧率
    """
    print(f"处理文件: {txt_path}")
    print(f"图片目录: {img_dir}")
    print(f"输出视频: {output_path}")

    det_dict = parse_txt_file(txt_path)
    txt_frame_ids = sorted(det_dict.keys()) if det_dict else []

    img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
    img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    if not img_files:
        print(f"错误: 图片目录中没有找到图片文件: {img_dir}")
        return False

    img_file_to_frame = {}
    frame_numbers = []

    for img_file in img_files:
        basename = os.path.basename(img_file)
        numbers = re.findall(r'\d+', basename)
        if numbers:
            try:
                frame_num = int(numbers[0])
                frame_numbers.append(frame_num)
                img_file_to_frame[img_file] = frame_num
            except Exception:
                pass

    if frame_numbers:
        min_frame = min(frame_numbers)
        max_frame = max(frame_numbers)
        frame_to_img_file = {v: k for k, v in img_file_to_frame.items()}
        print(f"从图片文件名提取帧号，帧范围: {min_frame} - {max_frame}，共 {len(frame_numbers)} 帧")
    else:
        min_frame = 1
        max_frame = len(img_files)
        frame_to_img_file = {i + 1: img_files[i] for i in range(len(img_files))}
        print(f"无法从文件名提取帧号，按顺序编号: {min_frame} - {max_frame}，共 {len(img_files)} 帧")

    all_frame_ids = list(range(min_frame, max_frame + 1))
    print(f"将处理 {len(all_frame_ids)} 帧（其中 {len(txt_frame_ids)} 帧有检测结果）")

    first_frame_id = min_frame
    if first_frame_id in frame_to_img_file:
        first_img_path = frame_to_img_file[first_frame_id]
    else:
        first_img_path = img_files[0]

    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"错误: 无法读取图片: {first_img_path}")
        return False

    height, width = first_img.shape[:2]
    print(f"图片尺寸: {width}x{height}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件: {output_path}")
        return False

    processed_frames = 0
    for frame_id in tqdm(all_frame_ids, desc="生成视频帧"):
        img_path = None

        if frame_id in frame_to_img_file:
            img_path = frame_to_img_file[frame_id]
        else:
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.bmp', '.BMP']:
                candidate = os.path.join(img_dir, f"{frame_id:06d}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if img_path is None:
            frame_idx = frame_id - min_frame
            if 0 <= frame_idx < len(img_files):
                img_path = img_files[frame_idx]
            else:
                print(f"警告: 找不到第 {frame_id} 帧的图片，跳过")
                continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片: {img_path}，跳过")
            continue

        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        if frame_id in det_dict:
            for det_data in det_dict[frame_id]:
                x1, y1, x2, y2, x3, y3, x4, y4, score, cls = det_data
                draw_rotated_bbox(img, x1, y1, x2, y2, x3, y3, x4, y4, score)

        frame_text = f"Frame: {frame_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)

        (text_width, text_height), baseline = cv2.getTextSize(
            frame_text, font, font_scale, thickness
        )
        padding = 5
        cv2.rectangle(
            img,
            (10, 10),
            (10 + text_width + padding * 2, 10 + text_height + baseline + padding * 2),
            bg_color,
            -1
        )
        cv2.putText(
            img, frame_text, (10 + padding, 10 + text_height + padding),
            font, font_scale, text_color, thickness, cv2.LINE_AA
        )

        video_writer.write(img)
        processed_frames += 1

    video_writer.release()

    print(f"成功生成视频: {output_path}")
    print(f"处理了 {processed_frames} 帧")

    return True


def process_single_file(txt_path, img_root, output_dir):
    """处理单个检测结果 txt 文件"""
    seq_name = os.path.splitext(os.path.basename(txt_path))[0]
    img_dir = os.path.join(img_root, seq_name.replace('_det', ''))
    output_path = os.path.join(output_dir, f"{seq_name}.mp4")

    if not os.path.exists(img_dir):
        print(f"警告: 图片目录不存在: {img_dir}，跳过")
        return False

    return visualize_detection_result(txt_path, img_dir, output_path)


def process_directory(txt_dir, img_root, output_dir):
    """批量处理目录下所有检测结果 txt 文件"""
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
        description='将MOT检测结果可视化为视频（按得分 0.1/0.5 着色）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--txt_path',
        type=str,
        help='单个检测结果txt文件路径',
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
        help='图片根目录路径（与--txt_dir配合使用）'
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

    if args.txt_path and args.txt_dir:
        print("错误: --txt_path 和 --txt_dir 不能同时指定")
        return

    if not args.txt_path and not args.txt_dir:
        print("错误: 必须指定 --txt_path 或 --txt_dir")
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)

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
        visualize_detection_result(args.txt_path, args.img_dir, output_path, args.fps)

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
