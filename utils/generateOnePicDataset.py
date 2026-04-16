import os
import shutil
import sys


def build_onepic_dataset(dataset_root, vid_name, frame_id, repeat_num, output_root="OnePic"):

    # -------------------------
    # 0 检查输出目录
    # -------------------------
    output_root = os.path.join(dataset_root, output_root)
    if os.path.exists(output_root):
        print(f"错误: {output_root} 已存在，请手动删除后再运行")
        sys.exit(1)

    # 创建目录
    mot_dst = os.path.join(output_root, "mot")
    npy_dst = os.path.join(output_root, "npy", vid_name)
    train_dst = os.path.join(output_root, "train")
    test_dst = os.path.join(output_root, "test")

    os.makedirs(mot_dst)
    os.makedirs(npy_dst)
    os.makedirs(train_dst)
    os.makedirs(test_dst)

    # -------------------------
    # 源路径
    # -------------------------
    mot_src = os.path.join(dataset_root, "mot")
    npy_src = os.path.join(dataset_root, "npy")

    txt_src = os.path.join(mot_src, f"{vid_name}.txt")
    npy_frame_src = os.path.join(npy_src, vid_name, f"{frame_id:06d}.npy")

    if not os.path.exists(txt_src):
        raise FileNotFoundError(txt_src)

    if not os.path.exists(npy_frame_src):
        raise FileNotFoundError(npy_frame_src)

    # -------------------------
    # 1 读取txt筛选帧
    # -------------------------
    new_labels = []

    with open(txt_src, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])

            if frame == frame_id:
                new_labels.append(parts)

    if len(new_labels) == 0:
        print("该帧没有标注")
        return

    # -------------------------
    # 2 写新txt
    # -------------------------
    txt_dst = os.path.join(mot_dst, f"{vid_name}.txt")

    with open(txt_dst, "w") as f:
        for i in range(repeat_num):
            frame_new = i + 1

            for parts in new_labels:
                parts_new = parts.copy()
                parts_new[0] = str(frame_new)
                f.write(",".join(parts_new) + "\n")

    # -------------------------
    # 3 复制npy
    # -------------------------
    for i in range(repeat_num):
        dst_name = f"{i+1:06d}.npy"
        dst_path = os.path.join(npy_dst, dst_name)
        shutil.copy(npy_frame_src, dst_path)

    # -------------------------
    # 4 创建软链接
    # -------------------------
    os.symlink("../mot", os.path.join(train_dst, "mot"))
    os.symlink("../npy", os.path.join(train_dst, "npy"))

    os.symlink("../mot", os.path.join(test_dst, "mot"))
    os.symlink("../npy", os.path.join(test_dst, "npy"))

    print("OnePic 数据集生成完成")
    print("输出目录:", output_root)


if __name__ == "__main__":

    dataset_root = "./data/hsmot"  # 原数据集目录
    vid_name = "data52-4"           # 视频名
    frame_id = 1              # 选取帧
    repeat_num = 20             # 复制次数

    build_onepic_dataset(
        dataset_root,
        vid_name,
        frame_id,
        repeat_num,
        output_root="OnePic_52_4"
    )