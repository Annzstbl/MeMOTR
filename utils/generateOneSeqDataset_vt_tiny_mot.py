"""从 VT-Tiny-MOT 抽取单个 scene（序列），构造最小 OneSeq 调试数据集。

输出目录结构与 VTTinyMOT 加载器一致::

    {output_root}/
      train2017/{scene}/00/*.jpg
      train2017/{scene}/01/*.jpg
      annotations/instances_train2017.json
      annotations/instances_00_train2017.json   # 若源文件存在
      annotations/instances_01_train2017.json   # 若源文件存在
      test2017 -> train2017                     # 软链，便于后续 eval/submit

训练时在 yaml 中设置::

    DATASET_DIR: VT-Tiny-MOT
    DATASET_VERSION: OneSeq_DJI_0022_1

实际数据路径: {DATA_ROOT}/VT-Tiny-MOT/OneSeq_DJI_0022_1/
"""

from __future__ import annotations

import json
import os
import shutil
import sys


ANN_FILE_TEMPLATES = {
    "plain": "instances_{split}2017.json",
    "00": "instances_00_{split}2017.json",
    "01": "instances_01_{split}2017.json",
}


def parse_scene_from_file_name(file_name: str) -> str:
    return file_name.split("/")[0]


def list_scenes(dataset_root: str, source_dataset: str = "VT-Tiny-MOT", split: str = "train") -> list[str]:
    ann_path = os.path.join(
        dataset_root,
        source_dataset,
        "annotations",
        ANN_FILE_TEMPLATES["plain"].format(split=split),
    )
    if not os.path.exists(ann_path):
        raise FileNotFoundError(ann_path)

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = sorted(
        {
            parse_scene_from_file_name(img["file_name"])
            for img in data["images"]
            if "/00/" in img["file_name"]
        }
    )
    return scenes


def filter_coco_by_scene(data: dict, scene_name: str) -> dict:
    images = [
        img
        for img in data["images"]
        if parse_scene_from_file_name(img["file_name"]) == scene_name
    ]
    image_ids = {img["id"] for img in images}
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]
    filtered = dict(data)
    filtered["images"] = images
    filtered["annotations"] = annotations
    return filtered


def _link_or_copy_dir(src: str, dst: str, use_symlink: bool) -> None:
    if use_symlink:
        os.symlink(src, dst)
    else:
        shutil.copytree(src, dst)


def build_one_seq_dataset(
    dataset_root: str,
    scene_name: str,
    source_dataset: str = "VT-Tiny-MOT",
    output_root: str | None = None,
    split: str = "train",
    use_symlink: bool = True,
    link_test_split: bool = True,
) -> str:
    """抽取单个 scene，生成 OneSeq 数据集。

    Args:
        dataset_root: DATA_ROOT，例如 /data1/users/litianhao01/hsmot/data
        scene_name: scene 名，例如 DJI_0022_1
        source_dataset: 源数据集子目录名，默认 VT-Tiny-MOT
        output_root: 输出子目录名，默认 OneSeq_{scene_name}
        split: 使用的标注 split，默认 train
        use_symlink: 图像目录是否用软链（默认 True，省空间）
        link_test_split: 是否将 test2017 软链到 train2017

    Returns:
        输出目录绝对路径
    """
    if output_root is None:
        output_root = f"OneSeq_{scene_name}"

    source_dir = os.path.join(dataset_root, source_dataset)
    output_dir = os.path.join(dataset_root, source_dataset, output_root)

    if os.path.exists(output_dir):
        print(f"错误: {output_dir} 已存在，请手动删除后再运行")
        sys.exit(1)

    src_scene_dir = os.path.join(source_dir, f"{split}2017", scene_name)
    src_rgb_dir = os.path.join(src_scene_dir, "00")
    src_ir_dir = os.path.join(src_scene_dir, "01")
    if not os.path.isdir(src_rgb_dir):
        raise FileNotFoundError(f"RGB 目录不存在: {src_rgb_dir}")
    if not os.path.isdir(src_ir_dir):
        raise FileNotFoundError(f"IR 目录不存在: {src_ir_dir}")

    dst_train_dir = os.path.join(output_dir, f"{split}2017", scene_name)
    dst_rgb_dir = os.path.join(dst_train_dir, "00")
    dst_ir_dir = os.path.join(dst_train_dir, "01")
    dst_ann_dir = os.path.join(output_dir, "annotations")

    os.makedirs(dst_ann_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dst_rgb_dir), exist_ok=True)

    src_rgb_dir = os.path.abspath(src_rgb_dir)
    src_ir_dir = os.path.abspath(src_ir_dir)
    _link_or_copy_dir(src_rgb_dir, dst_rgb_dir, use_symlink)
    _link_or_copy_dir(src_ir_dir, dst_ir_dir, use_symlink)

    src_ann_dir = os.path.join(source_dir, "annotations")
    written_anns: list[str] = []
    for ann_key, template in ANN_FILE_TEMPLATES.items():
        ann_name = template.format(split=split)
        src_ann_path = os.path.join(src_ann_dir, ann_name)
        if not os.path.exists(src_ann_path):
            continue

        with open(src_ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        filtered = filter_coco_by_scene(data, scene_name)
        if len(filtered["images"]) == 0:
            print(f"警告: {ann_name} 中未找到 scene={scene_name} 的图像，跳过")
            continue

        dst_ann_path = os.path.join(dst_ann_dir, ann_name)
        with open(dst_ann_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f)
        written_anns.append(
            f"{ann_name}: images={len(filtered['images'])}, "
            f"annotations={len(filtered['annotations'])}"
        )

    if not written_anns:
        raise RuntimeError(f"未生成任何标注文件，请检查 scene_name={scene_name}")

    if link_test_split:
        dst_test_dir = os.path.join(output_dir, "test2017")
        dst_train_root = os.path.abspath(os.path.join(output_dir, f"{split}2017"))
        os.symlink(dst_train_root, dst_test_dir)

        test_ann_src = os.path.join(src_ann_dir, ANN_FILE_TEMPLATES["plain"].format(split="test"))
        if os.path.exists(test_ann_src):
            with open(test_ann_src, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            test_filtered = filter_coco_by_scene(test_data, scene_name)
            if test_filtered["images"]:
                for ann_key, template in ANN_FILE_TEMPLATES.items():
                    test_ann_name = template.format(split="test")
                    test_src_path = os.path.join(src_ann_dir, test_ann_name)
                    if not os.path.exists(test_src_path):
                        continue
                    with open(test_src_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    filtered = filter_coco_by_scene(data, scene_name)
                    if not filtered["images"]:
                        continue
                    with open(os.path.join(dst_ann_dir, test_ann_name), "w", encoding="utf-8") as f:
                        json.dump(filtered, f)
                    written_anns.append(
                        f"{test_ann_name}: images={len(filtered['images'])}, "
                        f"annotations={len(filtered['annotations'])}"
                    )

    print("OneSeq 数据集生成完成")
    print("输出目录:", output_dir)
    print("scene:", scene_name)
    print("图像链接:" if use_symlink else "图像复制:", dst_rgb_dir, dst_ir_dir)
    for line in written_anns:
        print(" ", line)
    print("\n训练配置示例:")
    print(f"  DATA_ROOT: {dataset_root}")
    print(f"  DATASET_DIR: {source_dataset}")
    print(f"  DATASET_VERSION: {output_root}")
    return output_dir


if __name__ == "__main__":
    dataset_root = "/data1/users/litianhao01/hsmot/data"
    source_dataset = "VT-Tiny-MOT"
    scene_name = "DJI_0022_1"

    # 列出可用 scene: print(list_scenes(dataset_root, source_dataset))
    build_one_seq_dataset(
        dataset_root=dataset_root,
        scene_name=scene_name,
        source_dataset=source_dataset,
        output_root=f"OneSeq_{scene_name}",
        split="train",
        use_symlink=True,
        link_test_split=True,
    )
