#!/usr/bin/env python3
"""
对实验目录下每个 epoch_* 重新跑与训练时 submit_during_train 末尾相同的 TrackEval（或 ONLY_TRAIN_DETR 的 val_folder）。

命令行参数（与 argparse 一致）
------------------------------
  --exp-dir PATH           【必选】实验根目录；需能通过 resolve_two_stage_dir 找到 train/config.yaml
                           （单阶段即该目录；两阶段可指向顶层，会自动 fallback stage2_mot / stage1_detr）。

  --config PATH            【可选】直接指定 train/config.yaml；不设则从 resolve_two_stage_dir(exp-dir)/train/config.yaml 读取。

  --epoch-root PATH        【可选】实际存放各 epoch_* 子目录的根路径。
                           默认与保存 train/config.yaml 的目录一致（通常为 OUTPUTS_DIR/SUBMIT_DIR）。

  --dry-run                【可选】只打印将要执行的 TrackEval / val_folder，不真实运行；收尾可视化也不会执行。

  --skip-existing          【可选】若某 epoch 已存在 ``<SUBMIT_DATA_SPLIT>/eval/all_cls_summary.csv`` 则跳过该 epoch。
                           （ONLY_TRAIN_DETR 模式一般不生成该 csv，此选项多数情况下仍会执行各 epoch。）

  --force                  【可选】与 --skip-existing 同时使用时优先：即使已有 all_cls_summary.csv 仍强制重跑该 epoch。

  --no-vis                 【可选】跳过脚本末尾与训练结束相同的可视化（train/fig_loss、fig/ 验证曲线）。

默认行为简述
------------
  - 不加 --skip-existing：每个含 tracker 的 epoch 都会跑评测。
  - 收尾可视化默认开启（除非 --no-vis 或 --dry-run）；验证图读取 epoch_* / <SUBMIT_DATA_SPLIT> / eval / all_cls_summary.csv。

用法示例
--------
  cd /data/users/litianhao/hsmot_code/MeMOTR
  python scripts/rerun_trackeval_epoch_dirs.py --exp-dir /path/to/exp

  python scripts/rerun_trackeval_epoch_dirs.py --exp-dir /path/to/exp --skip-existing
  python scripts/rerun_trackeval_epoch_dirs.py --exp-dir /path/to/exp --dry-run --no-vis

依赖
----
  train/config.yaml；各 epoch_N/<SUBMIT_DATA_SPLIT>/tracker 下已有推理产生的 txt。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

import yaml


def _memotr_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_root() -> str:
    return os.path.dirname(_memotr_root())


def _ensure_imports() -> None:
    root = _memotr_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_train_config(cfg_path: str) -> dict:
    """读取 train/config.yaml：兼容纯 YAML 与 Logger 用 yaml.dump 保存的 TrackedConfig（含 !!python/object tag）。"""
    _ensure_imports()
    from utils.utils import TrackedConfig

    with open(cfg_path, encoding="utf-8") as f:
        raw = f.read()

    try:
        data = yaml.load(raw, Loader=yaml.FullLoader)
    except yaml.constructor.ConstructorError:
        data = yaml.unsafe_load(raw)

    def plain(x):
        if isinstance(x, TrackedConfig):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, list):
            return [plain(i) for i in x]
        if isinstance(x, tuple):
            return tuple(plain(i) for i in x)
        return x

    if data is None:
        raise ValueError(f"Empty or invalid YAML: {cfg_path}")
    return plain(data)


def _epoch_dirs(exp_dir: str) -> list[tuple[int, str]]:
    pattern = os.path.join(exp_dir, "epoch_*")
    out: list[tuple[int, str]] = []
    for d in glob.glob(pattern):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        m = re.match(r"epoch_(\d+)$", base)
        if not m:
            continue
        out.append((int(m.group(1)), d))
    out.sort(key=lambda x: x[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对 exp-dir 下所有 epoch_* 按训练逻辑重新执行 TrackEval / val_folder。"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="实验根目录（含 train/config.yaml，或与两阶段子目录结构一致）。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="显式指定 train config yaml；默认使用 resolve_two_stage_dir(exp-dir)/train/config.yaml。",
    )
    parser.add_argument(
        "--epoch-root",
        type=str,
        default=None,
        help="含 epoch_* 的目录；默认与 train/config.yaml 所在实验目录一致（单阶段即 exp-dir，两阶段为 stage2_mot 等）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的命令，不真正运行。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若已存在 <split>/eval/all_cls_summary.csv 则跳过（ONLY_TRAIN_DETR 时跳过已有 log 扩展结果仅依赖 log.txt，仍可用 --force）。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="与 --skip-existing 互斥时优先：即使已有 summary 也重跑。",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="跳过程序末尾与训练结束相同的可视化（训练损失曲线 + 验证指标图）。",
    )
    args = parser.parse_args()

    _ensure_imports()
    from submit_engine import resolve_two_stage_dir, resolve_submit_split_dir
    from hsmot.eval.validator import val_folder

    exp_dir = os.path.abspath(args.exp_dir)
    if args.config:
        cfg_path = os.path.abspath(args.config)
    else:
        resolved = resolve_two_stage_dir(exp_dir, prefer="stage2")
        cfg_path = os.path.join(resolved, "train", "config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"ERROR: 找不到训练配置: {cfg_path}", file=sys.stderr)
        return 1

    config = _load_train_config(cfg_path)
    data_root = config["DATA_ROOT"]
    dataset_name = config["DATASET"]
    dataset_split = config["SUBMIT_DATA_SPLIT"]
    dataset_version = config.get("DATASET_VERSION")
    dataset_type = config.get("DATASET_TYPE", "NPY")
    only_train_detr = bool(config.get("ONLY_TRAIN_DETR", False))

    # 与 submit_during_train 一致：由 data_split_dir 反推 dataset_root
    if "hsmot" not in str(dataset_name):
        print(f"ERROR: 本脚本仅支持 HSMOT 系 DATASET，当前为 {dataset_name}", file=sys.stderr)
        return 1

    data_split_dir = resolve_submit_split_dir(
        data_root=data_root,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_version=dataset_version,
        dataset_type=dataset_type,
    )
    if os.path.basename(data_split_dir) in ("npy", "npy2jpg"):
        dataset_root = os.path.dirname(os.path.dirname(data_split_dir))
    else:
        dataset_root = os.path.dirname(data_split_dir)

    gt_dir = os.path.join(dataset_root, dataset_split, "mot")
    if str(dataset_type).upper() == "3JPG":
        img_dir = os.path.join(dataset_root, dataset_split, "npy2jpg")
    else:
        img_dir = os.path.join(dataset_root, dataset_split, "npy")

    trackers_name = dataset_split
    trackers_subfolder = "tracker"

    trackeval_script = os.path.join(_repo_root(), "TrackEval", "scripts", "run_hsmot_8ch.py")
    if not os.path.isfile(trackeval_script):
        print(f"ERROR: 找不到 TrackEval 脚本: {trackeval_script}", file=sys.stderr)
        return 1

    # 实际含 epoch_* 的目录：默认与保存 train/config.yaml 的目录一致（SUBMIT_DIR）
    train_cfg_parent = os.path.dirname(os.path.dirname(cfg_path))
    work_root = (
        os.path.abspath(args.epoch_root) if args.epoch_root else train_cfg_parent
    )
    epoch_list = _epoch_dirs(work_root)
    if not epoch_list:
        print(f"ERROR: 在 {work_root} 下未找到任何 epoch_* 目录。", file=sys.stderr)
        return 1

    print(f"使用配置: {cfg_path}")
    print(f"工作目录(含 epoch_*): {work_root}")
    print(f"DATA_ROOT={data_root} DATASET={dataset_name} SUBMIT_DATA_SPLIT={dataset_split} DATASET_TYPE={dataset_type}")
    print(f"gt_dir={gt_dir}")
    print(f"img_dir={img_dir}")
    print(f"ONLY_TRAIN_DETR={only_train_detr}")
    print("---")

    py = sys.executable
    for epoch_idx, submit_dir_epoch in epoch_list:
        split_tracker = os.path.join(submit_dir_epoch, trackers_name, trackers_subfolder)
        if not os.path.isdir(split_tracker):
            print(f"[skip epoch_{epoch_idx}] 无 tracker 目录: {split_tracker}")
            continue

        eval_marker = os.path.join(submit_dir_epoch, trackers_name, "eval", "all_cls_summary.csv")
        if args.skip_existing and not args.force and os.path.isfile(eval_marker):
            print(f"[skip epoch_{epoch_idx}] 已存在 {eval_marker}")
            continue

        if only_train_detr:
            cmd_repr = (
                f"val_folder(gt_folder={gt_dir!r}, "
                f"pred_folder={os.path.join(submit_dir_epoch, trackers_name, trackers_subfolder)!r})"
            )
            print(f"\n== epoch_{epoch_idx} ONLY_TRAIN_DETR {cmd_repr}")
            if args.dry_run:
                continue
            lines = val_folder(
                gt_folder=gt_dir,
                pred_folder=os.path.join(submit_dir_epoch, trackers_name, trackers_subfolder),
            )
            log_append = os.path.join(submit_dir_epoch, trackers_name, "log.txt")
            os.makedirs(os.path.dirname(log_append), exist_ok=True)
            with open(log_append, "a", encoding="utf-8") as f:
                f.write("\n--- rerun_trackeval_epoch_dirs.py ONLY_TRAIN_DETR ---\n")
                f.write("\n".join(lines) + "\n")
            print("\n".join(lines))
        else:
            argv = [
                py,
                trackeval_script,
                "--USE_PARALLEL",
                "False",
                "--METRICS",
                "HOTA",
                "CLEAR",
                "Identity",
                "--GT_FOLDER",
                gt_dir,
                "--TRACKERS_FOLDER",
                submit_dir_epoch,
                "--TRACKERS_TO_EVAL",
                trackers_name,
                "--TRACKER_SUB_FOLDER",
                trackers_subfolder,
                "--IMG_FOLDER",
                img_dir,
            ]
            print(f"\n== epoch_{epoch_idx}\n  {' '.join(argv)}")
            if args.dry_run:
                continue
            r = subprocess.run(argv, cwd=os.path.dirname(trackeval_script))
            if r.returncode != 0:
                print(f"ERROR: epoch_{epoch_idx} TrackEval 退出码 {r.returncode}", file=sys.stderr)
                return r.returncode

    print("\n完成 TrackEval / val_folder。")

    # 与 train_engine._run_one_stage 收尾一致：训练损失可视化 + 验证指标可视化
    if not args.no_vis and not args.dry_run:
        from utils.vis_train_loss import visualize_train_loss
        from utils.vis_val import visualize_validation_metrics

        outputs_dir = config.get("OUTPUTS_DIR")
        outputs_dir = os.path.abspath(outputs_dir) if outputs_dir else train_cfg_parent

        try:
            print("开始生成训练损失可视化...")
            visualize_train_loss(
                log_file=os.path.join(outputs_dir, "train", "log.txt"),
                output_dir=os.path.join(outputs_dir, "train", "fig_loss"),
            )
            print("训练损失可视化完成")
        except Exception as e:
            print(f"训练损失可视化失败: {e}")

        if only_train_detr:
            print("ONLY_TRAIN_DETR=True，跳过 visualize_validation_metrics")
        else:
            try:
                print("开始生成验证指标可视化...")
                best_result = visualize_validation_metrics(
                    val_root_path=work_root,
                    fig_path=os.path.join(work_root, "fig"),
                    data_split=dataset_split,
                )
                if best_result:
                    print(
                        f"最佳组合分数: Epoch {best_result['epoch']}, "
                        f"Combined Score: {best_result['combined_score']:.4f}"
                    )
                print("验证指标可视化完成")
            except Exception as e:
                print(f"验证指标可视化失败: {e}")
    elif args.dry_run:
        print("(dry-run：未执行收尾可视化)")
    else:
        print("(已跳过收尾可视化：使用了 --no-vis)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
