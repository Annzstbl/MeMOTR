#!/usr/bin/env python3
"""对 stage2_mot 目录下每个 .pth 做 submit 推理 + VT-Tiny TrackEval（与 submit_during_train 一致）。"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

# submit 产物应位于 epoch_N/{split}/ 下；旧版脚本误将 test 重命名为 epoch_N，产物落在 epoch_N/tracker
_SPLIT_ARTIFACTS = (
    "tracker",
    "det",
    "config.yaml",
    "log.txt",
    "eval",
    "eval_00",
    "eval_01",
    "eval_01_agnostic",
    "vis",
    "vis_results",
)


def _memotr_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _epoch_from_ckpt(name: str, default_last_epoch: int) -> int:
    if name == "last.pth":
        return default_last_epoch
    m = re.match(r"checkpoint_(\d+)\.pth$", name)
    if not m:
        raise ValueError(f"Unrecognized checkpoint name: {name}")
    return int(m.group(1))


def _ckpt_done(epoch_dir: str, split: str) -> bool:
    return (
        os.path.isfile(os.path.join(epoch_dir, split, "eval_00", "all_cls_summary.csv"))
        and os.path.isfile(os.path.join(epoch_dir, split, "eval_01", "all_cls_summary.csv"))
    )


def _normalize_epoch_split_dir(epoch_dir: str, split: str) -> str:
    """规范为 epoch_N/{split}/，并将旧版 epoch_N/tracker 等迁移到 epoch_N/test/。"""
    split_dir = os.path.join(epoch_dir, split)
    legacy_tracker = os.path.join(epoch_dir, "tracker")
    if os.path.isdir(legacy_tracker) and not os.path.isdir(os.path.join(split_dir, "tracker")):
        os.makedirs(split_dir, exist_ok=True)
        for name in _SPLIT_ARTIFACTS:
            src = os.path.join(epoch_dir, name)
            if not os.path.exists(src):
                continue
            dst = os.path.join(split_dir, name)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        print(f"[migrate] legacy layout -> {split_dir}")
    return split_dir


def _has_tracker_outputs(split_dir: str) -> bool:
    tracker_dir = os.path.join(split_dir, "tracker")
    if not os.path.isdir(tracker_dir):
        return False
    return any(name.endswith(".txt") for name in os.listdir(tracker_dir))


def _move_submit_split(src_split: str, epoch_dir: str, split: str) -> str:
    """将 stage2_mot/test 挪到 stage2_mot/epoch_N/test/（不整目录删除 epoch_N）。"""
    os.makedirs(epoch_dir, exist_ok=True)
    dest_split = os.path.join(epoch_dir, split)
    if os.path.isdir(dest_split):
        shutil.rmtree(dest_split)
    shutil.move(src_split, dest_split)
    print(f"moved {src_split} -> {dest_split}")
    return dest_split


def _run_eval_for_epoch(
    *,
    memotr: str,
    submit_dir: str,
    epoch_dir: str,
    split_dir: str,
    data_root: str,
    split: str,
) -> None:
    if memotr not in sys.path:
        sys.path.insert(0, memotr)
    from log.logger import Logger
    from submit_engine import (
        _run_post_submit_eval,
        resolve_submit_dataset_root,
        resolve_submit_split_dir,
    )
    from utils.utils import load_train_config

    train_config = load_train_config(os.path.join(submit_dir, "train/config.yaml"))
    dataset_name = train_config["DATASET"]
    dataset_version = train_config.get("DATASET_VERSION")
    dataset_type = train_config.get("DATASET_TYPE", None)
    dataset_dir = train_config.get("DATASET_DIR", "VT-Tiny-MOT")

    data_split_dir = resolve_submit_split_dir(
        data_root=data_root,
        dataset_name=dataset_name,
        dataset_split=split,
        dataset_version=dataset_version,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
    )
    dataset_root = resolve_submit_dataset_root(
        data_root=data_root,
        dataset_name=dataset_name,
        dataset_split=split,
        dataset_version=dataset_version,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
    )

    merged = dict(train_config)
    merged.update(
        {
            "DATA_ROOT": data_root,
            "SUBMIT_DATA_SPLIT": split,
        }
    )
    submit_logger = Logger(logdir=split_dir, only_main=True)
    _run_post_submit_eval(
        config=merged,
        dataset_name=dataset_name,
        dataset_split=split,
        dataset_type=dataset_type,
        dataset_root=dataset_root,
        data_split_dir=data_split_dir,
        tracker_dir=epoch_dir,
        trackers_name=split,
        trackers_subfolder="tracker",
        only_train_detr=bool(train_config.get("ONLY_TRAIN_DETR", False)),
        submit_logger=submit_logger,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch submit + VT-Tiny eval for all stage2 checkpoints.")
    parser.add_argument("--submit-dir", required=True, help="stage2_mot 目录（含 *.pth 与 train/config.yaml）")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--submit-threads", type=int, default=4)
    parser.add_argument(
        "--available-gpus",
        default="2",
        help="物理 GPU id，传给 main.py --available-gpus（设置 CUDA_VISIBLE_DEVICES）",
    )
    parser.add_argument("--last-epoch", type=int, default=18, help="last.pth 对应 epoch 号")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不跑 submit，仅迁移旧目录结构并对已有 tracker 做 TrackEval",
    )
    args = parser.parse_args()

    memotr = _memotr_root()
    if memotr not in sys.path:
        sys.path.insert(0, memotr)

    submit_dir = os.path.abspath(args.submit_dir)
    ckpts = glob.glob(os.path.join(submit_dir, "*.pth"))
    ckpts.sort(key=lambda p: _epoch_from_ckpt(os.path.basename(p), args.last_epoch))
    if not ckpts and not args.eval_only:
        print(f"ERROR: no .pth under {submit_dir}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    py = sys.executable
    main_py = os.path.join(memotr, "main.py")

    epoch_items = (
        [(os.path.basename(p), _epoch_from_ckpt(os.path.basename(p), args.last_epoch)) for p in ckpts]
        if ckpts
        else [
            (f"epoch_{int(m.group(1))}", int(m.group(1)))
            for p in glob.glob(os.path.join(submit_dir, "epoch_*"))
            if (m := re.search(r"epoch_(\d+)$", p))
        ]
    )

    for ckpt_name, epoch in epoch_items:
        epoch_dir = os.path.join(submit_dir, f"epoch_{epoch}")
        if not os.path.isdir(epoch_dir):
            if args.eval_only:
                print(f"[skip] epoch_{epoch}: no directory {epoch_dir}")
                continue
        split_dir = _normalize_epoch_split_dir(epoch_dir, args.split)

        if args.skip_existing and not args.force and _ckpt_done(epoch_dir, args.split):
            print(f"[skip] epoch_{epoch} already evaluated")
            continue

        print(f"\n{'=' * 60}\n== epoch_{epoch}  checkpoint={ckpt_name}\n{'=' * 60}")

        need_submit = (
            not args.eval_only
            and (args.force or not _has_tracker_outputs(split_dir))
        )

        if need_submit:
            submit_cmd = [
                py,
                main_py,
                "--config-path",
                args.config_path,
                "--mode",
                "submit",
                "--submit-dir",
                submit_dir,
                "--submit-model",
                ckpt_name,
                "--submit-data-split",
                args.split,
                "--submit-threads",
                str(args.submit_threads),
                "--data-root",
                args.data_root,
                "--available-gpus",
                str(args.available_gpus),
            ]
            print("submit:", " ".join(submit_cmd))
            if args.dry_run:
                continue

            r = subprocess.run(submit_cmd, cwd=memotr, env=env)
            if r.returncode != 0:
                print(f"ERROR: submit failed for {ckpt_name}, exit={r.returncode}", file=sys.stderr)
                return r.returncode

            src_split = os.path.join(submit_dir, args.split)
            if not os.path.isdir(src_split):
                print(f"ERROR: missing submit output {src_split}", file=sys.stderr)
                return 1

            split_dir = _move_submit_split(src_split, epoch_dir, args.split)
        elif not _has_tracker_outputs(split_dir):
            print(f"ERROR: no tracker under {split_dir}, run submit first", file=sys.stderr)
            return 1
        else:
            print(f"[skip submit] tracker exists: {split_dir}/tracker")

        if args.dry_run:
            continue

        _run_eval_for_epoch(
            memotr=memotr,
            submit_dir=submit_dir,
            epoch_dir=epoch_dir,
            split_dir=split_dir,
            data_root=args.data_root,
            split=args.split,
        )
        print(f"[done] epoch_{epoch} eval -> {epoch_dir}/{args.split}/eval_00, eval_01")

    if not args.dry_run:
        try:
            from utils.vis_val import visualize_validation_metrics

            fig_path = os.path.join(submit_dir, "fig")
            best = visualize_validation_metrics(
                val_root_path=submit_dir,
                fig_path=fig_path,
                data_split=args.split,
            )
            if best:
                print(
                    f"Best: epoch={best['epoch']}, combined={best['combined_score']:.4f}"
                )
        except Exception as e:
            print(f"vis_val skipped: {e}")

    print("\nAll checkpoints finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python scripts/batch_submit_eval_vt_tiny_ckpts.py \
#   --submit-dir /data4/litianhao/hsmot/memotr/vt_tiny/vt_tiny_20260511-1-178/stage2_mot \
#   --config-path configs_vt_tiny_178/20260511-1.yaml \
#   --data-root /data/users/wangying01/lth/hsmot/data \
#   --available-gpus 0,1 \
#   --submit-threads 4 \
#   --force

# python scripts/batch_submit_eval_vt_tiny_ckpts.py --submit-dir /data4/litianhao/hsmot/memotr/vt_tiny/vt_tiny_20260511-3-178/stage2_mot --config-path /data/users/litianhao01/hsmot/MeMOTR/configs_vt_tiny_178/20260511-3.yaml --data-root /data/users/litianhao01/hsmot/data --available-gpus 0,1 --submit-threads 4
# python scripts/batch_submit_eval_vt_tiny_ckpts.py --submit-dir /data4/litianhao/hsmot/memotr/vt_tiny/20260511-4-99/stage2_mot --config-path /data/users/litianhao01/hsmot/MeMOTR/configs_vt_tiny_99/20260511-4.yaml --data-root /data/users/litianhao01/hsmot/data --available-gpus 2,3 --submit-threads 4