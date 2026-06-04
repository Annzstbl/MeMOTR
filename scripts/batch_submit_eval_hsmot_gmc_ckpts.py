#!/usr/bin/env python3
"""对 stage2_mot checkpoint 用 GMC 模型重新 submit + TrackEval。

支持 ``SUBMIT_CHECKPOINT_DIR``（权重）与 ``SUBMIT_OUTPUT_DIR``（评测产物）分离。
默认产物：``<SUBMIT_OUTPUT_DIR>/epoch_<N>/test/``，不写入训练目录。

- HSMOT：``eval/all_cls_summary.csv``
- VT-Tiny：``eval_00``、``eval_01`` 各跑一次（与 ``batch_submit_eval_vt_tiny_ckpts.py`` 一致）
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys


def _memotr_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _epoch_from_ckpt(name: str, default_last_epoch: int) -> int:
    if name == "last.pth":
        return default_last_epoch
    m = re.match(r"checkpoint_(\d+)\.pth$", name)
    if not m:
        raise ValueError(f"Unrecognized checkpoint name: {name}")
    return int(m.group(1))


def _ckpt_done(epoch_dir: str, split: str, dataset_name: str) -> bool:
    if _is_vt_tiny_dataset(dataset_name):
        split_dir = os.path.join(epoch_dir, split)
        return (
            os.path.isfile(os.path.join(split_dir, "eval_00", "all_cls_summary.csv"))
            and os.path.isfile(os.path.join(split_dir, "eval_01", "all_cls_summary.csv"))
        )
    return os.path.isfile(
        os.path.join(epoch_dir, split, "eval", "all_cls_summary.csv")
    )


def _is_vt_tiny_dataset(dataset_name: str) -> bool:
    return dataset_name in ("vt_tiny_mot", "VT-Tiny-MOT")


def _has_tracker_outputs(split_dir: str) -> bool:
    tracker_dir = os.path.join(split_dir, "tracker")
    if not os.path.isdir(tracker_dir):
        return False
    return any(name.endswith(".txt") for name in os.listdir(tracker_dir))


def _move_submit_split(src_split: str, epoch_dir: str, split: str) -> str:
    os.makedirs(epoch_dir, exist_ok=True)
    dest_split = os.path.join(epoch_dir, split)
    if os.path.isdir(dest_split):
        shutil.rmtree(dest_split)
    shutil.move(src_split, dest_split)
    print(f"moved {src_split} -> {dest_split}")
    return dest_split


def _load_submit_dirs_from_config(config_path: str) -> tuple[str, str]:
    if _memotr_root() not in sys.path:
        sys.path.insert(0, _memotr_root())
    from utils.utils import load_yaml_with_inheritance
    from submit_engine import resolve_submit_checkpoint_and_output

    cfg = load_yaml_with_inheritance(config_path)
    return resolve_submit_checkpoint_and_output(cfg)


def _run_eval_for_epoch(
    *,
    memotr: str,
    checkpoint_dir: str,
    epoch_dir: str,
    split_dir: str,
    data_root: str,
    split: str,
    submit_cfg: dict,
) -> None:
    if memotr not in sys.path:
        sys.path.insert(0, memotr)
    from log.logger import Logger
    from submit_engine import (
        _run_hsmot_trackeval,
        _run_post_submit_eval,
        resolve_submit_dataset_root,
        resolve_submit_split_dir,
    )
    from utils.utils import load_train_config

    train_config = load_train_config(os.path.join(checkpoint_dir, "train/config.yaml"))
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

    submit_logger = Logger(logdir=split_dir, only_main=True)
    submit_logger.show(head="GMC re-eval TrackEval", log=f"epoch_dir={epoch_dir}")

    if _is_vt_tiny_dataset(dataset_name):
        merged = dict(train_config)
        merged.update(submit_cfg)
        merged["DATA_ROOT"] = data_root
        merged["SUBMIT_DATA_SPLIT"] = split
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
        submit_logger.show(
            head="GMC re-eval done (VT-Tiny)",
            log=f"{split_dir}/eval_00, {split_dir}/eval_01",
        )
    else:
        _run_hsmot_trackeval(
            dataset_root=dataset_root,
            dataset_split=split,
            dataset_type=dataset_type,
            tracker_dir=epoch_dir,
            trackers_name=split,
            trackers_subfolder="tracker",
        )
        submit_logger.show(
            head="GMC re-eval done",
            log=os.path.join(split_dir, "eval", "all_cls_summary.csv"),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch GMC submit + hsmot TrackEval (separate output dir)."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="submit 配置（如 configs_hsmot_spectral_embed_252/20260511-1-gmc-submit.yaml）",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="覆盖 yaml 中 SUBMIT_CHECKPOINT_DIR（含 *.pth 与 train/config.yaml）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="覆盖 yaml 中 SUBMIT_OUTPUT_DIR（评测产物根目录）",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="数据集根目录；默认用 submit 配置中的 DATA_ROOT",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--submit-threads", type=int, default=4)
    parser.add_argument("--available-gpus", default="0,1")
    parser.add_argument("--last-epoch", type=int, default=18, help="last.pth 对应 epoch")
    parser.add_argument(
        "--ckpts",
        nargs="*",
        default=None,
        help="仅处理指定 checkpoint 文件名，如 checkpoint_17.pth last.pth",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不跑 submit，仅对已有 epoch_*/test/tracker 做 TrackEval",
    )
    parser.add_argument(
        "--miss-tolerance",
        type=int,
        default=None,
        help="覆盖 yaml 中 MISS_TOLERANCE（连续丢失多少帧后 id=-1 删除 track）",
    )
    args = parser.parse_args()

    memotr = _memotr_root()
    if memotr not in sys.path:
        sys.path.insert(0, memotr)

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(memotr, config_path)

    checkpoint_dir, output_dir = _load_submit_dirs_from_config(config_path)
    if args.checkpoint_dir is not None:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    from utils.utils import load_train_config, load_yaml_with_inheritance

    submit_cfg = load_yaml_with_inheritance(config_path)
    train_config_for_done = load_train_config(os.path.join(checkpoint_dir, "train/config.yaml"))
    dataset_name_for_done = train_config_for_done["DATASET"]
    if args.data_root is None:
        data_root = submit_cfg.get("DATA_ROOT")
        if data_root is None:
            from utils.utils import load_train_config

            data_root = load_train_config(os.path.join(checkpoint_dir, "train/config.yaml"))["DATA_ROOT"]
    else:
        data_root = args.data_root

    separate_output = os.path.normpath(checkpoint_dir) != os.path.normpath(output_dir)
    epoch_tag = "" if separate_output else "_gmc"

    if args.ckpts:
        ckpts = [os.path.join(checkpoint_dir, name) for name in args.ckpts]
        ckpts = [p for p in ckpts if os.path.isfile(p)]
    else:
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    ckpts.sort(key=lambda p: _epoch_from_ckpt(os.path.basename(p), args.last_epoch))
    if not ckpts and not args.eval_only:
        print(f"ERROR: no .pth under {checkpoint_dir}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    py = sys.executable
    main_py = os.path.join(memotr, "main.py")

    epoch_items = (
        [(os.path.basename(p), _epoch_from_ckpt(os.path.basename(p), args.last_epoch)) for p in ckpts]
        if ckpts
        else [
            (f"epoch_{int(m.group(1))}{epoch_tag}", int(m.group(1)))
            for p in glob.glob(os.path.join(output_dir, f"epoch_*{epoch_tag}"))
            if (m := re.search(rf"epoch_(\d+){epoch_tag}$", os.path.basename(p.rstrip("/"))))
        ]
    )

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"output_dir:     {output_dir}")

    for ckpt_name, epoch in epoch_items:
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch}{epoch_tag}")
        split_dir = os.path.join(epoch_dir, args.split)

        if args.skip_existing and not args.force and _ckpt_done(
            epoch_dir, args.split, dataset_name_for_done
        ):
            print(f"[skip] epoch_{epoch}{epoch_tag} already evaluated")
            continue

        print(f"\n{'=' * 60}\n== epoch_{epoch}{epoch_tag}  checkpoint={ckpt_name}\n{'=' * 60}")

        need_submit = (
            not args.eval_only
            and (args.force or not _has_tracker_outputs(split_dir))
        )

        if need_submit:
            submit_cmd = [
                py,
                main_py,
                "--config-path",
                config_path,
                "--mode",
                "submit",
                "--submit-model",
                ckpt_name,
                "--submit-data-split",
                args.split,
                "--submit-threads",
                str(args.submit_threads),
                "--data-root",
                data_root,
                "--available-gpus",
                str(args.available_gpus),
            ]
            if args.miss_tolerance is not None:
                submit_cmd.extend(["--miss-tolerance", str(args.miss_tolerance)])
            # main.py submit 读 yaml 的 SUBMIT_OUTPUT_DIR；须显式覆盖，否则仍写到配置里的旧目录
            submit_cmd.extend(["--submit-output-dir", output_dir, "--submit-dir", output_dir])
            print("submit:", " ".join(submit_cmd))
            if args.dry_run:
                continue

            r = subprocess.run(submit_cmd, cwd=memotr, env=env)
            if r.returncode != 0:
                print(f"ERROR: submit failed for {ckpt_name}, exit={r.returncode}", file=sys.stderr)
                return r.returncode

            src_split = os.path.join(output_dir, args.split)
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
            checkpoint_dir=checkpoint_dir,
            epoch_dir=epoch_dir,
            split_dir=split_dir,
            data_root=data_root,
            split=args.split,
            submit_cfg=submit_cfg,
        )
        if _is_vt_tiny_dataset(dataset_name_for_done):
            print(f"[done] epoch_{epoch}{epoch_tag} -> {split_dir}/eval_00, eval_01")
        else:
            print(f"[done] epoch_{epoch}{epoch_tag} -> {split_dir}/eval/all_cls_summary.csv")

    if not args.dry_run:
        try:
            from utils.vis_val import visualize_validation_metrics

            fig_path = os.path.join(output_dir, "fig")
            best = visualize_validation_metrics(
                val_root_path=output_dir,
                fig_path=fig_path,
                data_split=args.split,
                epoch_suffix=epoch_tag or None,
            )
            if best:
                print(
                    f"Best GMC: epoch={best['epoch']}, combined={best['combined_score']:.4f}"
                )
        except Exception as exc:
            print(f"vis_val skipped: {exc}")

    print("\nAll GMC checkpoints finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
