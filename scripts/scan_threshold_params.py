#!/usr/bin/env python3
"""MeMOTR 阈值参数分阶段扫描（submit + TrackEval）。

详见本文件 USAGE_GUIDE 或运行: python scripts/scan_threshold_params.py --help
"""

from __future__ import annotations

USAGE_GUIDE = r"""
================================================================================
MeMOTR 阈值参数分阶段扫描 — 使用说明
================================================================================

环境
----
  conda activate hsmot
  cd /data1/users/litianhao01/hsmot/MeMOTR

  或使用便捷脚本（自动激活 hsmot）:
  bash scripts/run_thresh_scan.sh --stage 1 ...

扫描流程（每阶段约 5 组实验，阶段间人工选最优参数）
--------------------------------------------------
  Stage 1: 扫描 DET_SCORE_THRESH，RESULT_SCORE_THRESH = DET
  Stage 2: 固定 Stage1 最优 DET，扫描 TRACK_SCORE_THRESH (< DET)，RESULT = TRACK
  Stage 3: 固定 DET/TRACK，单独扫描 RESULT_SCORE_THRESH
  Stage 4: 固定 DET/TRACK/RESULT，扫描 UPDATE_THRESH

参数如何传入模型
----------------
  DET / TRACK / RESULT_SCORE_THRESH
    -> 写入 submit config，传给 RuntimeTracker 与最终输出过滤
  UPDATE_THRESH
    -> 写入 train_config 后再 build_model()，传给 QueryUpdater.update_threshold

公共参数（可按需修改）
--------------------
  SCAN_ROOT=/data1/users/litianhao01/experiment/memotr/thresh_scan
  COMMON="--config-path configs_vt_tiny_99/20260511-1.yaml \
    --submit-dir /data1/users/litianhao01/experiment/memotr/vt_tiny_20260511-1-178 \
    --submit-model checkpoint_17.pth \
    --scan-root ${SCAN_ROOT} \
    --available-gpus 0 \
    --submit-threads 4 \
    --iou-threshold 0.2"

各阶段命令
----------
  # Stage 1: 扫描 DET（默认 0.2~0.8），RESULT = DET
  bash scripts/run_thresh_scan.sh --stage 1 ${COMMON}

  # Stage 2: 假设 Stage1 最优 DET=0.5
  bash scripts/run_thresh_scan.sh --stage 2 --det 0.5 ${COMMON}

  # Stage 3: 假设 Stage2 最优 TRACK=0.4
  bash scripts/run_thresh_scan.sh --stage 3 --det 0.5 --track 0.4 ${COMMON}

  # Stage 4: 假设 Stage3 最优 RESULT=0.45
  bash scripts/run_thresh_scan.sh --stage 4 --det 0.5 --track 0.4 --result 0.45 ${COMMON}

输出目录结构
------------
  {scan_root}/stage{N}/
  ├── manifest.json          # 全部实验记录
  ├── summary.md             # 对比表格（含 HOTA）
  ├── exp01_det0.30/
  │   ├── params.json
  │   ├── submit_config.yaml
  │   ├── train_config_override.yaml
  │   └── test/
  │       ├── tracker/*.txt  # 跟踪结果
  │       ├── eval_00/       # TrackEval（instances_00_test2017.json）
  │       └── eval_01/       # TrackEval（instances_01_test2017.json）

其他选项
--------
  --dry-run              仅预览实验计划，不跑推理
  --values 0.35 0.45     自定义当前阶段的扫描点（覆盖默认网格）
  --skip-submit          跳过 submit，仅重跑 TrackEval
  --skip-eval            仅 submit，不评测
  --num-experiments 5    自动网格点数（默认 5）

每阶段完成后查看 stage{N}/summary.md 与各实验 test/eval/，
选定最优参数后再跑下一阶段。
================================================================================
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# MeMOTR root on sys.path
MEMOTR_ROOT = Path(__file__).resolve().parents[1]
if str(MEMOTR_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMOTR_ROOT))

from submit_engine import (  # noqa: E402
    _run_submit_pipeline,
    is_vt_tiny_dataset,
    list_submit_sequences,
    resolve_submit_dataloader_workers,
    resolve_submit_split_dir,
    resolve_two_stage_dir,
)
from utils.utils import load_train_config, load_yaml_with_inheritance  # noqa: E402


TRACKEVAL_SCRIPT = MEMOTR_ROOT.parent / "TrackEval" / "scripts" / "run_vt_tiny_mot.py"


def _default_eval_runs(data_root: str, dataset_dir: str, submit_data_split: str) -> list[dict[str, Any]]:
    """Two TrackEval runs per experiment (instances_00 / instances_01 GT)."""
    ann_dir = Path(data_root) / dataset_dir / "annotations"
    split_tag = f"{submit_data_split}2017"
    return [
        {
            "name": "eval_00",
            "gt_coco_ann": ann_dir / f"instances_00_{split_tag}.json",
            "output_sub_folder": "eval_00",
            "eval_class_agnostic": None,
        },
        {
            "name": "eval_01",
            "gt_coco_ann": ann_dir / f"instances_01_{split_tag}.json",
            "output_sub_folder": "eval_01",
            "eval_class_agnostic": None,
        },
    ]


@dataclass
class ThresholdParams:
    det: float
    track: float
    result: float
    update_thresh: float

    def tag(self) -> str:
        return (
            f"det{self.det:.2f}_track{self.track:.2f}_"
            f"result{self.result:.2f}_upd{self.update_thresh:.2f}"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MeMOTR 阈值参数分阶段扫描（submit + VT-Tiny-MOT TrackEval）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_GUIDE,
    )
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--config-path", type=str, required=True, help="Base yaml (e.g. 20260511-1.yaml)")
    p.add_argument("--submit-dir", type=str, required=True, help="Training output root (two-stage auto-resolve)")
    p.add_argument("--submit-model", type=str, required=True, help="Checkpoint filename under submit-dir")
    p.add_argument("--submit-data-split", type=str, default="test")
    p.add_argument("--scan-root", type=str, required=True, help="Root dir to store all sweep experiments")
    p.add_argument("--available-gpus", type=str, default=None, help="Override AVAILABLE_GPUS, e.g. '0'")
    p.add_argument("--submit-threads", type=int, default=None)
    p.add_argument("--iou-threshold", type=float, default=None, help="TrackEval IOU; default from yaml TRACK_IOU_THRESH")

    # Fixed params from previous stages (required for stage >= 2)
    p.add_argument("--det", type=float, default=None)
    p.add_argument("--track", type=float, default=None)
    p.add_argument("--result", type=float, default=None)
    p.add_argument("--update-thresh", type=float, default=None, dest="update_thresh")

    p.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit sweep values for current stage (overrides auto grid)",
    )
    p.add_argument("--num-experiments", type=int, default=5, help="Auto grid size when --values not set")
    p.add_argument("--skip-submit", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print planned experiments only")
    return p.parse_args()


def _load_base_config(config_path: str) -> dict:
    cfg = load_yaml_with_inheritance(config_path)
    cfg["CONFIG_PATH"] = config_path
    return cfg


def _resolve_submit_dir(config: dict, submit_dir: str) -> str:
    prefer = str(config.get("SUBMIT_STAGE_PREFER", "stage2")).lower()
    return resolve_two_stage_dir(submit_dir, prefer=prefer)


def _linspace_values(low: float, high: float, n: int) -> list[float]:
    if n <= 1:
        return [round(high, 4)]
    step = (high - low) / (n - 1)
    return [round(low + i * step, 4) for i in range(n)]


def _default_det_grid(n: int) -> list[float]:
    return _linspace_values(0.2, 0.8, n)


def _default_track_grid(det: float, n: int) -> list[float]:
    high = max(0.05, det - 0.05)
    low = min(0.15, high * 0.5)
    if low >= high:
        low = max(0.05, high - 0.1)
    return _linspace_values(low, high, n)


def _default_result_grid(track: float, det: float, n: int) -> list[float]:
    center = track
    low = max(0.1, center - 0.2)
    high = min(0.9, max(det, center) + 0.2)
    return _linspace_values(low, high, n)


def _default_update_grid(n: int) -> list[float]:
    return _linspace_values(0.3, 0.7, n)


def build_stage_experiments(
    stage: int,
    *,
    det: float | None,
    track: float | None,
    result: float | None,
    update_thresh: float | None,
    values: list[float] | None,
    num_experiments: int,
    default_update: float,
) -> list[ThresholdParams]:
    base_det = det if det is not None else 0.5
    base_track = track if track is not None else base_det
    base_result = result if result is not None else base_track
    base_update = update_thresh if update_thresh is not None else default_update

    if stage == 1:
        if det is not None or track is not None or result is not None:
            raise ValueError("Stage 1 should not set --det/--track/--result (only scans DET).")
        grid = values if values is not None else _default_det_grid(num_experiments)
        return [
            ThresholdParams(det=v, track=v, result=v, update_thresh=base_update)
            for v in grid
        ]

    if det is None:
        raise ValueError("Stage >= 2 requires --det from previous stage best.")

    if stage == 2:
        if track is not None or result is not None:
            raise ValueError("Stage 2 only scans TRACK; do not pass --track/--result.")
        grid = values if values is not None else _default_track_grid(det, num_experiments)
        for v in grid:
            if v >= det:
                raise ValueError(f"Stage 2 requires TRACK < DET ({det}), got {v}")
        return [
            ThresholdParams(det=det, track=v, result=v, update_thresh=base_update)
            for v in grid
        ]

    if track is None:
        raise ValueError("Stage >= 3 requires --track from previous stage best.")

    if stage == 3:
        if result is not None:
            raise ValueError("Stage 3 only scans RESULT; do not pass --result.")
        grid = values if values is not None else _default_result_grid(track, det, num_experiments)
        return [
            ThresholdParams(det=det, track=track, result=v, update_thresh=base_update)
            for v in grid
        ]

    if stage == 4:
        if result is None:
            raise ValueError("Stage 4 requires --result from previous stage best.")
        grid = values if values is not None else _default_update_grid(num_experiments)
        return [
            ThresholdParams(det=det, track=track, result=result, update_thresh=v)
            for v in grid
        ]

    raise ValueError(f"Unsupported stage: {stage}")


def _experiment_name(stage: int, params: ThresholdParams, idx: int) -> str:
    if stage == 1:
        return f"exp{idx:02d}_det{params.det:.2f}"
    if stage == 2:
        return f"exp{idx:02d}_track{params.track:.2f}"
    if stage == 3:
        return f"exp{idx:02d}_result{params.result:.2f}"
    return f"exp{idx:02d}_upd{params.update_thresh:.2f}"


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _run_submit_for_experiment(
    *,
    base_config: dict,
    resolved_submit_dir: str,
    submit_model: str,
    submit_data_split: str,
    params: ThresholdParams,
    exp_dir: Path,
    submit_threads: int,
) -> Path:
    config = copy.deepcopy(base_config)
    config["MODE"] = "submit"
    config["SUBMIT_DIR"] = resolved_submit_dir
    config["SUBMIT_MODEL"] = submit_model
    config["SUBMIT_DATA_SPLIT"] = submit_data_split
    config["DET_SCORE_THRESH"] = params.det
    config["TRACK_SCORE_THRESH"] = params.track
    config["RESULT_SCORE_THRESH"] = params.result
    config["UPDATE_THRESH"] = params.update_thresh
    if submit_threads is not None:
        config["SUBMIT_THREADS"] = submit_threads

    train_config = load_train_config(os.path.join(resolved_submit_dir, "train/config.yaml"))
    train_config = copy.deepcopy(train_config)
    # UPDATE_THRESH is consumed at model build time (QueryUpdater).
    train_config["UPDATE_THRESH"] = params.update_thresh

    dataset_name = train_config["DATASET"]
    config["DATASET"] = dataset_name
    dataset_split = submit_data_split
    outputs_dir = str(exp_dir / dataset_split)
    dataset_type = config.get("DATASET_TYPE", train_config.get("DATASET_TYPE", None))
    use_scem_gt = config.get("SCEM", {}).get("USE_GT", False)
    dataset_dir = train_config.get("DATASET_DIR", "VT-Tiny-MOT")
    dataset_version = config.get("DATASET_VERSION", train_config.get("DATASET_VERSION"))
    checkpoint_path = os.path.join(resolved_submit_dir, submit_model)

    data_split_dir = resolve_submit_split_dir(
        data_root=config["DATA_ROOT"],
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_version=dataset_version,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
    )
    seq_names = list_submit_sequences(data_split_dir, dataset_name)
    submit_workers = int(config.get("SUBMIT_THREADS", 1))
    dataloader_num_workers = resolve_submit_dataloader_workers(config, submit_workers)

    _save_yaml(exp_dir / "submit_config.yaml", config)
    _save_yaml(exp_dir / "train_config_override.yaml", {"UPDATE_THRESH": params.update_thresh})

    _run_submit_pipeline(
        config=config,
        train_config=train_config,
        outputs_dir=outputs_dir,
        data_split_dir=data_split_dir,
        dataset_name=dataset_name,
        seq_names=seq_names,
        logger=None,
        use_scem_gt=use_scem_gt,
        dataset_type=dataset_type,
        checkpoint_path=checkpoint_path,
        source_model=None,
        only_train_detr=train_config.get("ONLY_TRAIN_DETR", False),
        epoch=None,
        draw_pic_dir=None,
    )
    return Path(outputs_dir)


def _run_trackeval(
    *,
    trackers_folder: Path,
    trackers_to_eval: str,
    gt_coco_ann: Path,
    img_folder: Path,
    iou_threshold: float,
    eval_output_subfolder: str = "eval",
    eval_class_agnostic: bool | None = None,
) -> Path:
    if not TRACKEVAL_SCRIPT.is_file():
        raise FileNotFoundError(f"TrackEval script not found: {TRACKEVAL_SCRIPT}")
    if not gt_coco_ann.is_file():
        raise FileNotFoundError(f"GT COCO annotation not found: {gt_coco_ann}")

    cmd = [
        sys.executable,
        str(TRACKEVAL_SCRIPT),
        "--USE_PARALLEL", "False",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--GT_COCO_ANN", str(gt_coco_ann),
        "--IMG_FOLDER", str(img_folder),
        "--TRACKERS_FOLDER", str(trackers_folder.parent),
        "--TRACKERS_TO_EVAL", trackers_to_eval,
        "--TRACKER_SUB_FOLDER", "tracker",
        "--IOU_THRESHOLD", str(iou_threshold),
        "--OUTPUT_SUB_FOLDER", eval_output_subfolder,
    ]
    if eval_class_agnostic is not None:
        cmd.extend(["--EVAL_CLASS_AGNOSTIC", "True" if eval_class_agnostic else "False"])
    print("[eval]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(TRACKEVAL_SCRIPT.parent.parent))
    return trackers_folder / eval_output_subfolder


def _run_all_trackevals(
    *,
    outputs_dir: Path,
    eval_runs: list[dict[str, Any]],
    img_folder: Path,
    iou_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Run multiple TrackEval configs; return {eval_name: {dir, metrics?}}."""
    results: dict[str, dict[str, Any]] = {}
    trackers_to_eval = outputs_dir.name
    for run in eval_runs:
        eval_dir = _run_trackeval(
            trackers_folder=outputs_dir,
            trackers_to_eval=trackers_to_eval,
            gt_coco_ann=Path(run["gt_coco_ann"]),
            img_folder=img_folder,
            iou_threshold=iou_threshold,
            eval_output_subfolder=run["output_sub_folder"],
            eval_class_agnostic=run.get("eval_class_agnostic"),
        )
        record: dict[str, Any] = {
            "dir": str(eval_dir),
            "gt_coco_ann": str(run["gt_coco_ann"]),
            "output_sub_folder": run["output_sub_folder"],
        }
        hota = _try_read_hota_summary(eval_dir)
        if hota is not None:
            record["metrics"] = hota
        results[run["name"]] = record
    return results


def _try_read_hota_summary(eval_dir: Path) -> dict[str, Any] | None:
    """Best-effort parse combined HOTA from TrackEval summary csv."""
    if not eval_dir.is_dir():
        return None
    candidates = sorted(eval_dir.rglob("*HOTA*.csv"))
    if not candidates:
        candidates = sorted(eval_dir.rglob("*.csv"))
    for csv_path in candidates:
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if len(rows) < 2:
                continue
            header = rows[0]
            values = rows[1]
            summary = dict(zip(header, values))
            if "HOTA" in summary:
                return {"file": str(csv_path), "HOTA": float(summary["HOTA"])}
            for key in ("HOTA(0)", "HOTA___HOTA"):
                if key in summary:
                    return {"file": str(csv_path), "HOTA": float(summary[key])}
        except (OSError, ValueError, IndexError):
            continue
    return None


def _write_stage_summary(stage_dir: Path, stage: int, manifest: dict[str, Any]) -> Path:
    """Write a human-readable markdown summary for manual comparison."""
    summary_path = stage_dir / "summary.md"
    lines = [
        f"# Stage {stage} Threshold Scan Summary",
        "",
        f"- Created: {manifest.get('created_at', '')}",
        f"- Config: `{manifest.get('config_path', '')}`",
        f"- Checkpoint: `{manifest.get('submit_model', '')}`",
        "",
        "| Exp | DET | TRACK | RESULT | UPDATE | HOTA_00 | HOTA_01 | Status |",
        "|-----|-----|-------|--------|--------|---------|---------|--------|",
    ]
    for exp in manifest.get("experiments", []):
        p = exp.get("params", {})
        evals = exp.get("evals", {})
        hota_00 = evals.get("eval_00", {}).get("metrics", {}).get("HOTA", "-")
        hota_01 = evals.get("eval_01", {}).get("metrics", {}).get("HOTA", "-")
        lines.append(
            f"| {exp.get('name', '-')} "
            f"| {p.get('det', '-')} "
            f"| {p.get('track', '-')} "
            f"| {p.get('result', '-')} "
            f"| {p.get('update_thresh', '-')} "
            f"| {hota_00} "
            f"| {hota_01} "
            f"| {exp.get('status', '-')} |"
        )
    lines.extend([
        "",
        "## Next step",
        "",
        "Pick the best experiment, then run the next stage with the corresponding flags:",
        "",
        "```bash",
        "conda activate hsmot",
        "cd /data1/users/litianhao01/hsmot/MeMOTR",
        "# stage 2 example (after picking best DET=0.5 from stage 1):",
        "bash scripts/run_thresh_scan.sh --stage 2 --det 0.5 ...",
        "```",
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _validate_checkpoint(resolved_submit_dir: str, submit_model: str) -> None:
    ckpt = os.path.join(resolved_submit_dir, submit_model)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")


def main() -> None:
    args = _parse_args()
    base_config = _load_base_config(args.config_path)
    if args.available_gpus is not None:
        base_config["AVAILABLE_GPUS"] = args.available_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(base_config.get("AVAILABLE_GPUS", "0"))

    resolved_submit_dir = _resolve_submit_dir(base_config, args.submit_dir)
    if not args.dry_run:
        _validate_checkpoint(resolved_submit_dir, args.submit_model)
    default_update = float(base_config.get("UPDATE_THRESH", 0.5))
    experiments = build_stage_experiments(
        args.stage,
        det=args.det,
        track=args.track,
        result=args.result,
        update_thresh=args.update_thresh,
        values=args.values,
        num_experiments=args.num_experiments,
        default_update=default_update,
    )

    stage_dir = Path(args.scan_root) / f"stage{args.stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "stage": args.stage,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": args.config_path,
        "submit_dir": args.submit_dir,
        "resolved_submit_dir": resolved_submit_dir,
        "submit_model": args.submit_model,
        "experiments": [],
    }

    print(f"Stage {args.stage}: {len(experiments)} experiment(s)")
    for i, params in enumerate(experiments, start=1):
        exp_name = _experiment_name(args.stage, params, i)
        exp_dir = stage_dir / exp_name
        print(f"  [{i}/{len(experiments)}] {exp_name}  params={params}")

        if args.dry_run:
            manifest["experiments"].append(
                {"name": exp_name, "dir": str(exp_dir), "params": asdict(params), "status": "dry_run"}
            )
            continue

        exp_dir.mkdir(parents=True, exist_ok=True)
        with (exp_dir / "params.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(params), f, indent=2)

        exp_record: dict[str, Any] = {
            "name": exp_name,
            "dir": str(exp_dir),
            "params": asdict(params),
        }

        try:
            if not args.skip_submit:
                outputs_dir = _run_submit_for_experiment(
                    base_config=base_config,
                    resolved_submit_dir=resolved_submit_dir,
                    submit_model=args.submit_model,
                    submit_data_split=args.submit_data_split,
                    params=params,
                    exp_dir=exp_dir,
                    submit_threads=args.submit_threads,
                )
            else:
                outputs_dir = exp_dir / args.submit_data_split

            exp_record["tracker_dir"] = str(outputs_dir / "tracker")

            if not args.skip_eval:
                train_config = load_train_config(os.path.join(resolved_submit_dir, "train/config.yaml"))
                dataset_name = train_config["DATASET"]
                if not is_vt_tiny_dataset(dataset_name):
                    raise RuntimeError(f"Only vt_tiny_mot supported, got {dataset_name}")

                dataset_dir = train_config.get("DATASET_DIR", "VT-Tiny-MOT")
                dataset_version = base_config.get("DATASET_VERSION", train_config.get("DATASET_VERSION"))
                dataset_type = base_config.get("DATASET_TYPE", train_config.get("DATASET_TYPE", None))
                data_split_dir = resolve_submit_split_dir(
                    data_root=base_config["DATA_ROOT"],
                    dataset_name=dataset_name,
                    dataset_split=args.submit_data_split,
                    dataset_version=dataset_version,
                    dataset_type=dataset_type,
                    dataset_dir=dataset_dir,
                )
                iou = args.iou_threshold
                if iou is None:
                    iou = float(base_config.get("EVAL_IOU_THRESHOLD", base_config.get("TRACK_IOU_THRESH", 0.3)))

                eval_runs = _default_eval_runs(
                    base_config["DATA_ROOT"], dataset_dir, args.submit_data_split
                )
                eval_results = _run_all_trackevals(
                    outputs_dir=Path(outputs_dir),
                    eval_runs=eval_runs,
                    img_folder=Path(data_split_dir),
                    iou_threshold=iou,
                )
                exp_record["evals"] = eval_results
                # backward-compatible single eval_dir (eval_00)
                exp_record["eval_dir"] = eval_results.get("eval_00", {}).get("dir")
                if eval_results.get("eval_00", {}).get("metrics") is not None:
                    exp_record["metrics"] = eval_results["eval_00"]["metrics"]

            exp_record["status"] = "ok"
        except Exception as exc:
            exp_record["status"] = "failed"
            exp_record["error"] = repr(exc)
            print(f"    FAILED: {exc!r}")

        manifest["experiments"].append(exp_record)

    manifest_path = stage_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    summary_path = _write_stage_summary(stage_dir, args.stage, manifest)
    print(f"\nManifest saved: {manifest_path}")
    print(f"Summary saved:  {summary_path}")
    print("Review eval results under each experiment dir, then run the next stage with --det/--track/--result.")


if __name__ == "__main__":
    main()
