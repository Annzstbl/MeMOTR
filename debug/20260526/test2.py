"""
加载 checkpoint，对一条或多条视频序列做序贯推理，保存 SCEM token_debug 中的可视化图。

使用 model_20260511_figure：可视化由模型 forward 内直接落盘。
默认只保存 pool_weights；通过 --save-maps 可扩展其它图。

Track query 可视化示例：
CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/test2.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data31-1 \
  --end-frames 69 \
  --save-last-n-frames 0 \
  --vis-track-id 0 \
  --vis-track-spectral \
  --vis-cross-attn \
  --save-maps none

  CUDA_VISIBLE_DEVICES=0 \
python debug/20260526/test2.py \
  --train-config 20260511-2.yaml \
  --checkpoint last.pth \
  --seq data33-1 \
  --end-frames 137 \
  --save-last-n-frames 0 \
  --vis-track-id 4 \
  --vis-track-spectral \
  --vis-cross-attn \
  --save-maps none
"""

import os
import sys
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DEBUG_DIR not in sys.path:
    sys.path.insert(0, DEBUG_DIR)

from models import build_model
from models.matcher import is_rect_memotr_version
from models.runtime_tracker import RuntimeTracker
from models.utils import load_checkpoint, get_model
from models.model_20260511_figure.figure_context import FigureContext
from models.model_20260511_figure.figure_save import MAP_SAVER_REGISTRY, save_track_spectral_timeline
from utils.utils import load_yaml_with_inheritance
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.GMC import compute_gmc_sequence
from structures.track_instances import TrackInstances
from data.seq_dataset import SeqDataset


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

def _resolve_train_config_path(train_config_arg: str) -> str:
    if os.path.isabs(train_config_arg):
        return train_config_arg
    return os.path.abspath(os.path.join(CURRENT_DIR, train_config_arg))


def _resolve_checkpoint_path(checkpoint_arg: str, train_config: dict, train_cfg_path: str) -> str:
    if os.path.isabs(checkpoint_arg):
        return checkpoint_arg
    outputs_dir = train_config.get("OUTPUTS_DIR", "")
    if not outputs_dir:
        raise ValueError("Relative checkpoint path requires OUTPUTS_DIR in train config.")
    if not os.path.isabs(outputs_dir):
        outputs_dir = os.path.abspath(os.path.join(os.path.dirname(train_cfg_path), outputs_dir))
    return os.path.abspath(os.path.join(outputs_dir, checkpoint_arg))


def _normalize_img_format(fmt: str) -> Tuple[str, str]:
    key = fmt.strip().lower()
    if key in ("npy2jpg", "3jpg", "jpg"):
        return "npy2jpg", "3JPG"
    if key in ("npy",):
        return "npy", "NPY"
    raise ValueError(f"Unsupported img format: {fmt!r}. Use npy2jpg or npy.")


def _resolve_img_format(img_format_arg: Optional[str], train_config: dict) -> Tuple[str, str]:
    if img_format_arg:
        return _normalize_img_format(img_format_arg)
    cfg_type = str(train_config.get("DATASET_TYPE", "3JPG")).upper()
    if cfg_type == "3JPG":
        return "npy2jpg", "3JPG"
    return "npy", "NPY"


def _resolve_img_root(data_root: str, base_name: str, split: str, img_subdir: str) -> str:
    return os.path.join(data_root, base_name, split, img_subdir)


def _resolve_config_root(train_cfg_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        return os.path.abspath(output_dir)
    config_name = os.path.splitext(os.path.basename(train_cfg_path))[0]
    return os.path.join(CURRENT_DIR, config_name)


def _resolve_track_vis_out_dir(config_root: str, seq: str, track_id: int) -> str:
    return os.path.join(config_root, "track", seq, f"id{track_id}")


def _init_track_instances_static(train_config: dict, model: nn.Module) -> None:
    rect_bbox = is_rect_memotr_version(train_config.get("MEMOTR_VERSION", ""))
    TrackInstances.set_static_properties(
        use_spectral_decoder=train_config.get("DECODER_SPECTRAL", True),
        use_dab=train_config["USE_DAB"],
        use_q_spec=bool(getattr(get_model(model), "use_q_spec", False)),
        bbox_dim=4 if rect_bbox else 5,
    )


def _parse_seq_specs(
    seq_arg: str,
    start_frames_arg: Optional[str],
    end_frames_arg: Optional[str],
    default_start: int,
    default_end: Optional[int],
) -> List[Dict[str, Any]]:
    seq_list = [s.strip() for s in seq_arg.split(",") if s.strip()]
    if not seq_list:
        raise ValueError("Empty --seq.")

    def _expand_values(raw: Optional[str], n: int, name: str) -> List[Optional[int]]:
        if raw is None:
            return [None] * n
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == 1:
            return [int(parts[0])] * n
        if len(parts) != n:
            raise ValueError(f"--{name} count ({len(parts)}) must match --seq count ({n}) or be 1.")
        return [int(p) for p in parts]

    starts = _expand_values(start_frames_arg, len(seq_list), "start-frames")
    ends = _expand_values(end_frames_arg, len(seq_list), "end-frames")

    specs: List[Dict[str, Any]] = []
    for i, seq in enumerate(seq_list):
        start = starts[i] if starts[i] is not None else default_start
        end = ends[i] if ends[i] is not None else default_end
        if start < 1:
            raise ValueError(f"start frame must be >= 1, got {start} for seq {seq}")
        if end is not None and end < start:
            raise ValueError(f"end frame ({end}) must be >= start frame ({start}) for seq {seq}")
        specs.append({"seq": seq, "start_frame": start, "end_frame": end})
    return specs


def _resolve_save_maps(raw: str) -> List[str]:
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if not names:
        return ["pool_weights"]
    if len(names) == 1 and names[0].lower() == "none":
        return []
    if len(names) == 1 and names[0].lower() == "all":
        return list(MAP_SAVER_REGISTRY.keys())
    unknown = [n for n in names if n not in MAP_SAVER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown save-maps: {unknown}. Available: {list(MAP_SAVER_REGISTRY.keys())}, all, none"
        )
    return names


def _build_frame_meta(frame_num: int, outputs: Dict[str, Any]) -> Dict[str, Any]:
    frame_meta: Dict[str, Any] = {"frame": frame_num, "outputs": {}}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            frame_meta["outputs"][k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        elif k == "scem_token_debug" and isinstance(v, dict):
            frame_meta["outputs"][k] = {
                sub_k: ("tensor" if torch.is_tensor(sub_v) else type(sub_v).__name__)
                for sub_k, sub_v in v.items()
            }
    return frame_meta


def _find_track_local_index(tracks: TrackInstances, track_id: int) -> Optional[int]:
    if len(tracks) == 0:
        return None
    ids = tracks.ids.detach().cpu()
    matches = (ids == track_id).nonzero(as_tuple=False).view(-1)
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"[Warn] Multiple track queries with id={track_id}, use first index {matches[0].item()}")
    return int(matches[0].item())


def _global_query_index(n_det_queries: int, track_local_idx: int) -> int:
    return n_det_queries + track_local_idx


def _build_figure_context(
    *,
    enabled: bool,
    seq: str,
    frame_num: int,
    map_out_dir: str,
    track_vis_out_dir: Optional[str],
    save_maps: List[str],
    vis_track_id: Optional[int],
    track_local_idx: Optional[int],
    global_q_idx: Optional[int],
    vis_track_spectral: bool,
    vis_cross_attn: bool,
    skip_existing: bool,
    vis_max_size: int,
    frame_pad_mask: Optional[torch.Tensor],
    spectral_timeline: Optional[List[Dict[str, Any]]],
) -> Optional[FigureContext]:
    if not enabled:
        return None
    return FigureContext(
        enabled=True,
        seq=seq,
        frame_num=frame_num,
        scem_out_dir=map_out_dir,
        track_out_dir=track_vis_out_dir or "",
        save_maps=tuple(save_maps),
        track_id=vis_track_id,
        track_local_idx=track_local_idx,
        global_q_idx=global_q_idx,
        save_track_spectral=vis_track_spectral,
        save_cross_attn=vis_cross_attn,
        skip_existing=skip_existing,
        vis_max_size=vis_max_size,
        frame_pad_mask=frame_pad_mask,
        spectral_timeline=spectral_timeline,
    )


def run_forward_sequential(
    model: nn.Module,
    train_config: dict,
    seq_dir: str,
    npy2rgb: bool,
    dataset_type: Optional[str],
    start_frame: int,
    end_frame: Optional[int],
    seq: str,
    map_out_dir: str,
    save_maps: List[str],
    save_last_n_frames: int = 30,
    skip_existing_images: bool = False,
    vis_max_size: int = 1200,
    vis_track_id: Optional[int] = None,
    vis_track_spectral: bool = False,
    vis_cross_attn: bool = False,
    track_vis_out_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    dataset = SeqDataset(seq_dir=seq_dir, npy2rgb=npy2rgb, dataset_type=dataset_type)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {seq_dir}")

    start_idx = max(0, start_frame - 1)
    if start_idx >= len(dataset):
        raise RuntimeError(
            f"Start frame {start_frame} exceeds dataset length {len(dataset)} at {seq_dir}"
        )

    if end_frame is None:
        end_idx = len(dataset) - 1
    else:
        end_idx = min(len(dataset) - 1, end_frame - 1)
    if end_idx < start_idx:
        raise RuntimeError(
            f"Invalid frame range [{start_frame}, {end_frame}] for dataset length {len(dataset)}"
        )

    num_frames = end_idx - start_idx + 1
    actual_end_frame = start_frame + num_frames - 1
    save_from_frame = (
        max(start_frame, actual_end_frame - save_last_n_frames + 1)
        if save_last_n_frames > 0
        else start_frame
    )

    print(
        f"[Info] Processing frames {start_frame}-{actual_end_frame} "
        f"({num_frames} frames, dataset total={len(dataset)})"
    )
    if save_from_frame > start_frame:
        print(
            f"[Info] Save maps for last {save_last_n_frames} frames only: "
            f"{save_from_frame}-{actual_end_frame}"
        )
    else:
        print(f"[Info] Save maps for frames {save_from_frame}-{actual_end_frame}")

    inner_model = get_model(model)
    tracker = RuntimeTracker(
        det_score_thresh=0.5,
        track_score_thresh=0.5,
        miss_tolerance=30,
        use_motion=False,
        motion_min_length=3,
        motion_max_length=5,
        visualize=False,
        use_dab=train_config["USE_DAB"],
        decoder_spectral=train_config.get("DECODER_SPECTRAL", True),
    )

    tracks = [
        TrackInstances(
            hidden_dim=inner_model.hidden_dim,
            num_classes=inner_model.num_classes,
        ).to(device)
    ]

    use_prior_map = (
        hasattr(inner_model, "scem_module")
        and inner_model.scem_module is not None
        and inner_model.scem_module.prior_mode is not None
    )

    saved_meta: List[Dict[str, Any]] = []
    prev_frame = None
    track_vis_enabled = vis_track_id is not None and (vis_track_spectral or vis_cross_attn)
    spectral_timeline: List[Dict[str, Any]] = []
    if track_vis_enabled:
        if not track_vis_out_dir:
            raise ValueError("track_vis_out_dir is required when track visualization is enabled.")
        os.makedirs(track_vis_out_dir, exist_ok=True)
        print(
            f"[Info] Track vis enabled: id={vis_track_id}, "
            f"spectral={vis_track_spectral}, cross_attn={vis_cross_attn}, "
            f"out={track_vis_out_dir}"
        )

    model.eval()
    with torch.no_grad():
        for i in range(num_frames):
            dataset_idx = start_idx + i
            frame_num = start_frame + i
            should_save = frame_num >= save_from_frame
            need_debug = should_save or track_vis_enabled
            log_tag = "forward+save" if should_save else ("forward+track_vis" if track_vis_enabled else "forward")
            print(f"[Info] {log_tag} frame {frame_num} (index {dataset_idx})")

            track_local_idx = None
            global_q_idx = None
            if track_vis_enabled:
                track_local_idx = _find_track_local_index(tracks[0], vis_track_id)
                if track_local_idx is not None:
                    global_q_idx = _global_query_index(inner_model.n_det_queries, track_local_idx)
                    print(
                        f"[Info] frame {frame_num}: track id {vis_track_id} "
                        f"local={track_local_idx}, global_q={global_q_idx}"
                    )
                else:
                    print(f"[Warn] frame {frame_num}: track id {vis_track_id} not in input tracks, skip track vis")

            image, _ = dataset[dataset_idx][0]
            frame = tensor_list_to_nested_tensor([image]).to(device)

            gmc = None
            if use_prior_map:
                if prev_frame is not None:
                    gmc = compute_gmc_sequence(
                        images=[prev_frame[0], frame.tensors[0]],
                        method="sparseOptFlow",
                        downscale=1,
                    )[-1]
                else:
                    gmc = np.eye(2, 3, dtype=np.float32)
                prev_frame = frame.tensors.detach().clone()
                gmc = torch.tensor(gmc, dtype=torch.float32).unsqueeze(0).to(device)

            figure = _build_figure_context(
                enabled=should_save or track_vis_enabled,
                seq=seq,
                frame_num=frame_num,
                map_out_dir=map_out_dir,
                track_vis_out_dir=track_vis_out_dir,
                save_maps=save_maps if should_save else [],
                vis_track_id=vis_track_id,
                track_local_idx=track_local_idx,
                global_q_idx=global_q_idx,
                vis_track_spectral=vis_track_spectral,
                vis_cross_attn=vis_cross_attn,
                skip_existing=skip_existing_images,
                vis_max_size=vis_max_size,
                frame_pad_mask=frame.masks[0].detach() if should_save else None,
                spectral_timeline=spectral_timeline if track_vis_enabled else None,
            )

            forward_kwargs = dict(frame=frame, tracks=tracks, debug=need_debug)
            if figure is not None:
                forward_kwargs["figure"] = figure
            if gmc is not None:
                forward_kwargs["gmc"] = gmc
            res = model(**forward_kwargs)

            previous_tracks, new_tracks = tracker.update(model_outputs=res, tracks=tracks)
            tracks = inner_model.postprocess_single_frame(previous_tracks, new_tracks, None)

            if should_save and isinstance(res, dict):
                saved_meta.append(_build_frame_meta(frame_num, res))

            del res, frame, image

    if track_vis_enabled and vis_track_spectral and track_vis_out_dir and vis_track_id is not None:
        save_track_spectral_timeline(
            timeline=spectral_timeline,
            figure=FigureContext(
                enabled=True,
                seq=seq,
                track_id=vis_track_id,
                track_out_dir=track_vis_out_dir,
                skip_existing=skip_existing_images,
                vis_max_size=vis_max_size,
            ),
        )

    return saved_meta


def main():
    parser = argparse.ArgumentParser(
        description="Load model_20260511_figure, run sequential inference, save debug figures."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="20260511-2.yaml",
        help="Train config yaml (absolute, or relative to this script dir).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.pth",
        help="Checkpoint path (absolute, or relative to OUTPUTS_DIR in train config).",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override DATA_ROOT from train config.")
    parser.add_argument("--dataset-name", type=str, default="hsmot_8ch")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seq", type=str, default="data39-1", help="Comma-separated sequence names.")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--start-frames", type=str, default=None)
    parser.add_argument("--end-frames", type=str, default="1")
    parser.add_argument(
        "--img-format",
        type=str,
        default="npy2jpg",
        choices=["npy2jpg", "npy"],
    )
    parser.add_argument("--npy2rgb", action="store_true")
    parser.add_argument(
        "--save-maps",
        type=str,
        default="pool_weights",
        help=f"Comma-separated SCEM map types; use none to skip SCEM. Available: {','.join(MAP_SAVER_REGISTRY.keys())}, all, none",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--save-last-n-frames",
        type=int,
        default=30,
        help="Only save SCEM maps for the last N frames. Use 0 to save all frames.",
    )
    parser.add_argument("--skip-existing-images", action="store_true")
    parser.add_argument("--vis-max-size", type=int, default=1200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dump-json", type=str, default="")
    parser.add_argument("--vis-track-id", type=int, default=None)
    parser.add_argument("--vis-track-spectral", action="store_true")
    parser.add_argument(
        "--vis-cross-attn",
        action="store_true",
        help="保存 decoder 各层 cross-attention 空间 attn 图（由 model_20260511_figure 内部捕获）。",
    )

    args = parser.parse_args()
    save_maps = _resolve_save_maps(args.save_maps)
    if args.vis_track_id is not None and not (args.vis_track_spectral or args.vis_cross_attn):
        raise ValueError("--vis-track-id 需要同时指定 --vis-track-spectral 和/或 --vis-cross-attn")

    train_cfg_path = _resolve_train_config_path(args.train_config)
    if not os.path.isfile(train_cfg_path):
        raise FileNotFoundError(f"Config not found: {train_cfg_path}")
    train_config = load_yaml_with_inheritance(path=train_cfg_path)

    data_root_arg = args.data_root
    if data_root_arg is None:
        args.data_root = train_config.get("DATA_ROOT", "") or ""
    if not args.data_root:
        raise ValueError("DATA_ROOT is empty: set in train config or pass --data-root.")
    if not os.path.isabs(args.data_root):
        if data_root_arg is None:
            args.data_root = os.path.abspath(
                os.path.join(os.path.dirname(train_cfg_path), args.data_root)
            )
        else:
            args.data_root = os.path.abspath(args.data_root)

    config_root = _resolve_config_root(train_cfg_path, args.output_dir)
    heatmap_root = os.path.join(config_root, "heatmaps")
    os.makedirs(heatmap_root, exist_ok=True)

    end_frames_raw = args.end_frames
    if end_frames_raw is not None and end_frames_raw.strip().lower() in ("none", "all"):
        end_frames_raw = None

    base_name = args.dataset_name.replace("_8ch", "") if "hsmot" in args.dataset_name else args.dataset_name
    img_subdir, dataset_type = _resolve_img_format(args.img_format, train_config)
    img_root = _resolve_img_root(args.data_root, base_name, args.split, img_subdir)
    print(f"[Info] img format={args.img_format} -> {img_root} (dataset_type={dataset_type})")

    seq_arg = args.seq
    if seq_arg.lower() == "all":
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"image root not found: {img_root}")
        seq_arg = ",".join(
            sorted(d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d)))
        )
        print(f"[Info] --seq all -> {len(seq_arg.split(','))} sequences")

    seq_specs = _parse_seq_specs(
        seq_arg=seq_arg,
        start_frames_arg=args.start_frames,
        end_frames_arg=end_frames_raw,
        default_start=args.start_frame,
        default_end=None,
    )

    print(f"[Info] Building model from: {train_cfg_path}")
    train_config["MEMOTR_VERSION"] = "20260511_figure"
    print("[Info] Using MEMOTR_VERSION='20260511_figure'")
    model = build_model(config=train_config)
    model.to(torch.device(args.device))

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, train_config, train_cfg_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[Info] Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model=model, path=checkpoint_path)
    _init_track_instances_static(train_config, model)

    all_meta: Dict[str, Any] = {"config": train_cfg_path, "checkpoint": checkpoint_path, "per_seq": []}

    for spec in seq_specs:
        seq = spec["seq"]
        seq_dir = os.path.join(img_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[Warn] Skip missing seq dir: {seq_dir}")
            continue

        print(f"\n[Info] seq={seq}, frames [{spec['start_frame']}, {spec['end_frame'] or 'end'}]")
        map_out_dir = os.path.join(heatmap_root, seq)
        os.makedirs(map_out_dir, exist_ok=True)

        track_vis_out_dir = None
        if args.vis_track_id is not None:
            track_vis_out_dir = _resolve_track_vis_out_dir(config_root, seq, args.vis_track_id)

        saved_frames = run_forward_sequential(
            model=model,
            train_config=train_config,
            seq_dir=seq_dir,
            npy2rgb=args.npy2rgb,
            dataset_type=dataset_type,
            start_frame=spec["start_frame"],
            end_frame=spec["end_frame"],
            seq=seq,
            map_out_dir=map_out_dir,
            save_maps=save_maps,
            save_last_n_frames=args.save_last_n_frames,
            skip_existing_images=args.skip_existing_images,
            vis_max_size=args.vis_max_size,
            vis_track_id=args.vis_track_id,
            vis_track_spectral=args.vis_track_spectral,
            vis_cross_attn=args.vis_cross_attn,
            track_vis_out_dir=track_vis_out_dir,
        )
        all_meta["per_seq"].append({"seq": seq, "frames": saved_frames})

    if args.dump_json:
        try:
            with open(args.dump_json, "w") as f:
                json.dump(all_meta, f, indent=2)
            print(f"[Info] Dumped metadata to {args.dump_json}")
        except Exception as e:
            print(f"[Warn] Failed to dump json: {e}")

    print(f"\n[Done] SCEM heatmaps under: {heatmap_root}")
    if args.vis_track_id is not None:
        print(f"[Done] Track vis under: {os.path.join(config_root, 'track')}")


if __name__ == "__main__":
    main()
