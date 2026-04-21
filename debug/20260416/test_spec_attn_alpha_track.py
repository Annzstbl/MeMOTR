import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 checkpoint_xx.pth 顺序统计所有层的 spec_attn_alpha，"
            "并记录每个 checkpoint 相对上一个 checkpoint 的变化量。"
        )
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default = "/data4/litianhao/hsmot/memotr/spectralemb/20260420-1",
        help="checkpoint 目录，脚本会读取其中 checkpoint_*.pth 文件。",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="checkpoint_*.pth",
        help="checkpoint 文件匹配模式，默认 checkpoint_*.pth。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认当前脚本目录下/ckpt目录名/。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch.load 的 map_location，默认 cpu。",
    )
    return parser.parse_args()


def _extract_ckpt_index(path: Path) -> Tuple[int, str]:
    match = re.search(r"checkpoint_(\d+)\.pth$", path.name)
    if match:
        return int(match.group(1)), path.name
    # 不匹配时放到最后，并按文件名稳定排序
    return 10**18, path.name


def list_checkpoint_files(ckpt_dir: Path, pattern: str) -> List[Path]:
    files = sorted(ckpt_dir.glob(pattern), key=_extract_ckpt_index)
    return [p for p in files if p.is_file()]


def _is_state_dict_like(obj: object) -> bool:
    if not isinstance(obj, Mapping) or len(obj) == 0:
        return False
    sample_values = list(obj.values())[:50]
    return any(torch.is_tensor(v) for v in sample_values)


def resolve_state_dict(ckpt_obj: object) -> Mapping[str, torch.Tensor]:
    if _is_state_dict_like(ckpt_obj):
        return ckpt_obj  # type: ignore[return-value]

    if not isinstance(ckpt_obj, Mapping):
        raise ValueError("checkpoint 内容不是字典，无法解析 state_dict。")

    candidates = [
        "state_dict",
        "model",
        "model_state_dict",
        "net",
        "module",
    ]
    for key in candidates:
        value = ckpt_obj.get(key, None)
        if _is_state_dict_like(value):
            return value  # type: ignore[return-value]

    # 兜底：在顶层字段里找第一个像 state_dict 的子字典
    for _, value in ckpt_obj.items():
        if _is_state_dict_like(value):
            return value  # type: ignore[return-value]

    raise ValueError("未在 checkpoint 中找到可用的 state_dict。")


def extract_spec_attn_alpha(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in state_dict.items():
        if "spec_attn_alpha" not in key:
            continue
        if not torch.is_tensor(value):
            continue
        if value.numel() != 1:
            continue
        result[key] = float(value.detach().cpu().item())
    return dict(sorted(result.items()))


def write_wide_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    layer_names: Sequence[str],
) -> None:
    fields = ["checkpoint", "checkpoint_idx"] + list(layer_names)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            writer.writerow(out)


def write_long_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    layer_names: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "checkpoint_idx",
                "layer",
                "value",
                "delta_from_prev",
            ],
        )
        writer.writeheader()
        for row in rows:
            for layer in layer_names:
                value = row.get(layer, None)
                delta = row.get(f"{layer}__delta", None)
                writer.writerow(
                    {
                        "checkpoint": row["checkpoint"],
                        "checkpoint_idx": row["checkpoint_idx"],
                        "layer": layer,
                        "value": "" if value is None else value,
                        "delta_from_prev": "" if delta is None else delta,
                    }
                )


def format_console_table(
    rows: Sequence[Dict[str, object]],
    layer_names: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append("=== spec_attn_alpha 变化概览（每层最后值与总变化）===")
    for layer in layer_names:
        values: List[float] = []
        for row in rows:
            v = row.get(layer, None)
            if isinstance(v, float):
                values.append(v)
        if not values:
            continue
        total_delta = values[-1] - values[0]
        lines.append(
            f"- {layer}: first={values[0]:.8f}, last={values[-1]:.8f}, "
            f"delta={total_delta:+.8f}, points={len(values)}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint 目录不存在: {ckpt_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (SCRIPT_DIR / ckpt_dir.name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_files = list_checkpoint_files(ckpt_dir, args.pattern)
    if not ckpt_files:
        raise FileNotFoundError(
            f"在目录 {ckpt_dir} 下未找到匹配 {args.pattern} 的 checkpoint 文件。"
        )

    per_layer_values = defaultdict(list)  # type: Dict[str, List[Tuple[str, float]]]
    rows: List[Dict[str, object]] = []

    for ckpt_path in ckpt_files:
        ckpt_idx, _ = _extract_ckpt_index(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        state_dict = resolve_state_dict(checkpoint)
        alpha_dict = extract_spec_attn_alpha(state_dict)

        row: Dict[str, object] = {
            "checkpoint": ckpt_path.name,
            "checkpoint_idx": "" if ckpt_idx >= 10**18 else ckpt_idx,
        }
        for layer_name, value in alpha_dict.items():
            row[layer_name] = value
            per_layer_values[layer_name].append((ckpt_path.name, value))

        rows.append(row)

    layer_names = sorted(per_layer_values.keys())
    if not layer_names:
        raise ValueError("未在任何 checkpoint 中找到 spec_attn_alpha 参数。")

    # 计算相邻 checkpoint 的变化量（delta）
    prev_values: Dict[str, Optional[float]] = {name: None for name in layer_names}
    for row in rows:
        for layer in layer_names:
            v = row.get(layer, None)
            if isinstance(v, float):
                prev = prev_values[layer]
                row[f"{layer}__delta"] = None if prev is None else (v - prev)
                prev_values[layer] = v
            else:
                row[f"{layer}__delta"] = None

    wide_path = output_dir / "spec_attn_alpha_track_wide.csv"
    long_path = output_dir / "spec_attn_alpha_track_long.csv"
    write_wide_csv(wide_path, rows, layer_names)
    write_long_csv(long_path, rows, layer_names)

    print(f"已处理 checkpoint 数量: {len(rows)}")
    print(f"检测到 spec_attn_alpha 层数量: {len(layer_names)}")
    print(f"宽表输出: {wide_path}")
    print(f"长表输出: {long_path}")
    print()
    print(format_console_table(rows, layer_names))


if __name__ == "__main__":
    main()
