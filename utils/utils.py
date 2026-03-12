# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Some utils.
import os
import yaml
import torch
import random
import torch.distributed
import torch.backends.cudnn
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Set
import inspect
from collections import defaultdict


def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def is_main_process():
    return distributed_rank() == 0


def distributed_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1


def set_seed(seed: int):
    seed = seed + distributed_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If you don't want to wait until the universe is silent, do not use this below code :)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return


def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


def labels_to_one_hot(labels: np.ndarray, class_num: int):
    return np.eye(N=class_num)[labels]


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    Args:
        x:
        eps:

    Returns:
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


# ---------------- New: YAML loader with inheritance and cycle detection ----------------

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts. Child overrides parent on conflicts.
    - If both values are dicts -> recurse
    - Else -> override value replaces base
    """
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_yaml_with_inheritance(path: str, parent_key: str = "PARENT_CONFIG", _visited: Set[str] | None = None) -> Dict[str, Any]:
    """Load YAML with support for multi-level inheritance via PARENT_CONFIG.
    - Child overrides parent (deep merge)
    - Detect cyclic references and raise ValueError
    - Paths are resolved relative to the current YAML file's directory if not absolute
    """
    if _visited is None:
        _visited = set()

    abs_path = os.path.abspath(path)
    if abs_path in _visited:
        raise ValueError(f"Cyclic PARENT_CONFIG reference detected at: {abs_path}")
    _visited.add(abs_path)

    cur_cfg = yaml_to_dict(abs_path) or {}
    if not isinstance(cur_cfg, dict):
        raise ValueError(f"YAML at {abs_path} must load to a dict, got: {type(cur_cfg)}")

    parent_cfg: Dict[str, Any] = {}
    if parent_key in cur_cfg and cur_cfg[parent_key] is not None:
        parent_path = cur_cfg[parent_key]
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(os.path.dirname(abs_path), parent_path)
        parent_cfg = load_yaml_with_inheritance(parent_path, parent_key=parent_key, _visited=_visited)

    # Child overrides parent
    merged = _deep_update(parent_cfg, {k: v for k, v in cur_cfg.items() if k != parent_key})
    return merged


class TrackedConfig(dict):
    """
    一个带访问记录功能的配置类，兼容 dict 的所有用法。
    每次读取（__getitem__ / get）都会记录：
        - 该 key 被读取的次数
        - 读取发生的文件路径和行号
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._access_counts = defaultdict(int)
        self._access_locations = defaultdict(list)

    def _record_access(self, key):
        frame = inspect.currentframe()
        if frame is None:
            return
        caller = frame.f_back
        if caller is None:
            return
        filename = caller.f_code.co_filename
        lineno = caller.f_lineno
        self._access_counts[key] += 1
        self._access_locations[key].append((filename, lineno))

    def __getitem__(self, key):
        self._record_access(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            self._record_access(key)
        return super().get(key, default)

    # 便于后续分析使用情况的几个辅助方法
    def access_summary(self) -> str:
        """
        以人类可读的多行字符串形式返回：
        每个 key 的访问次数及访问位置列表，适合直接写入日志。
        """
        lines: list[str] = []
        lines.append("=== Config Access Summary ===")
        for k in sorted(self.keys(), key=str):
            count = self._access_counts.get(k, 0)
            lines.append(f"- {k}: count={count}")
            locations = self._access_locations.get(k, [])
            for fname, lineno in locations:
                lines.append(f"    @ {fname}:{lineno}")
        return "\n".join(lines)

    def unused_keys(self) -> str:
        """
        以人类可读的多行字符串形式返回：
        所有从未被读取过的配置项 key，适合直接写入日志。
        """
        unused = sorted([k for k in self.keys() if self._access_counts.get(k, 0) == 0], key=str)
        lines: list[str] = []
        lines.append("=== Unused Config Keys ===")
        if not unused:
            lines.append("(none)")
        else:
            for k in unused:
                lines.append(f"- {k}")
        return "\n".join(lines)

