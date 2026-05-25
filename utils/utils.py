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
from collections import defaultdict, deque


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


def plain_config_tree(value: Any) -> Any:
    """将 TrackedConfig 等嵌套结构转为普通 Python 对象，便于 YAML 读写。"""
    if isinstance(value, TrackedConfig):
        return {k: plain_config_tree(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: plain_config_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain_config_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(plain_config_tree(item) for item in value)
    return value


def load_train_config(path: str) -> dict:
    """读取 train/config.yaml，兼容纯 YAML 与 TrackedConfig 序列化格式。"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    try:
        data = yaml.load(raw, Loader=yaml.FullLoader)
    except yaml.constructor.ConstructorError:
        data = yaml.unsafe_load(raw)

    if data is None:
        raise ValueError(f"Empty or invalid YAML: {path}")

    data = plain_config_tree(data)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a dict, got: {type(data)}")
    return data


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
        - 读取发生的文件路径和行号（每条 key 仅保留最近若干条，避免训练长跑后写 config.yaml 体积失控）
    """

    # 每个 key 在 _access_locations 中最多保留的访问位置条数；计数 _access_counts 不受限。
    MAX_ACCESS_LOCATION_HISTORY = max(
        1, int(os.environ.get("TRACKED_CONFIG_MAX_ACCESS_LOCATIONS", "32"))
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._access_counts = defaultdict(int)
        self._access_locations: dict[Any, deque] = {}
        self._this_file = os.path.abspath(__file__)
        # 递归包装嵌套配置，保证子 dict 也可追踪访问
        for key, value in list(super().items()):
            super().__setitem__(key, self._wrap_value(value))

    @classmethod
    def _wrap_value(cls, value):
        if isinstance(value, TrackedConfig):
            return value
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap_value(item) for item in value)
        return value

    def _find_business_caller(self):
        frame = inspect.currentframe()
        while frame is not None:
            code = frame.f_code
            filename = os.path.abspath(code.co_filename)
            func_name = code.co_name
            if not (
                filename == self._this_file and
                func_name in {"_find_business_caller", "_record_access", "__getitem__", "get"}
            ):
                return filename, frame.f_lineno
            frame = frame.f_back
        return None, None

    def _record_access(self, key):
        filename, lineno = self._find_business_caller()
        if filename is None or lineno is None:
            return
        self._access_counts[key] += 1
        if key not in self._access_locations:
            self._access_locations[key] = deque(maxlen=self.MAX_ACCESS_LOCATION_HISTORY)
        self._access_locations[key].append((filename, lineno))

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap_value(value))

    def __getitem__(self, key):
        self._record_access(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            self._record_access(key)
        return super().get(key, default)

    def update(self, *args, **kwargs):
        raw = dict(*args, **kwargs)
        for key, value in raw.items():
            self[key] = value

    # 便于后续分析使用情况的几个辅助方法
    def _collect_access_summary(self, lines: list[str], prefix: str = "") -> None:
        for k in sorted(self.keys(), key=str):
            key_str = str(k)
            full_key = f"{prefix}.{key_str}" if prefix else key_str
            count = self._access_counts.get(k, 0)
            locations = self._access_locations.get(k, [])
            if locations:
                last_fname, last_lineno = locations[-1]
                lines.append(f"- {full_key}: count={count}, last={last_fname}:{last_lineno}")
            else:
                lines.append(f"- {full_key}: count={count}, last=(never)")
            for fname, lineno in locations:
                lines.append(f"    @ {fname}:{lineno}")

            value = dict.__getitem__(self, k)
            if isinstance(value, TrackedConfig):
                value._collect_access_summary(lines=lines, prefix=full_key)

    def access_summary(self) -> str:
        """
        以人类可读的多行字符串形式返回：
        每个 key 的访问次数及访问位置列表，适合直接写入日志。
        """
        lines: list[str] = []
        lines.append("=== Config Access Summary ===")
        self._collect_access_summary(lines=lines)
        return "\n".join(lines)

    def _collect_unused_keys(self, unused: list[str], prefix: str = "") -> None:
        for k in sorted(self.keys(), key=str):
            key_str = str(k)
            full_key = f"{prefix}.{key_str}" if prefix else key_str
            if self._access_counts.get(k, 0) == 0:
                unused.append(full_key)

            value = dict.__getitem__(self, k)
            if isinstance(value, TrackedConfig):
                value._collect_unused_keys(unused=unused, prefix=full_key)

    def unused_keys(self) -> str:
        """
        以人类可读的多行字符串形式返回：
        所有从未被读取过的配置项 key，适合直接写入日志。
        """
        unused: list[str] = []
        self._collect_unused_keys(unused=unused)
        lines: list[str] = []
        lines.append("=== Unused Config Keys ===")
        if not unused:
            lines.append("(none)")
        else:
            for k in unused:
                lines.append(f"- {k}")
        return "\n".join(lines)

