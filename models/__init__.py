
import importlib
import torch

from utils.utils import distributed_rank
from .memotr import build as build_memotr


def build_model(config: dict):
    """
    根据 config 选择使用哪个 MeMOTR 实现。

    config["MEMOTR_VERSION"]:
        - "old"（默认）: 使用 models/memotr.py
        - "20260310": 使用 models/model_20260310/memotr_20260310.py
    """
    version = config.get("MEMOTR_VERSION", "old")

    if version == "old":
        model = build_memotr(config=config)
    elif version == "20260310":
        module = importlib.import_module(".model_20260310.memotr_20260310", package=__name__)
        model = module.build(config=config)
    elif version == "20260317":
        module = importlib.import_module(".model_20260317.memotr_20260317", package=__name__)
        model = module.build(config=config)
    else:
        raise ValueError(f"Unknown MEMOTR_VERSION='{version}', expected 'old' or '20260310'.")

    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model
