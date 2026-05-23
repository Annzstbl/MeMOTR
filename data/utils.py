import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from typing import Tuple, Any, Union, Type, Dict, List
from collections import defaultdict


def resolve_stage_scalar(value, sample_stage: int, key: str = "config") -> int:
    """Resolve a scalar or stage-aligned list (same indexing as SAMPLE_LENGTHS / sample_stage)."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{key} list must not be empty.")
        idx = min(len(value) - 1, sample_stage)
        return int(value[idx])
    return int(value)


def collate_fn(batch):
    collated_batch = defaultdict(list)
    for data in batch:
        collated_batch["imgs"].append(data["images"])
        collated_batch["infos"].append(data["targets"])
        collated_batch["img_metas"].append(data['img_metas'])
    return collated_batch
