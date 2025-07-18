# @Author       : Ruopeng Gao
# @Date         : 2022/7/13
# @Description  :
import torch.distributed

from typing import List, Any
from collections import deque, defaultdict
from utils.utils import is_distributed, distributed_world_size


class Value:
    def __init__(self, window_size: int = 100):
        self.value_deque = deque(maxlen=window_size)
        self.total_value = 0.0
        self.total_count = 0

        self.value_sync: None | torch.Tensor = None
        self.total_value_sync = None
        self.total_count_sync = None

    def update(self, value):
        self.value_deque.append(value)
        self.total_value += value
        self.total_count += 1

    def sync(self):
        if is_distributed():
            try:
                # 添加内存清理
                # torch.cuda.empty_cache()
                
                torch.distributed.barrier()
                value_list_gather = [None] * distributed_world_size()
                value_count_gather = [None] * distributed_world_size()
                
                # 减少传输的数据量，只传输最近的数据
                recent_values = list(self.value_deque)[-50:] if len(self.value_deque) > 50 else list(self.value_deque)
                
                torch.distributed.all_gather_object(value_list_gather, recent_values)
                torch.distributed.all_gather_object(value_count_gather, [self.total_value, self.total_count])
                
                # 安全地处理收集到的数据
                value_list = []
                for v_list in value_list_gather:
                    if v_list is not None:
                        value_list.extend(v_list)
                
                self.value_sync = torch.as_tensor(value_list) if value_list else torch.as_tensor([0.0])
                
                # 安全地处理统计数据
                valid_counts = [item for item in value_count_gather if item is not None]
                if valid_counts:
                    self.total_value_sync = sum([item[0] for item in valid_counts])
                    self.total_count_sync = int(sum([item[1] for item in valid_counts]))
                else:
                    self.total_value_sync = self.total_value
                    self.total_count_sync = self.total_count
                
            except Exception as e:
                print(f"Warning: Distributed sync failed: {e}. Using local values.")
                # 如果同步失败，使用本地值
                self.value_sync = torch.as_tensor(list(self.value_deque))
                self.total_value_sync = self.total_value
                self.total_count_sync = self.total_count
        else:
            self.value_sync = torch.as_tensor(list(self.value_deque))
            self.total_value_sync = self.total_value
            self.total_count_sync = self.total_count
        return

    @property
    def avg(self):
        self.check_sync()
        if self.value_sync is not None:
            return self.value_sync.mean().item()
        return 0.0

    @property
    def global_avg(self):
        self.check_sync()
        if self.total_value_sync is not None and self.total_count_sync is not None and self.total_count_sync > 0:
            return self.total_value_sync / self.total_count_sync
        return 0.0

    def check_sync(self):
        if self.value_sync is None:
            print("Warning: .sync() not called successfully, using default values.")
        return


class MetricLog:
    def __init__(self):
        self.metrics = defaultdict(Value)

    def update(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.metrics[name].update(value)
        return

    def sync(self):
        for name, value in self.metrics.items():
            value.sync()
        return

    def get(self, name: str, mode: str):
        return self.metrics[name].__getattribute__(mode)

    def __str__(self):
        s = str()
        if "total_loss" in self.metrics:
            s += f"loss = {self.metrics['total_loss'].avg:.4f} ({self.metrics['total_loss'].global_avg:.4f}); "
        for name, value in self.metrics.items():
            if name == "time per iter":
                continue
            if name == "total_loss":
                continue
            s += f"{name} = {value.avg:.4f} ({value.global_avg:.4f}); "
        return s


def merge_dicts(dicts: List[dict]) -> dict:
    merged = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in merged.keys():
                merged[k] = list()
            merged[k] += v
    return merged
