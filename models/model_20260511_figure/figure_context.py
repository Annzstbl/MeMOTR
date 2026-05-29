from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class FigureContext:
    """推理可视化上下文：由 test 脚本构造，模型 forward 末尾直接落盘。"""

    enabled: bool = False
    seq: str = ""
    frame_num: int = 0
    scem_out_dir: str = ""
    track_out_dir: str = ""
    save_maps: Tuple[str, ...] = ("pool_weights",)
    track_id: Optional[int] = None
    track_local_idx: Optional[int] = None
    global_q_idx: Optional[int] = None
    save_track_spectral: bool = False
    save_cross_attn: bool = False
    skip_existing: bool = False
    vis_max_size: int = 1200
    frame_pad_mask: Optional[torch.Tensor] = None
    spectral_timeline: Optional[List[dict]] = field(default=None, repr=False)

    @property
    def frame_tag(self) -> str:
        return f"f{self.frame_num:06d}"

    def should_save_scem(self) -> bool:
        return self.enabled and bool(self.save_maps) and bool(self.scem_out_dir)

    def should_save_track(self) -> bool:
        return (
            self.enabled
            and self.track_id is not None
            and self.track_local_idx is not None
            and self.global_q_idx is not None
            and bool(self.track_out_dir)
            and (self.save_track_spectral or self.save_cross_attn)
        )
