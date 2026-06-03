import torch

from typing import Optional, List


class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, masks: Optional[torch.Tensor]):
        """
        Args:
            tensors: Tensor, (B, C, H, W)
            masks: Tensor, (B, H, W)
        """
        assert tensors.shape[0] == masks.shape[0], \
            f"tensors have batch size {tensors.shape[0]} but get {masks.shape[0]} for mask."
        self.tensors = tensors
        self.masks = masks

    def to(self, device, non_blocking=False):
        """
        Args:
            device:
            non_blocking:
        """
        tensors = self.tensors.to(device, non_blocking=non_blocking)
        if self.masks is None:
            masks = None
        else:
            masks = self.masks.to(device, non_blocking=non_blocking)
        return NestedTensor(tensors=tensors, masks=masks)

    def decompose(self) -> [torch.Tensor, torch.Tensor]:
        return self.tensors, self.masks

    def __repr__(self):
        return str(self.tensors)


def tensor_list_to_nested_tensor(tensor_list: List[torch.Tensor], size_divisibility: int = 32) -> NestedTensor:
    assert tensor_list[0].dim() == 3, f"Tensor should have 3 dimensions, but get {tensor_list[0].dim()}"
    heights, widths = zip(*[t.shape[1:] for t in tensor_list])
    final_shape = [len(tensor_list)] + [tensor_list[0].shape[0]] + list(map(max, (heights, widths)))
    final_b, final_c, final_h, final_w = final_shape
    if size_divisibility > 0:
        stride = size_divisibility
        final_h = (final_h + (stride - 1)) // stride * stride
        final_w = (final_w + (stride - 1)) // stride * stride
    final_shape = [final_b, final_c, final_h, final_w]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensors = torch.zeros(final_shape, dtype=dtype, device=device)
    masks = torch.ones((final_b, final_h, final_w), dtype=torch.bool, device=device)
    for input_tensor, pad_tensor, mask in zip(tensor_list, tensors, masks):
        assert input_tensor.shape[0] == final_shape[1], "Tensor channel size should be equal."
        pad_tensor[: input_tensor.shape[0], : input_tensor.shape[1], : input_tensor.shape[2]].copy_(input_tensor)
        mask[: input_tensor.shape[1], : input_tensor.shape[2]] = False
    return NestedTensor(tensors=tensors, masks=masks)

def _to_hwc_tuple(shape_like) -> tuple[int, int, int]:
    """Convert shape-like metadata to a concrete (H, W, C) tuple."""
    if torch.is_tensor(shape_like):
        shape_like = shape_like.detach().cpu().tolist()
    shape = tuple(int(v) for v in shape_like)
    if len(shape) != 3:
        raise ValueError(f"Expected shape with 3 dims (H, W, C), got: {shape}")
    return shape


def _extract_shapes_from_frame_meta(frame_meta) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Extract effective and padded HWC shapes from one frame meta item.

    effective shape: pre-pad valid region after geometric transforms.
    padded shape: final tensor spatial size after padding.
    """
    meta = frame_meta
    if isinstance(meta, dict) and "transform_metas" in meta:
        meta = meta["transform_metas"]
    if hasattr(meta, "data"):
        meta = meta.data
    if not isinstance(meta, dict):
        raise TypeError(f"Unsupported frame meta type: {type(meta)}")
    if "img_shape" not in meta or "pad_shape" not in meta:
        raise KeyError("Frame meta must include both 'img_shape' and 'pad_shape'.")
    effective_img_shape = _to_hwc_tuple(meta["img_shape"])
    padded_img_shape = _to_hwc_tuple(meta["pad_shape"])
    return effective_img_shape, padded_img_shape


def tensor_list_to_nested_tensor_already_padded(
        tensor_list: List[torch.Tensor], frame_metas: List[dict]) -> NestedTensor:
    """
    Build a NestedTensor from already-padded tensors using per-frame metadata.

    Args:
        tensor_list: list of tensors in (C, H_pad, W_pad), all already padded.
        frame_metas: list of frame metadata aligned with tensor_list.
            Each item must provide:
            - img_shape: effective pre-pad shape (H, W, C)
            - pad_shape: padded shape (H, W, C)

    Returns:
        NestedTensor:
            - tensors: stacked padded tensors, shape (B, C, H_pad, W_pad)
            - masks: False in valid region, True in padded region
    """
    assert tensor_list[0].dim() == 3, f"Tensor should have 3 dimensions, but get {tensor_list[0].dim()}"
    assert len(tensor_list) == len(frame_metas), (
        f"tensor_list length {len(tensor_list)} should equal frame_metas length {len(frame_metas)}."
    )

    meta_shapes = [_extract_shapes_from_frame_meta(m) for m in frame_metas]
    channels = {int(t.shape[0]) for t in tensor_list}
    if len(channels) != 1:
        raise ValueError(f"All tensors must have same channel dim, got {[t.shape[0] for t in tensor_list]}.")

    for b, (tensor, (_, padded_img_shape)) in enumerate(zip(tensor_list, meta_shapes)):
        tensor_hw = (int(tensor.shape[1]), int(tensor.shape[2]))
        meta_hw = (padded_img_shape[0], padded_img_shape[1])
        if tensor_hw != meta_hw:
            raise ValueError(
                f"Tensor/meta padded shape mismatch at batch index {b}: "
                f"tensor={tensor_hw} vs meta={meta_hw}."
            )

    final_b = len(tensor_list)
    final_c = channels.pop()
    final_h = max(padded_img_shape[0] for _, padded_img_shape in meta_shapes)
    final_w = max(padded_img_shape[1] for _, padded_img_shape in meta_shapes)

    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensors = torch.zeros((final_b, final_c, final_h, final_w), dtype=dtype, device=device)
    masks = torch.ones((final_b, final_h, final_w), dtype=torch.bool, device=device)

    for b, (input_tensor, (effective_img_shape, padded_img_shape)) in enumerate(zip(tensor_list, meta_shapes)):
        pad_h, pad_w = padded_img_shape[0], padded_img_shape[1]
        tensors[b, :, :pad_h, :pad_w].copy_(input_tensor)

        eff_h = max(0, min(effective_img_shape[0], pad_h))
        eff_w = max(0, min(effective_img_shape[1], pad_w))
        masks[b, :eff_h, :eff_w] = False

    return NestedTensor(tensors=tensors, masks=masks)


def tensor_list_to_nested_tensor_with_shared_shapes(
        tensor_list: List[torch.Tensor],
        effective_img_shape,
        padded_img_shape) -> NestedTensor:
    """
    Build a NestedTensor when all samples share the same shapes.

    Args:
        tensor_list: list of tensors in (C, H_pad, W_pad), all already padded.
        effective_img_shape: shared valid pre-pad image shape in (H, W, C).
        padded_img_shape: shared padded image shape in (H, W, C).
    """
    effective_img_shape = _to_hwc_tuple(effective_img_shape)
    padded_img_shape = _to_hwc_tuple(padded_img_shape)

    final_b = len(tensor_list)
    final_c = tensor_list[0].shape[0]
    final_h = padded_img_shape[0]
    final_w = padded_img_shape[1]

    for b, t in enumerate(tensor_list):
        if t.shape[0] != final_c:
            raise ValueError(f"Channel mismatch at batch index {b}: {t.shape[0]} vs {final_c}.")
        if t.shape[1] != final_h or t.shape[2] != final_w:
            raise ValueError(
                f"Tensor shape mismatch at batch index {b}: "
                f"{(t.shape[1], t.shape[2])} vs shared padded shape {(final_h, final_w)}."
            )

    masks = torch.ones((final_b, final_h, final_w), dtype=torch.bool, device=tensor_list[0].device)
    masks[:, :effective_img_shape[0], :effective_img_shape[1]] = False
    tensors = torch.stack(tensor_list, dim=0)
    return NestedTensor(tensors=tensors, masks=masks)


def tensor_list_to_nested_tensor_already_padded_shape(
        tensor_list: List[torch.Tensor],
        effective_img_shape,
        padded_img_shape) -> NestedTensor:
    """Backward-compatible alias of tensor_list_to_nested_tensor_with_shared_shapes."""
    return tensor_list_to_nested_tensor_with_shared_shapes(
        tensor_list=tensor_list,
        effective_img_shape=effective_img_shape,
        padded_img_shape=padded_img_shape
    )


def effective_hw_from_nested_tensor(frame: NestedTensor, batch_idx: int = 0) -> tuple[int, int]:
    """
    Valid (pre-pad) region size from NestedTensor mask.

    Model box/ref normalization uses this (H, W), not padded tensor spatial size.
    """
    mask = frame.masks[batch_idx]
    eff_h = int((~mask).any(dim=1).sum().item())
    eff_w = int((~mask).any(dim=0).sum().item())
    if eff_h <= 0 or eff_w <= 0:
        eff_h = int(frame.tensors.shape[-2])
        eff_w = int(frame.tensors.shape[-1])
    return eff_h, eff_w