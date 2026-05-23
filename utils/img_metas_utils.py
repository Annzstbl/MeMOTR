from __future__ import annotations

from typing import Any


def normalize_img_metas_list(img_metas: Any, batch_size: int) -> list:
    """Normalize img_metas to list[dict] with length == batch_size.

    Accepts a single dict (legacy, broadcast to all samples) or list[dict] (recommended
    for bs>1 + multi-scale, where img_shape / pad_shape / version may differ).
    """
    if isinstance(img_metas, dict):
        return [img_metas] * batch_size
    if isinstance(img_metas, (list, tuple)):
        assert len(img_metas) == batch_size, (
            f"len(img_metas)={len(img_metas)} != batch_size={batch_size}"
        )
        return list(img_metas)
    raise TypeError(f"img_metas must be dict or list[dict], got {type(img_metas)}")


def resolve_img_metas_entry(img_metas: Any, batch_idx: int = 0) -> dict:
    """Return the img_metas dict for one batch index."""
    if isinstance(img_metas, (list, tuple)):
        entry = img_metas[batch_idx]
    elif isinstance(img_metas, dict):
        entry = img_metas
    else:
        raise TypeError(f"img_metas must be dict or list[dict], got {type(img_metas)}")

    if hasattr(entry, "data"):
        entry = entry.data
    if not isinstance(entry, dict):
        raise TypeError(f"img_metas entry must be dict, got {type(entry)}")
    return entry


def get_img_shape(img_metas: Any, batch_idx: int = 0) -> tuple:
    """Return (h, w) for one batch sample."""
    entry = resolve_img_metas_entry(img_metas, batch_idx=batch_idx)
    if "img_shape" not in entry:
        raise KeyError("img_metas must provide 'img_shape'.")
    h_img, w_img = entry["img_shape"][:2]
    if hasattr(h_img, "item"):
        h_img = h_img.item()
    if hasattr(w_img, "item"):
        w_img = w_img.item()
    return float(h_img), float(w_img)


def get_img_version(img_metas: Any, batch_idx: int = 0):
    """Return rotation version (str or index tensor) for one batch sample."""
    entry = resolve_img_metas_entry(img_metas, batch_idx=batch_idx)
    if "version" not in entry:
        raise KeyError("img_metas must provide 'version' for rotated-box training.")
    return entry["version"]
