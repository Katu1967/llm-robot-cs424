"""
range_finder_util.py — Read Webots RangeFinder depth buffers and build BGR previews.

Used by ``SceneStateExtractor`` (per-object depth) and ``simple_controller`` (live HUD).
Handles ``getRangeImage(data_type='list')`` and raw float ``bytes`` buffers (google-genai era Webots).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import numpy as np


def default_range_finder_name() -> str:
    return (os.environ.get("NAO_RANGE_FINDER_NAME") or "HeadRangeFinder").strip()


def read_range_depth_hw(rf: Any) -> Optional[np.ndarray]:
    """
    Return depth in metres as ``float32`` array shape ``(H, W)``, or ``None``.

    Webots returns ``inf`` outside ``[minRange, maxRange]``; those values are kept
    so callers can mask them consistently.
    """
    if rf is None:
        return None
    try:
        w = int(rf.getWidth())
        h = int(rf.getHeight())
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    n = w * h
    raw: Any = None
    try:
        raw = rf.getRangeImage(data_type="list")
    except TypeError:
        try:
            raw = rf.getRangeImage()
        except Exception:
            raw = None
    except Exception:
        try:
            raw = rf.getRangeImage()
        except Exception:
            raw = None

    if raw is None:
        try:
            raw = rf.getLayerRangeImage(0)
        except Exception:
            return None
    if raw is None:
        return None

    arr = _coerce_raw_to_hw_float32(raw, w, h, n)
    return arr


def _coerce_raw_to_hw_float32(raw: Any, w: int, h: int, n: int) -> Optional[np.ndarray]:
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(bytes(raw), dtype=np.float32, count=n)
        if buf.size != n:
            return None
        return buf.reshape((h, w)).copy()

    if hasattr(raw, "__len__"):
        ln = len(raw)
        if ln < n:
            return None
        # list or array-like of floats
        flat = np.asarray(raw[:n], dtype=np.float32)
        if flat.size != n:
            return None
        return flat.reshape((h, w))

    return None


def depth_hw_to_bgr_vis(
    depth_hw: np.ndarray,
    min_m: float,
    max_m: float,
    *,
    colormap: int | None = None,
) -> np.ndarray:
    """
    Normalise finite depths in ``[min_m, max_m]`` to a colour-mapped BGR image (HxW x 3).
    """
    cmap = colormap if colormap is not None else getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)

    d = depth_hw.astype(np.float32, copy=False)
    finite = np.isfinite(d)
    out = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    if not np.any(finite):
        return out

    lo = float(min_m)
    hi = float(max_m)
    if hi <= lo:
        hi = lo + 1e-3

    norm = np.zeros_like(d, dtype=np.float32)
    m = finite & (d >= lo) & (d <= hi)
    norm[m] = (d[m] - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)

    return cv2.applyColorMap(u8, cmap)


def median_depth_in_camera_box(
    depth_hw: np.ndarray,
    cam_w: int,
    cam_h: int,
    box_x: int,
    box_y: int,
    box_w: int,
    box_h: int,
    *,
    min_m: float = 0.05,
    max_m: float = 10.0,
    shrink: float = 0.20,
) -> Optional[float]:
    """Median valid depth inside a camera-pixel bbox mapped onto ``depth_hw`` resolution."""
    rf_h, rf_w = depth_hw.shape[:2]
    if cam_w <= 0 or cam_h <= 0 or rf_w <= 0 or rf_h <= 0:
        return None

    sx = box_x + box_w * shrink
    sy = box_y + box_h * shrink
    sw = box_w * (1.0 - 2.0 * shrink)
    sh = box_h * (1.0 - 2.0 * shrink)

    x1 = int(max(0, sx / cam_w * rf_w))
    y1 = int(max(0, sy / cam_h * rf_h))
    x2 = int(min(rf_w, (sx + sw) / cam_w * rf_w))
    y2 = int(min(rf_h, (sy + sh) / cam_h * rf_h))
    if x2 <= x1 or y2 <= y1:
        return None

    patch = depth_hw[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch)]
    valid = valid[(valid > min_m) & (valid < max_m)]
    if valid.size == 0:
        return None
    return round(float(np.median(valid)), 3)
