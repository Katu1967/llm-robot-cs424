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


def read_range_depth_hw(rangefinder: Any) -> Optional[np.ndarray]:
    """
    Return depth in metres as ``float32`` array shape ``(H, W)``, or ``None``.

    Webots returns ``inf`` outside ``[minRange, maxRange]``;
    """
    if rangefinder is None:
        return None
    try:
        width = int(rangefinder.getWidth())
        height = int(rangefinder.getHeight())
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    pixel_count = width * height
    raw_data: Any = None
    try:
        raw_data = rangefinder.getRangeImage(data_type="list")
    except TypeError:
        try:
            raw_data = rangefinder.getRangeImage()
        except Exception:
            raw_data = None
    except Exception:
        try:
            raw_data = rangefinder.getRangeImage()
        except Exception:
            raw_data = None

    if raw_data is None:
        try:
            raw_data = rangefinder.getLayerRangeImage(0)
        except Exception:
            return None
    if raw_data is None:
        return None

    depth_array = _coerce_raw_to_hw_float32(raw_data, width, height, pixel_count)
    return depth_array


def _coerce_raw_to_hw_float32(raw_data: Any, width: int, height: int, pixel_count: int) -> Optional[np.ndarray]:
    if isinstance(raw_data, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(bytes(raw_data), dtype=np.float32, count=pixel_count)
        if buffer.size != pixel_count:
            return None
        return buffer.reshape((height, width)).copy()

    if hasattr(raw_data, "__len__"):
        raw_length = len(raw_data)
        if raw_length < pixel_count:
            return None
        flat_array = np.asarray(raw_data[:pixel_count], dtype=np.float32)
        if flat_array.size != pixel_count:
            return None
        return flat_array.reshape((height, width))

    return None


def depth_hw_to_bgr_vis(
    depth_hw: np.ndarray,
    min_m: float,
    max_m: float,
    *,
    colormap: int | None = None,
) -> np.ndarray:
    """
    Normalise finite depths in ``[min_m, max_m]`` to a colour-mapped BGR image .
    """
    colormap_id = colormap if colormap is not None else getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)

    depth_array = depth_hw.astype(np.float32, copy=False)
    is_finite = np.isfinite(depth_array)
    output_image = np.zeros((depth_array.shape[0], depth_array.shape[1], 3), dtype=np.uint8)
    if not np.any(is_finite):
        return output_image

    min_depth = float(min_m)
    max_depth = float(max_m)
    if max_depth <= min_depth:
        max_depth = min_depth + 1e-3

    normalized = np.zeros_like(depth_array, dtype=np.float32)
    in_range = is_finite & (depth_array >= min_depth) & (depth_array <= max_depth)
    normalized[in_range] = (depth_array[in_range] - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0.0, 1.0)
    depth_uint8 = (normalized * 255.0).astype(np.uint8)

    return cv2.applyColorMap(depth_uint8, colormap_id)


def median_depth_in_camera_box(
    depth_hw: np.ndarray,
    camera_width: int,
    camera_height: int,
    box_x: int,
    box_y: int,
    box_width: int,
    box_height: int,
    *,
    min_m: float = 0.05,
    max_m: float = 10.0,
    shrink: float = 0.20,
) -> Optional[float]:
    """Median valid depth inside a camera-pixel bbox mapped onto ``depth_hw`` resolution."""
    depth_height, depth_width = depth_hw.shape[:2]
    if camera_width <= 0 or camera_height <= 0 or depth_width <= 0 or depth_height <= 0:
        return None

    shrink_x = box_x + box_width * shrink
    shrink_y = box_y + box_height * shrink
    shrink_width = box_width * (1.0 - 2.0 * shrink)
    shrink_height = box_height * (1.0 - 2.0 * shrink)

    box_x_min = int(max(0, shrink_x / camera_width * depth_width))
    box_y_min = int(max(0, shrink_y / camera_height * depth_height))
    box_x_max = int(min(depth_width, (shrink_x + shrink_width) / camera_width * depth_width))
    box_y_max = int(min(depth_height, (shrink_y + shrink_height) / camera_height * depth_height))
    if box_x_max <= box_x_min or box_y_max <= box_y_min:
        return None

    depth_patch = depth_hw[box_y_min:box_y_max, box_x_min:box_x_max]
    valid_depths = depth_patch[np.isfinite(depth_patch)]
    valid_depths = valid_depths[(valid_depths > min_m) & (valid_depths < max_m)]
    if valid_depths.size == 0:
        return None
    return round(float(np.median(valid_depths)), 3)
