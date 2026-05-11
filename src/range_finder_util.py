
from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import numpy as np


def default_range_finder_name() -> str:
    return (os.environ.get("NAO_RANGE_FINDER_NAME") or "HeadRangeFinder").strip()


def read_range_depth_hw(range_finder_device: Any) -> Optional[np.ndarray]:
    if range_finder_device is None:
        return None

    try:
        width = int(range_finder_device.getWidth())
        height = int(range_finder_device.getHeight())
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    pixel_count = width * height
    raw_buffer: Any = None

    try:
        raw_buffer = range_finder_device.getRangeImage(data_type="list")
    except TypeError:
        try:
            raw_buffer = range_finder_device.getRangeImage()
        except Exception:
            raw_buffer = None
    except Exception:
        try:
            raw_buffer = range_finder_device.getRangeImage()
        except Exception:
            raw_buffer = None

    if raw_buffer is None:
        try:
            raw_buffer = range_finder_device.getLayerRangeImage(0)
        except Exception:
            return None
    if raw_buffer is None:
        return None

    return buffer_to_depth_hw_float32(raw_buffer, width, height, pixel_count)


def buffer_to_depth_hw_float32(raw_depth_buffer: Any, width: int, height: int, pixel_count: int) -> Optional[np.ndarray]:
    if isinstance(raw_depth_buffer, (bytes, bytearray, memoryview)):
        flat = np.frombuffer(bytes(raw_depth_buffer), dtype=np.float32, count=pixel_count)

        if flat.size != pixel_count:
            return None

        return flat.reshape((height, width)).copy()

    if hasattr(raw_depth_buffer, "__len__"):
        if len(raw_depth_buffer) < pixel_count:
            return None

        flat = np.asarray(raw_depth_buffer[:pixel_count], dtype=np.float32)

        if flat.size != pixel_count:
            return None

        return flat.reshape((height, width))

    return None


def depth_hw_to_bgr_vis(
    depth_hw: np.ndarray,
    min_m: float,
    max_m: float,
    *,
    colormap: int | None = None,
) -> np.ndarray:
    cmap = colormap if colormap is not None else getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)

    depth = depth_hw.astype(np.float32, copy=False)
    finite_mask = np.isfinite(depth)
    output_bgr = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if not np.any(finite_mask):
        return output_bgr

    lo = float(min_m)
    hi = float(max_m)
    if hi <= lo:
        hi = lo + 1e-3

    normalized = np.zeros_like(depth, dtype=np.float32)
    in_range = finite_mask & (depth >= lo) & (depth <= hi)
    normalized[in_range] = (depth[in_range] - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    gray_u8 = (normalized * 255.0).astype(np.uint8)

    return cv2.applyColorMap(gray_u8, cmap)


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
    depth_height, depth_width = depth_hw.shape[:2]

    if cam_w <= 0 or cam_h <= 0 or depth_width <= 0 or depth_height <= 0:
        return None

    inner_x = box_x + box_w * shrink
    inner_y = box_y + box_h * shrink
    inner_w = box_w * (1.0 - 2.0 * shrink)
    inner_h = box_h * (1.0 - 2.0 * shrink)

    x1 = int(max(0, inner_x / cam_w * depth_width))
    y1 = int(max(0, inner_y / cam_h * depth_height))
    x2 = int(min(depth_width, (inner_x + inner_w) / cam_w * depth_width))
    y2 = int(min(depth_height, (inner_y + inner_h) / cam_h * depth_height))
    if x2 <= x1 or y2 <= y1:
        return None

    patch = depth_hw[y1:y2, x1:x2]
    finite_values = patch[np.isfinite(patch)]
    finite_values = finite_values[(finite_values > min_m) & (finite_values < max_m)]
    if finite_values.size == 0:
        return None
    return round(float(np.median(finite_values)), 3)
