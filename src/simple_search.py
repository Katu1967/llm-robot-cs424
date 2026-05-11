
from __future__ import annotations

from typing import List, Optional, Tuple


def normalize_aliases(aliases: List[str]) -> List[str]:
    return [a.lower().strip() for a in aliases if a and str(a).strip()]


def match_target_in_scene(
    scene_state: Optional[dict],
    aliases: List[str],
    locked_label: Optional[str] = None,
) -> Optional[Tuple[str, dict]]:
    if not scene_state or not isinstance(scene_state, dict):
        return None

    normalized_aliases = normalize_aliases(aliases)
    if not normalized_aliases:
        return None

    locked_label_lower = (locked_label or "").lower().strip() or None

    for scene_object in scene_state.get("objects", []):
        object_label_lower = (scene_object.get("label") or "").lower()

        if locked_label_lower and object_label_lower != locked_label_lower:
            continue

        for alias in normalized_aliases:
            if alias in object_label_lower or object_label_lower in alias:
                return object_label_lower, scene_object

    return None


def format_object_in_view_line(label: str, obj: dict) -> str:
    height_frac = obj.get("height_frac")
    cx_norm = obj.get("cx_norm")
    cy_norm = obj.get("cy_norm")

    text_parts = [
        f"OBJECT_IN_VIEW: YOLO sees target '{label}'",
        f"position={obj.get('position', '?')}",
        f"distance={obj.get('distance', '?')}",
    ]

    raw_depth = obj.get("distance_m")
    try:
        depth_meters = float(raw_depth) if raw_depth is not None else None
    except (TypeError, ValueError):
        depth_meters = None

    if depth_meters is not None:
        text_parts.append(f"distance_m={depth_meters:.3f} (RangeFinder depth at bbox, meters)")
    else:
        text_parts.append("distance_m=n/a (RangeFinder not available for this box)")

    if height_frac is not None:
        text_parts.append(f"height_frac={height_frac}")

    if cx_norm is not None and cy_norm is not None:
        text_parts.append(f"cx_norm={cx_norm} cy_norm={cy_norm} (vs image center 0.5)")

    return " | ".join(text_parts)


def format_feedback_line(scene_state: Optional[dict], aliases: List[str]) -> str:
    match_result = match_target_in_scene(scene_state, aliases)

    if not match_result:
        return (
            "FEEDBACK: target not in current VISIBLE OBJECTS list "
            "(may be occluded, motion blur, or wrong angle)."
        )

    matched_label, scene_object = match_result
    cx_norm = scene_object.get("cx_norm")
    cy_norm = scene_object.get("cy_norm")
    screen_side = scene_object.get("position", "?")

    centering_hint = "centered"
    if screen_side == "left":
        centering_hint = "left of image center; try turn_degrees left"
    elif screen_side == "right":
        centering_hint = "right of image center; try turn_degrees right"

    height_frac = scene_object.get("height_frac", "?")
    raw_depth = scene_object.get("distance_m")

    try:
        depth_meters = float(raw_depth) if raw_depth is not None else None
    except (TypeError, ValueError):
        depth_meters = None

    depth_clause = (
        f", distance_m={depth_meters:.3f}m (RangeFinder)"
        if depth_meters is not None
        else ", distance_m=n/a"
    )

    return (
        f"FEEDBACK: target '{matched_label}' visible — position={screen_side} ({centering_hint}), "
        f"height_frac={height_frac}, cx_norm={cx_norm}, cy_norm={cy_norm}{depth_clause}"
    )
