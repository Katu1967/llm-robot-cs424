"""
Shared YOLO/scene matching helpers for navigation goals.

These functions connect planner aliases, such as "bottle" or "red cup",
to detected YOLO objects stored in scene_state["objects"].

Used by:
- controller during SEARCH_MODE
- SimpleExecutor during object approach
"""

from __future__ import annotations

from typing import Optional, Tuple, List


def normalize_aliases(aliases: List[str]) -> List[str]:
    """Lowercase aliases and remove empty values."""
    return [
        alias.lower().strip()
        for alias in aliases
        if alias and str(alias).strip()
    ]


def match_target_in_scene(
    scene_state: Optional[dict],
    aliases: List[str],
    locked_label: Optional[str] = None,
) -> Optional[Tuple[str, dict]]:
    """
    Find a scene object whose YOLO label matches one of the target aliases.

    If locked_label is provided, only objects with that exact YOLO label are
    considered. This is used during approach so the robot stays locked onto
    the same type of object it originally found.

    Returns:
        (matched_label, object_dict) if a match is found, otherwise None.
    """
    if not scene_state or not isinstance(scene_state, dict):
        return None

    normalized_aliases = normalize_aliases(aliases)
    if not normalized_aliases:
        return None

    locked_label_normalized = (locked_label or "").lower().strip() or None

    for detected_object in scene_state.get("objects", []):
        detected_label = (detected_object.get("label") or "").lower()

        if locked_label_normalized and detected_label != locked_label_normalized:
            continue

        for alias in normalized_aliases:
            if alias in detected_label or detected_label in alias:
                return detected_label, detected_object

    return None


def format_object_in_view_line(label: str, detected_object: dict) -> str:
    """
    Format a detailed OBJECT_IN_VIEW message for the planner.

    This message tells the LLM that the target object is currently visible,
    along with image position and estimated distance information.
    """
    height_fraction = detected_object.get("height_frac")
    center_x = detected_object.get("cx_norm")
    center_y = detected_object.get("cy_norm")

    message_parts = [
        f"OBJECT_IN_VIEW: YOLO sees target '{label}'",
        f"position={detected_object.get('position', '?')}",
        f"distance={detected_object.get('distance', '?')}",
    ]

    raw_distance_meters = detected_object.get("distance_m")

    try:
        distance_meters = (
            float(raw_distance_meters)
            if raw_distance_meters is not None
            else None
        )
    except (TypeError, ValueError):
        distance_meters = None

    if distance_meters is not None:
        message_parts.append(
            f"distance_m={distance_meters:.3f} "
            "(RangeFinder depth at bbox, meters)"
        )
    else:
        message_parts.append(
            "distance_m=n/a "
            "(RangeFinder not available for this box)"
        )

    if height_fraction is not None:
        message_parts.append(f"height_frac={height_fraction}")

    if center_x is not None and center_y is not None:
        message_parts.append(
            f"cx_norm={center_x} cy_norm={center_y} "
            "(vs image center 0.5)"
        )

    return " | ".join(message_parts)


def format_feedback_line(scene_state: Optional[dict], aliases: List[str]) -> str:
    """
    Format a short target visibility message after a primitive action finishes.

    This is mainly used after STEP_DONE so the planner knows whether the target
    is visible, centered, left, right, or missing from the current camera view.
    """
    match = match_target_in_scene(scene_state, aliases)

    if not match:
        return (
            "FEEDBACK: target not in current VISIBLE OBJECTS list "
            "(may be occluded, motion blur, or wrong angle)."
        )

    matched_label, detected_object = match

    center_x = detected_object.get("cx_norm")
    center_y = detected_object.get("cy_norm")
    position = detected_object.get("position", "?")
    height_fraction = detected_object.get("height_frac", "?")

    position_guidance = "centered"

    if position == "left":
        position_guidance = "left of image center — consider turning left"
    elif position == "right":
        position_guidance = "right of image center — consider turning right"

    raw_distance_meters = detected_object.get("distance_m")

    try:
        distance_meters = (
            float(raw_distance_meters)
            if raw_distance_meters is not None
            else None
        )
    except (TypeError, ValueError):
        distance_meters = None

    distance_message = (
        f", distance_m={distance_meters:.3f}m (RangeFinder)"
        if distance_meters is not None
        else ", distance_m=n/a"
    )

    return (
        f"FEEDBACK: target '{matched_label}' visible — "
        f"position={position} ({position_guidance}), "
        f"height_frac={height_fraction}, "
        f"cx_norm={center_x}, cy_norm={center_y}"
        f"{distance_message}"
    )