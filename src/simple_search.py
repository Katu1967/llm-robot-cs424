"""
Shared YOLO/scene matching for navigation goals (aliases ↔ scene.objects).

Used by the controller during SEARCH_MODE and by SimpleExecutor during approach.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, List


def normalize_aliases(aliases: List[str]) -> List[str]:
    return [a.lower().strip() for a in aliases if a and str(a).strip()]


def match_target_in_scene(
    scene_state: Optional[dict],
    aliases: List[str],
    locked_label: Optional[str] = None,
) -> Optional[Tuple[str, dict]]:
    """
    Return (label, object_dict) if any scene object matches ``aliases``.
    If ``locked_label`` is set, only that YOLO label is considered (approach lock).
    """
    if not scene_state or not isinstance(scene_state, dict):
        return None
    als = normalize_aliases(aliases)
    if not als:
        return None
    lock = (locked_label or "").lower().strip() or None

    for obj in scene_state.get("objects", []):
        lbl = (obj.get("label") or "").lower()
        if lock and lbl != lock:
            continue
        for alias in als:
            if alias in lbl or lbl in alias:
                return lbl, obj
    return None


def format_object_in_view_line(label: str, obj: dict) -> str:
    hf = obj.get("height_frac")
    cx = obj.get("cx_norm")
    cy = obj.get("cy_norm")
    parts = [
        f"OBJECT_IN_VIEW: YOLO sees target '{label}'",
        f"position={obj.get('position', '?')}",
        f"distance={obj.get('distance', '?')}",
    ]
    if hf is not None:
        parts.append(f"height_frac={hf}")
    if cx is not None and cy is not None:
        parts.append(f"cx_norm={cx} cy_norm={cy} (vs image center 0.5)")
    return " | ".join(parts)


def format_feedback_line(scene_state: Optional[dict], aliases: List[str]) -> str:
    """Short line for primitive STEP_DONE: target vs center / visibility."""
    m = match_target_in_scene(scene_state, aliases)
    if not m:
        return (
            "FEEDBACK: target not in current VISIBLE OBJECTS list "
            "(may be occluded, motion blur, or wrong angle)."
        )
    lbl, obj = m
    cx = obj.get("cx_norm")
    cy = obj.get("cy_norm")
    pos = obj.get("position", "?")
    rel = "centered"
    if pos == "left":
        rel = "left of image center — consider turning left or look_left yaw"
    elif pos == "right":
        rel = "right of image center — consider turning right"
    hf = obj.get("height_frac", "?")
    return (
        f"FEEDBACK: target '{lbl}' visible — position={pos} ({rel}), "
        f"height_frac={hf}, cx_norm={cx}, cy_norm={cy}"
    )
