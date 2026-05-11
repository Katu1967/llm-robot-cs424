"""
Temporal smoothing for YOLO outputs.

Matches boxes across frames with IoU and votes on class names in a short window.
That reduces flicker when the classifier jumps on a stable box.
"""

from __future__ import annotations

import os
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple


def _iou_xywh(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    # Intersection over union for two boxes as x y width height in pixels.
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a_x2, b_x2), min(a_y2, b_y2)
    iw, ih = ix2 - ix1, iy2 - iy1

    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return float(inter / ua) if ua > 0 else 0.0


def _class_id_for_names(names: Optional[Dict[int, str]], label: str) -> int:
    # Map a string label back to YOLO class id using the model names dict.
    if not names or not label:
        return -1

    ll = str(label).lower().strip()
    for cid, name in names.items():
        if str(name).lower().strip() == ll:
            return int(cid)

    return -1


class _PassthroughStabilizer:
    """Used when YOLO_STAB_WINDOW is zero or negative. No smoothing state."""

    def __init__(self, names: Optional[Dict[int, str]] = None):
        self.names = names

    def update(self, detections: List[dict]) -> List[dict]:
        return list(detections)

    def reset(self) -> None:
        pass


class DetectionStabilizer:
    """
    Each track is a box matched by IoU across frames.

    The track keeps the last window_size raw labels. None means no match that frame.

    Picking the label shown for the box:
    If the top class appears at least min_hits times in the window use it.
    Else if there are at least min_hits labeled samples use the most common one.
    Else use the current frame label for a cold start.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_hits: int = 4,
        iou_threshold: float = 0.25,
        max_miss: int = 5,
        names: Optional[Dict[int, str]] = None,
    ):
        self.window_size = max(1, int(window_size))

        mh = max(1, int(min_hits))
        self.min_hits = min(mh, self.window_size)

        self.iou_threshold = float(iou_threshold)
        self.max_miss = max(0, int(max_miss))
        self.names = names
        self._tracks: List[Dict[str, Any]] = []

    @classmethod
    def from_env(cls, names: Optional[Dict[int, str]] = None) -> Any:
        w = int(os.getenv("YOLO_STAB_WINDOW", "10"))

        if w <= 0:
            return _PassthroughStabilizer(names=names)

        return cls(
            window_size=w,
            min_hits=int(os.getenv("YOLO_STAB_MIN_HITS", "4")),
            iou_threshold=float(os.getenv("YOLO_STAB_IOU", "0.25")),
            max_miss=int(os.getenv("YOLO_STAB_MAX_MISS", "5")),
            names=names,
        )

    def reset(self) -> None:
        self._tracks.clear()

    def _vote_label(self, labels: Deque[Optional[str]], raw_label: str) -> str:
        # Vote over recent non empty labels in the deque.
        hist = [x for x in labels if x is not None]

        if not hist:
            return (raw_label or "").strip()

        c = Counter(hist)
        best, n_best = c.most_common(1)[0]

        if n_best >= self.min_hits:
            return str(best)

        if len(hist) >= self.min_hits:
            return str(best)

        return (raw_label or str(best)).strip()

    def update(self, detections: List[dict]) -> List[dict]:
        dets = list(detections)
        used: set[int] = set()
        matched_track_idx: set[int] = set()

        # Match each existing track to the best overlapping detection this frame.
        for ti, tr in enumerate(self._tracks):
            tb = tr["box"]
            best_j = -1
            best_iou = 0.0

            for j, det in enumerate(dets):
                if j in used:
                    continue

                iou = _iou_xywh(tb, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_threshold:
                used.add(best_j)
                matched_track_idx.add(ti)
                det = dets[best_j]
                lab = str(det.get("label", "") or "")
                tr["labels"].append(lab)
                tr["miss"] = 0
                tr["raw"] = det
                tr["box"] = tuple(int(x) for x in det["box"])

        # Tracks with no match get None in the label history and a higher miss count.
        for ti, tr in enumerate(self._tracks):
            if ti in matched_track_idx:
                continue

            tr["labels"].append(None)
            tr["miss"] = int(tr.get("miss", 0)) + 1

        # New detections that were not used start a new track.
        for j, det in enumerate(dets):
            if j in used:
                continue

            lab = str(det.get("label", "") or "")
            self._tracks.append(
                {
                    "box": tuple(int(x) for x in det["box"]),
                    "labels": deque([lab], maxlen=self.window_size),
                    "miss": 0,
                    "raw": det,
                }
            )

        # Drop tracks that have been missing too long.
        if self.max_miss > 0:
            self._tracks = [t for t in self._tracks if int(t.get("miss", 0)) <= self.max_miss]

        out: List[dict] = []

        # Only output tracks that matched this frame so the list matches current boxes.
        for tr in self._tracks:
            if int(tr.get("miss", 0)) != 0:
                continue

            raw = tr["raw"]
            if not isinstance(raw, dict):
                continue

            raw_lab = str(raw.get("label", "") or "")
            voted = self._vote_label(tr["labels"], raw_lab)

            if not voted:
                voted = raw_lab

            d = dict(raw)
            d["label"] = voted

            cid = _class_id_for_names(self.names, voted)
            if cid >= 0:
                d["class_id"] = cid

            out.append(d)

        return out
