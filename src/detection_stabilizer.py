
from __future__ import annotations

import os
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple


def overlap_between_boxes_xywh(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_right, a_bottom = ax + aw, ay + ah
    b_right, b_bottom = bx + bw, by + bh
    inter_left, inter_top = max(ax, bx), max(ay, by)
    inter_right, inter_bottom = min(a_right, b_right), min(a_bottom, b_bottom)
    inter_w, inter_h = inter_right - inter_left, inter_bottom - inter_top
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def yolo_class_id_for_label(names: Optional[Dict[int, str]], label: str) -> int:
    if not names or not label:
        return -1
    label_lower = str(label).lower().strip()
    for class_id, class_name in names.items():
        if str(class_name).lower().strip() == label_lower:
            return int(class_id)
    return -1


class _PassthroughStabilizer:
    def __init__(self, names: Optional[Dict[int, str]] = None):
        self.names = names

    def update(self, detections: List[dict]) -> List[dict]:
        return list(detections)

    def reset(self) -> None:
        pass


class DetectionStabilizer:
    def __init__(
        self,
        window_size: int = 10,
        min_hits: int = 4,
        iou_threshold: float = 0.25,
        max_miss: int = 5,
        names: Optional[Dict[int, str]] = None,
    ):
        self.window_size = max(1, int(window_size))
        min_hits_clamped = max(1, int(min_hits))
        self.min_hits = min(min_hits_clamped, self.window_size)
        self.iou_threshold = float(iou_threshold)
        self.max_miss = max(0, int(max_miss))
        self.names = names
        self._tracks: List[Dict[str, Any]] = []

    @classmethod
    def from_env(cls, names: Optional[Dict[int, str]] = None) -> Any:
        window = int(os.getenv("YOLO_STAB_WINDOW", "10"))
        if window <= 0:
            return _PassthroughStabilizer(names=names)
        return cls(
            window_size=window,
            min_hits=int(os.getenv("YOLO_STAB_MIN_HITS", "4")),
            iou_threshold=float(os.getenv("YOLO_STAB_IOU", "0.25")),
            max_miss=int(os.getenv("YOLO_STAB_MAX_MISS", "5")),
            names=names,
        )

    def reset(self) -> None:
        self._tracks.clear()

    def _pick_label_from_history(
        self, recent_labels: Deque[Optional[str]], current_frame_label: str
    ) -> str:
        labels_with_data = [x for x in recent_labels if x is not None]
        if not labels_with_data:
            return (current_frame_label or "").strip()
        counts = Counter(labels_with_data)
        top_label, top_count = counts.most_common(1)[0]
        if top_count >= self.min_hits:
            return str(top_label)
        if len(labels_with_data) >= self.min_hits:
            return str(top_label)
        return (current_frame_label or str(top_label)).strip()

    def update(self, detections: List[dict]) -> List[dict]:
        frame_detections = list(detections)
        used_detection_indices: set[int] = set()
        matched_track_indices: set[int] = set()

        for track_index, track in enumerate(self._tracks):
            track_box = track["box"]
            best_detection_index = -1
            best_overlap = 0.0
            for detection_index, detection in enumerate(frame_detections):
                if detection_index in used_detection_indices:
                    continue
                overlap = overlap_between_boxes_xywh(track_box, detection["box"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_detection_index = detection_index
            if best_detection_index >= 0 and best_overlap >= self.iou_threshold:
                used_detection_indices.add(best_detection_index)
                matched_track_indices.add(track_index)
                matched_detection = frame_detections[best_detection_index]
                new_label = str(matched_detection.get("label", "") or "")
                track["labels"].append(new_label)
                track["miss"] = 0
                track["last_detection"] = matched_detection
                track["box"] = tuple(int(v) for v in matched_detection["box"])

        for track_index, track in enumerate(self._tracks):
            if track_index in matched_track_indices:
                continue
            track["labels"].append(None)
            track["miss"] = int(track.get("miss", 0)) + 1

        for detection_index, detection in enumerate(frame_detections):
            if detection_index in used_detection_indices:
                continue
            new_label = str(detection.get("label", "") or "")
            self._tracks.append(
                {
                    "box": tuple(int(v) for v in detection["box"]),
                    "labels": deque([new_label], maxlen=self.window_size),
                    "miss": 0,
                    "last_detection": detection,
                }
            )

        if self.max_miss > 0:
            self._tracks = [
                track_record
                for track_record in self._tracks
                if int(track_record.get("miss", 0)) <= self.max_miss
            ]

        stabilized_output: List[dict] = []
        for track in self._tracks:
            if int(track.get("miss", 0)) != 0:
                continue
            last_detection = track["last_detection"]
            if not isinstance(last_detection, dict):
                continue
            raw_label = str(last_detection.get("label", "") or "")
            stable_label = self._pick_label_from_history(track["labels"], raw_label)
            if not stable_label:
                stable_label = raw_label
            output_detection = dict(last_detection)
            output_detection["label"] = stable_label
            class_id = yolo_class_id_for_label(self.names, stable_label)
            if class_id >= 0:
                output_detection["class_id"] = class_id
            stabilized_output.append(output_detection)
        return stabilized_output
