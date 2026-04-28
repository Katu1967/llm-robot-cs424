"""
YOLOv3 object detection module.

Provides a reusable YOLODetector class that can be imported by other scripts
(e.g. nao_cam.py) to run detection on individual frames from any image source.

Model files required (place in src/models/):
  - yolov3.cfg
  - yolov3.weights
  - coco.names

Download links:
  weights : https://pjreddie.com/media/files/yolov3.weights
  cfg     : https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
  names   : https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
"""

import os
import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# Default paths (relative to this file so they work regardless of CWD)
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(_SRC_DIR, "models")


class YOLODetector:
    """
    Wraps YOLOv3 (via OpenCV DNN) for single-frame object detection.

    Parameters
    ----------
    models_dir : str
        Directory containing yolov3.cfg, yolov3.weights, and coco.names.
    confidence_threshold : float
        Minimum detection confidence to keep (0–1).
    nms_threshold : float
        Non-maximum suppression overlap threshold (0–1).
    input_size : tuple[int, int]
        Network input resolution (width, height). Must be a multiple of 32.
    """

    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple = (416, 416),
    ):
        cfg_path     = os.path.join(models_dir, "yolov3.cfg")
        weights_path = os.path.join(models_dir, "yolov3.weights")
        names_path   = os.path.join(models_dir, "coco.names")

        for path in (cfg_path, weights_path, names_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"YOLO model file not found: {path}\n"
                    "See the module docstring for download links."
                )

        # Load class names
        with open(names_path) as f:
            self.classes = f.read().strip().split("\n")

        # Assign a fixed colour per class for consistent visualisation
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.classes), 3), dtype="uint8"
        )

        # Load network
        self.net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Identify output layers
        all_layers = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        # Handle both OpenCV 4.x (array of arrays) and 4.5.4+ (flat array)
        if unconnected.ndim == 2:
            self.output_layers = [all_layers[i[0] - 1] for i in unconnected]
        else:
            self.output_layers = [all_layers[i - 1] for i in unconnected]

        self.conf_thresh = confidence_threshold
        self.nms_thresh  = nms_threshold
        self.input_size  = input_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Run YOLO detection on a single BGR frame.

        Returns
        -------
        list[dict]
            Each dict has keys:
              'label'      : str   – class name
              'confidence' : float – detection confidence
              'box'        : (x, y, w, h) in pixel coords
              'class_id'   : int
        """
        h, w = bgr_frame.shape[:2]

        blob = cv.dnn.blobFromImage(
            bgr_frame, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores     = detection[5:]
                class_id   = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.conf_thresh:
                    continue

                cx, cy, bw, bh = detection[:4] * np.array([w, h, w, h])
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Non-maximum suppression
        indices = cv.dnn.NMSBoxes(
            boxes, confidences, self.conf_thresh, self.nms_thresh
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(
                    {
                        "label":      self.classes[class_ids[i]],
                        "confidence": confidences[i],
                        "box":        tuple(boxes[i]),   # (x, y, w, h)
                        "class_id":   class_ids[i],
                    }
                )
        return detections

    def annotate(self, bgr_frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of *bgr_frame*.

        Returns
        -------
        np.ndarray
            Annotated BGR image (same resolution as input).
        """
        annotated = bgr_frame.copy()
        for det in detections:
            x, y, w, h = det["box"]
            color = [int(c) for c in self.colors[det["class_id"]]]
            cv.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{det['label']}: {det['confidence']:.2f}"
            # Draw a filled background rectangle behind the text for readability
            (text_w, text_h), baseline = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv.rectangle(
                annotated,
                (x, y - text_h - baseline - 4),
                (x + text_w, y),
                color,
                cv.FILLED,
            )
            cv.putText(
                annotated,
                label,
                (x, y - baseline - 2),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
        return annotated
