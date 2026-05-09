"""
YOLOv8 object detection module.

Provides a reusable YOLODetector class that can be imported by other scripts
(e.g. nao_cam.py) to run detection on individual frames from any image source.
Uses the Ultralytics YOLOv8 implementation.

Model files required (place in src/models/):
  - yolov8n.pt (or other YOLOv8 .pt files)
"""

import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Default paths (relative to this file so they work regardless of CWD)
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(_SRC_DIR, "models")


class YOLODetector:
    """
    Wraps YOLOv8 (via Ultralytics) for single-frame object detection.

    Parameters
    ----------
    models_dir : str
        Directory containing the .pt model file.
    model_name : str
        Name of the model file (default: yolov8n.pt).
    confidence_threshold : float
        Minimum detection confidence to keep (0–1).
    """

    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        **kwargs,  # Accept extra args for backward compatibility
    ):
        model_path = os.path.join(models_dir, model_name)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YOLOv8 model file not found: {model_path}\n"
                "Run 'make models' to download it."
            )

        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        self.classes = self.model.names  # dict of {id: name}
        self.conf_thresh = confidence_threshold

        # Assign a fixed colour per class for consistent visualisation
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.classes), 3), dtype="uint8"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 detection on a single BGR frame.

        Returns
        -------
        list[dict]
            Each dict has keys:
              'label'      : str   – class name
              'confidence' : float – detection confidence
              'box'        : (x, y, w, h) in pixel coords
              'class_id'   : int
        """
        # Run inference
        results = self.model(bgr_frame, conf=self.conf_thresh, verbose=False)
        
        detections = []
        if not results:
            return detections

        # Process results (usually there's only one item in results for a single image)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detections.append({
                    "label":      self.classes[class_id],
                    "confidence": confidence,
                    "box":        (int(x1), int(y1), int(w), int(h)),
                    "class_id":   class_id,
                })
                
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
