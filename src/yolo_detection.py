"""
YOLOv8 object detection helper.

This module provides YOLODetector, a small wrapper around Ultralytics YOLOv8
for running object detection on single image frames.

Expected model location:
    src/models/yolov8n.pt
"""

import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO


src_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(src_dir, "models")


class YOLODetector:
    """
    Runs YOLOv8 detection on individual BGR frames.

    Args:
        models_dir:
            Directory containing the YOLO .pt model file.
        model_name:
            Name of the model file to load.
        confidence_threshold:
            Minimum confidence required to keep a detection.
    """

    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        **kwargs,
    ):
        model_path = os.path.join(models_dir, model_name)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YOLOv8 model file not found: {model_path}\n"
                "Run 'make models' to download it."
            )

        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.confidence_threshold = confidence_threshold

        np.random.seed(42)
        self.class_colors = np.random.randint(
            0,
            255,
            size=(len(self.class_names), 3),
            dtype="uint8",
        )

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Detect objects in one BGR frame.

        Returns a list of dictionaries with:
            label:
                Detected class name.
            confidence:
                Detection confidence score.
            box:
                Bounding box as (x, y, width, height).
            class_id:
                Numeric YOLO class ID.
        """
        yolo_results = self.model(
            bgr_frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []

        if not yolo_results:
            return detections

        for result in yolo_results:
            for detected_box in result.boxes:
                x_min, y_min, x_max, y_max = detected_box.xyxy[0].cpu().numpy()

                box_width = x_max - x_min
                box_height = y_max - y_min

                class_id = int(detected_box.cls[0])
                confidence = float(detected_box.conf[0])

                detections.append(
                    {
                        "label": self.class_names[class_id],
                        "confidence": confidence,
                        "box": (
                            int(x_min),
                            int(y_min),
                            int(box_width),
                            int(box_height),
                        ),
                        "class_id": class_id,
                    }
                )

        return detections

    def annotate(self, bgr_frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw detection boxes and labels on a copy of the input frame.
        """
        annotated_frame = bgr_frame.copy()

        for detection in detections:
            x, y, box_width, box_height = detection["box"]
            class_id = detection["class_id"]
            box_color = [int(channel) for channel in self.class_colors[class_id]]

            cv.rectangle(
                annotated_frame,
                (x, y),
                (x + box_width, y + box_height),
                box_color,
                2,
            )

            label_text = f"{detection['label']}: {detection['confidence']:.2f}"

            text_size, baseline = cv.getTextSize(
                label_text,
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
            text_width, text_height = text_size

            cv.rectangle(
                annotated_frame,
                (x, y - text_height - baseline - 4),
                (x + text_width, y),
                box_color,
                cv.FILLED,
            )

            cv.putText(
                annotated_frame,
                label_text,
                (x, y - baseline - 2),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )

        return annotated_frame