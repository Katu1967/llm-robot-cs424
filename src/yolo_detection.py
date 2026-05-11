

import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(_SRC_DIR, "models")


class YOLODetector:
    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        **kwargs,
    ):
        weights_path = os.path.join(models_dir, model_name)

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Missing {weights_path} (put yolov8n.pt in src/models/ or let Ultralytics download)."
            )

        self.model = YOLO(weights_path)
        self.classes = self.model.names
        self.conf_thresh = confidence_threshold

        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.classes), 3), dtype="uint8"
        )

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        inference_results = self.model(bgr_frame, conf=self.conf_thresh, verbose=False)

        detections_out: list[dict] = []

        if not inference_results:
            return detections_out

        for one_result in inference_results:
            for box in one_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_width = x2 - x1
                box_height = y2 - y1

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                detections_out.append(
                    {
                        "label": self.classes[class_id],
                        "confidence": confidence,
                        "box": (int(x1), int(y1), int(box_width), int(box_height)),
                        "class_id": class_id,
                    }
                )

        return detections_out

    def annotate(self, bgr_frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        output_bgr = bgr_frame.copy()

        for det in detections:
            box_x, box_y, box_w, box_h = det["box"]
            box_color = [int(c) for c in self.colors[det["class_id"]]]
            cv.rectangle(output_bgr, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, 2)

            label_text = f"{det['label']}: {det['confidence']:.2f}"

            (text_width, text_height), baseline = cv.getTextSize(
                label_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv.rectangle(
                output_bgr,
                (box_x, box_y - text_height - baseline - 4),
                (box_x + text_width, box_y),
                box_color,
                cv.FILLED,
            )

            cv.putText(
                output_bgr,
                label_text,
                (box_x, box_y - baseline - 2),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )

        return output_bgr
