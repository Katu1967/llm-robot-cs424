"""
NAO camera stream with live YOLO object detection overlay.

Running command:
  $WEBOTS_HOME/Contents/MacOS/webots-controller --robot-name=NAO src/nao_cam.py

Controls:
  q  – quit
  +  – increase detection frequency (run YOLO more often)
  -  – decrease detection frequency (run YOLO less often)

Model files must be placed in src/models/:
  yolov3.cfg, yolov3.weights, coco.names
"""

import numpy as np
import cv2
from controller import Robot

from yolo_detection import YOLODetector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIMESTEP = 32           # ms; must be a multiple of your world's basicTimeStep
CAMERA_NAME = "CameraTop"  # or "CameraBottom"

# Run YOLO every N frames.  Frames in between reuse the previous detections.
# Lower  = more accurate but slower (heavier CPU load).
# Higher = faster stream but stale boxes between detections.
DETECT_EVERY_N_FRAMES = 5

CONFIDENCE_THRESHOLD = 0.5   # minimum detection confidence to display
NMS_THRESHOLD        = 0.4   # non-max suppression overlap threshold
YOLO_INPUT_SIZE      = (416, 416)  # must be multiple of 32


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    robot = Robot()

    # --- Camera setup ---
    camera = robot.getDevice(CAMERA_NAME)
    if camera is None:
        raise RuntimeError(
            f"Camera '{CAMERA_NAME}' not found. Check the name in your .wbt file."
        )
    camera.enable(TIMESTEP)

    width  = camera.getWidth()
    height = camera.getHeight()
    print(f"[nao_cam] Connected: {width}x{height} @ {1000 // TIMESTEP} fps")

    # --- YOLO setup ---
    print("[nao_cam] Loading YOLO model…")
    detector = YOLODetector(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        input_size=YOLO_INPUT_SIZE,
    )
    print(f"[nao_cam] YOLO ready — detecting every {DETECT_EVERY_N_FRAMES} frames.")

    # --- Display window ---
    cv2.namedWindow("NAO Camera – YOLO", cv2.WINDOW_AUTOSIZE)

    frame_count     = 0
    last_detections = []          # reused between detection frames
    detect_interval = DETECT_EVERY_N_FRAMES

    while robot.step(TIMESTEP) != -1:
        raw = camera.getImage()
        if raw is None:
            continue

        # Webots returns BGRA bytes
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        bgr = img[:, :, :3]  # drop alpha channel

        # Run detection on every Nth frame
        if frame_count % detect_interval == 0:
            last_detections = detector.detect(bgr)

        # Annotate current frame with the most recent detections
        annotated = detector.annotate(bgr, last_detections)

        # HUD: detection count + current frame interval
        hud = (
            f"Objects: {len(last_detections)}  |  "
            f"Detect every {detect_interval} frames  |  [+/-] to adjust  |  [q] quit"
        )
        cv2.putText(
            annotated, hud,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1, cv2.LINE_AA,
        )

        cv2.imshow("NAO Camera – YOLO", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            detect_interval = max(1, detect_interval - 1)
            print(f"[nao_cam] Detect interval → {detect_interval}")
        elif key == ord("-"):
            detect_interval += 1
            print(f"[nao_cam] Detect interval → {detect_interval}")

        frame_count += 1

    cv2.destroyAllWindows()
    print("[nao_cam] Exited cleanly.")


if __name__ == "__main__":
    main()