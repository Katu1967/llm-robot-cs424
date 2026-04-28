"""
NAO camera stream with live YOLO object detection overlay
and LLM scene-state extraction.

Running command:
  make run
  -- or --
  $WEBOTS_HOME/Contents/MacOS/webots-controller --robot-name=NAO src/nao_cam.py

Controls:
  q      – quit
  SPACE  – trigger an immediate scene-state extraction + LLM dispatch
  +/=    – increase detection frequency (run YOLO more often)
  -      – decrease detection frequency (run YOLO less often)

Model files must be placed in src/models/:
  yolov3.cfg, yolov3.weights, coco.names
"""

import numpy as np
import cv2
from controller import Robot

from yolo_detection import YOLODetector
from scene_state    import SceneStateExtractor
from llm_bridge     import LLMBridge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIMESTEP     = 32           # ms; must be a multiple of world's basicTimeStep
CAMERA_NAME  = "CameraTop"  # or "CameraBottom"

# YOLO — how often to run inference
DETECT_EVERY_N_FRAMES  = 5   # run YOLO every N frames
CONFIDENCE_THRESHOLD   = 0.5
NMS_THRESHOLD          = 0.4
YOLO_INPUT_SIZE        = (416, 416)

# Scene extraction — automatic periodic trigger
# Set to 0 to disable automatic triggering (manual SPACE only).
SCENE_EXTRACT_EVERY_N_FRAMES = 150   # ~5 s at 31 fps

# LLM Bridge — set verbose=False to suppress the full JSON dump
LLM_VERBOSE = True


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

    # --- YOLO ---
    print("[nao_cam] Loading YOLO model…")
    detector = YOLODetector(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        input_size=YOLO_INPUT_SIZE,
    )
    print(f"[nao_cam] YOLO ready — detecting every {DETECT_EVERY_N_FRAMES} frames.")

    # --- Scene state extractor ---
    extractor = SceneStateExtractor(robot, camera, TIMESTEP, CAMERA_NAME)

    # --- LLM bridge ---
    bridge = LLMBridge(verbose=LLM_VERBOSE)

    # --- Display window ---
    cv2.namedWindow("NAO Camera – YOLO", cv2.WINDOW_AUTOSIZE)

    frame_count     = 0
    last_detections = []
    detect_interval = DETECT_EVERY_N_FRAMES

    while robot.step(TIMESTEP) != -1:
        raw = camera.getImage()
        if raw is None:
            continue

        # Webots returns BGRA bytes
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        bgr = img[:, :, :3]  # drop alpha

        # --- YOLO inference (every N frames) ---
        if frame_count % detect_interval == 0:
            last_detections = detector.detect(bgr)

        # --- Annotate display frame ---
        annotated = detector.annotate(bgr, last_detections)

        hud = (
            f"Objects: {len(last_detections)}  |  "
            f"Detect/{detect_interval}f  |  "
            f"[SPACE] extract  [+/-] freq  [q] quit"
        )
        cv2.putText(
            annotated, hud,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 255, 0), 1, cv2.LINE_AA,
        )

        cv2.imshow("NAO Camera – YOLO", annotated)

        # --- Keyboard ---
        key = cv2.waitKey(1) & 0xFF
        trigger = None

        if key == ord("q"):
            break
        elif key == ord(" "):
            trigger = "manual_keypress"
        elif key == ord("+") or key == ord("="):
            detect_interval = max(1, detect_interval - 1)
            print(f"[nao_cam] Detect interval → {detect_interval}")
        elif key == ord("-"):
            detect_interval += 1
            print(f"[nao_cam] Detect interval → {detect_interval}")

        # --- Automatic periodic trigger ---
        if (
            trigger is None
            and SCENE_EXTRACT_EVERY_N_FRAMES > 0
            and frame_count > 0
            and frame_count % SCENE_EXTRACT_EVERY_N_FRAMES == 0
        ):
            trigger = "periodic"

        # --- Scene extraction + LLM dispatch ---
        if trigger is not None:
            sim_time_ms = int(robot.getTime() * 1000)
            print(f"\n[nao_cam] Scene extraction triggered ({trigger}) @ {sim_time_ms} ms")
            state, snapshot_path = extractor.capture(
                bgr_frame=bgr,
                detections=last_detections,
                sim_time_ms=sim_time_ms,
                frame_count=frame_count,
                trigger=trigger,
            )
            bridge.send(state, snapshot_path)

        frame_count += 1

    cv2.destroyAllWindows()
    print("[nao_cam] Exited cleanly.")


if __name__ == "__main__":
    main()