"""
NAO camera stream with live YOLO object detection overlay,
LLM scene-state extraction, and GPT-4 task planning.

Running command:
  make run
  -- or --
  $WEBOTS_HOME/Contents/MacOS/webots-controller --robot-name=NAO src/nao_cam.py

Controls:
  q      – quit
  SPACE  – immediately capture scene state and publish to bus
  +/=    – increase YOLO detection frequency
  -      – decrease YOLO detection frequency

Model files must be in src/models/:  yolov3.cfg, yolov3.weights, coco.names
API key must be in .env:             GOOGLE_API_KEY=AIza...
"""

import numpy as np
import cv2
from controller import Robot

from yolo_detection import YOLODetector
from scene_state    import SceneStateExtractor
from scene_bus      import SceneBus
from task_planner   import TaskPlanner
from nao_interface import NaoInterface
from plan_executor import PlanExecutor, ExecutorState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIMESTEP    = 32            # ms — must be a multiple of world's basicTimeStep
CAMERA_NAME = "CameraTop"   # or "CameraBottom"

# YOLO inference frequency
DETECT_EVERY_N_FRAMES = 5
CONFIDENCE_THRESHOLD  = 0.5
NMS_THRESHOLD         = 0.4
YOLO_INPUT_SIZE       = (416, 416)

# Scene extraction — automatic periodic trigger.
# Set to 0 to disable (SPACE-only triggering).
SCENE_EXTRACT_EVERY_N_FRAMES = 60   # ≈ 2 s at 31 fps  (set 0 to use SPACE only)


# ---------------------------------------------------------------------------
# Main
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

    # --- SceneBus (topic-based message backbone) ---
    bus = SceneBus()

    # --- TaskPlanner — constructed here (AFTER camera/YOLO) so the heavy
    #     google.generativeai import happens after Webots is already running ---
    print("[nao_cam] Initialising TaskPlanner (loading Gemini…)")
    planner = TaskPlanner()
    planner.attach(bus)
    nao      = NaoInterface(robot, TIMESTEP)
    executor = PlanExecutor(nao, scene_bus=bus)
    print("[nao_cam] All systems ready. Starting main loop.")
    print("[nao_cam] Press SPACE in the camera window to trigger a scene capture.")

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
        bgr = img[:, :, :3]

        # --- YOLO inference ---
        if frame_count % detect_interval == 0:
            last_detections = detector.detect(bgr)

        # --- Annotate display frame ---
        annotated = detector.annotate(bgr, last_detections)

        # HUD overlay
        planning_flag = " [PLANNING…]" if planner.is_planning() else ""
        waiting_flag  = " [WAITING FOR GOAL]" if planner.is_waiting_for_user() else ""
        hud = (
            f"Objects: {len(last_detections)}  |  "
            f"Detect/{detect_interval}f  |  "
            f"[SPACE] capture{planning_flag}{waiting_flag}  [q] quit"
        )
        cv2.putText(
            annotated, hud, (8, 20),
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

        # --- Automatic periodic trigger (skip if planner is busy or waiting for user) ---
        if (
            trigger is None
            and SCENE_EXTRACT_EVERY_N_FRAMES > 0
            and frame_count > 0
            and frame_count % SCENE_EXTRACT_EVERY_N_FRAMES == 0
            and not planner.is_planning()
            and not planner.is_waiting_for_user()
            and executor.is_idle
        ):
            trigger = "periodic"

        # --- Capture scene state and publish to bus ---
        if trigger is not None:
            sim_time_ms = int(robot.getTime() * 1000)
            print(f"\n[nao_cam] Scene capture triggered ({trigger}) @ {sim_time_ms} ms")

            state, snapshot_path = extractor.capture(
                bgr_frame=bgr,
                detections=last_detections,
                sim_time_ms=sim_time_ms,
                frame_count=frame_count,
                trigger=trigger,
            )

            # Publish to bus — TaskPlanner (and any other subscribers) will react
            bus.publish("scene_state", state, snapshot_path)
            
        # ── Execution ─────────────────────────────────
        # ── Execution ─────────────────────────────────
        new_plan = planner.get_plan()
        if new_plan:
            print(f"[DEBUG] Plan available, executor.is_idle={executor.is_idle}, state={executor.state}")
            if executor.is_idle:
                loaded = executor.load_plan(new_plan)
                print(f"[DEBUG] load_plan returned: {loaded}")
                planner.consume_plan()

        executor.tick()

        frame_count += 1

    cv2.destroyAllWindows()
    print("[nao_cam] Exited cleanly.")


if __name__ == "__main__":
    main()