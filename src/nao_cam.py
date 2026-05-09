"""
NAO camera stream with live YOLO object detection overlay,
LLM scene-state extraction, and task planning + execution.
"""

import numpy as np
import cv2
from controller import Robot

from yolo_detection import YOLODetector
from scene_state    import SceneStateExtractor
from scene_bus      import SceneBus
from task_planner   import TaskPlanner
from nao_interface  import NaoInterface
from plan_executor  import PlanExecutor, ExecutorState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIMESTEP    = 32
CAMERA_NAME = "CameraTop"

# Run YOLO every 2 frames instead of 5 — gives executor fresher detections
# while the robot is moving, reducing lost_count false-positives.
DETECT_EVERY_N_FRAMES = 2

CONFIDENCE_THRESHOLD  = 0.4   # lowered from 0.5 so bottle is detected more reliably
NMS_THRESHOLD         = 0.4
YOLO_INPUT_SIZE       = (416, 416)

# Only trigger a new scene capture / LLM call when executor is idle.
# Value here is just the minimum frame gap — the executor.is_idle check
# in the loop is what really gates it.
SCENE_EXTRACT_EVERY_N_FRAMES = 60


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    robot = Robot()

    camera = robot.getDevice(CAMERA_NAME)
    if camera is None:
        raise RuntimeError(f"Camera '{CAMERA_NAME}' not found.")
    camera.enable(TIMESTEP)

    width  = camera.getWidth()
    height = camera.getHeight()
    print(f"[nao_cam] Connected: {width}x{height} @ {1000 // TIMESTEP} fps")
    rangefinder = robot.getDevice("HeadRangeFinder")
    print("RangeFinder:", rangefinder)
    if rangefinder is not None:
        rangefinder.enable(TIMESTEP)
        print("RangeFinder enabled")
    print("[nao_cam] Loading YOLO model…")
    detector = YOLODetector(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        input_size=YOLO_INPUT_SIZE,
    )
    print(f"[nao_cam] YOLO ready — detecting every {DETECT_EVERY_N_FRAMES} frames.")

    extractor = SceneStateExtractor(robot, camera, TIMESTEP, CAMERA_NAME)
    bus       = SceneBus()

    print("[nao_cam] Initialising TaskPlanner…")
    planner = TaskPlanner()
    planner.attach(bus)

    nao      = NaoInterface(robot, TIMESTEP)
    executor = PlanExecutor(nao, scene_bus=bus)

    print("[nao_cam] All systems ready.")
    print("[nao_cam] Press SPACE to trigger scene capture. Press 'r' to reset executor.")

    cv2.namedWindow("NAO Camera – YOLO", cv2.WINDOW_AUTOSIZE)

    frame_count     = 0
    last_detections = []
    detect_interval = DETECT_EVERY_N_FRAMES

    while robot.step(TIMESTEP) != -1:
        raw = camera.getImage()
        if raw is None:
            continue

        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        bgr = img[:, :, :3]

        # ── YOLO inference ───────────────────────────────────────────
        if frame_count % detect_interval == 0:
            last_detections = detector.detect(bgr)

        # ── Annotate + HUD ───────────────────────────────────────────
        annotated = detector.annotate(bgr, last_detections)

        exec_state = executor.state.name
        step = executor.current_step
        step_str = step.get("action", "?") if step else "—"

        planning_flag = " [PLANNING]"   if planner.is_planning()        else ""
        waiting_flag  = " [WAITING]"    if planner.is_waiting_for_user() else ""

        hud_line1 = (
            f"Objs:{len(last_detections)}  "
            f"Exec:{exec_state}  Step:{step_str}"
            f"{planning_flag}{waiting_flag}"
        )
        hud_line2 = "[SPACE] capture  [r] reset  [q] quit"

        cv2.putText(annotated, hud_line1, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated, hud_line2, (4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("NAO Camera – YOLO", annotated)

        # ── Keyboard ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        trigger = None

        if key == ord("q"):
            break
        elif key == ord(" "):
            trigger = "manual_keypress"
        elif key == ord("r"):
            print("[nao_cam] Executor RESET by operator")
            executor.reset()
        elif key == ord("+") or key == ord("="):
            detect_interval = max(1, detect_interval - 1)
            print(f"[nao_cam] Detect interval → {detect_interval}")
        elif key == ord("-"):
            detect_interval = min(30, detect_interval + 1)
            print(f"[nao_cam] Detect interval → {detect_interval}")

        # ── Periodic scene capture — ONLY when executor is idle ──────
        # This prevents the planner from interrupting mid-execution.
        if (
            trigger is None
            and SCENE_EXTRACT_EVERY_N_FRAMES > 0
            and frame_count > 0
            and frame_count % SCENE_EXTRACT_EVERY_N_FRAMES == 0
            and not planner.is_planning()
            and not planner.is_waiting_for_user()
            and executor.is_idle          # ← key guard: don't re-plan mid-execution
        ):
            trigger = "periodic"

        # ── Capture + publish ────────────────────────────────────────
        if trigger is not None:
            sim_time_ms = int(robot.getTime() * 1000)
            print(f"\n[nao_cam] Scene capture ({trigger}) @ {sim_time_ms} ms")

            state, snapshot_path = extractor.capture(
                bgr_frame=bgr,
                detections=last_detections,
                sim_time_ms=sim_time_ms,
                frame_count=frame_count,
                trigger=trigger,
            )
            bus.publish("scene_state", state, snapshot_path)

        # ── Plan handoff ─────────────────────────────────────────────
        new_plan = planner.get_plan()
        if new_plan:
            print(f"[DEBUG] Plan available — executor.is_idle={executor.is_idle} state={executor.state}")
            if executor.is_idle:
                loaded = executor.load_plan(new_plan)
                print(f"[DEBUG] load_plan returned: {loaded}")
                planner.consume_plan()

        # ── Tick executor ─────────────────────────────────────────────
        executor.tick()

        frame_count += 1

    cv2.destroyAllWindows()
    print("[nao_cam] Exited cleanly.")


if __name__ == "__main__":
    main()