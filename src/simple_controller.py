"""
simple_controller.py — NAO simple stack (Webots).

Wires SimplePlanner (LLM), SimpleExecutor (primitives + approach), SceneBus,
SceneStateExtractor + YOLO, and NaoInterface.

SEARCH_MODE: After ``locate_object``, the robot runs a **360° spin** with **head neutral** while YOLO
runs each frame. The controller also appends ``OBJECT_IN_VIEW`` when the target
appears in scene state. After spin, ``FOUND:`` / ``TIMEOUT:`` drive the next LLM step;
the LLM should prefer ``move_forward`` / ``turn_degrees`` until the target is in context. Head stays **neutral** during search; during approach, pitch tracks the target vertically only (not an LLM action).

On **SUPER_CLOSE** (after ``move_to_object`` when within ~2 ft by RangeFinder depth), the controller **clears the goal automatically** so the run ends without another LLM ``done``.

Run: make simple WEBOTS_HOME=/Applications/Webots.app

Optional: ``SIMPLE_SHOW_DEPTH=0`` to hide the live RangeFinder window (default: shown).

SPACE — goal prompt  |  q — quit
"""

import os
import sys
import time
import threading
import logging
from typing import List, Optional

import cv2
import numpy as np
from controller import Robot

from yolo_detection import YOLODetector
from detection_stabilizer import DetectionStabilizer
from scene_state import SceneStateExtractor
from scene_bus import SceneBus
from nao_interface import NaoInterface
from simple_planner import SimplePlanner
from simple_executor import SimpleExecutor
from simple_search import (
    format_object_in_view_line,
    match_target_in_scene,
    normalize_aliases,
)
from range_finder_util import depth_hw_to_bgr_vis, read_range_depth_hw

TIMESTEP = 32
CAMERA_NAME = "CameraTop"
DETECT_EVERY_N = 3
CONFIDENCE_THRESHOLD = 0.40
DEPTH_WINDOW = "NAO — Simple depth (RangeFinder)"
SHOW_DEPTH = os.getenv("SIMPLE_SHOW_DEPTH", "1").strip().lower() not in ("0", "false", "no")

logging.basicConfig(level=logging.WARNING)

_capture_requested = False
_capture_reason = "external"
_capture_lock = threading.Lock()


def request_capture(reason: str = "external") -> None:
    global _capture_requested, _capture_reason
    with _capture_lock:
        _capture_requested = True
        _capture_reason = reason


def goal_requires_locate_first(goal: str) -> bool:
    if not goal:
        return False
    goal_lower = goal.lower()
    nav_keywords = (
        "go to", "walk to", "move to", "approach", "head to", "get to",
        "navigate to", "travel to", "reach the", "reach a ", "find the",
        "find a ", "locate ", "locate the",
    )
    return any(keyword in goal_lower for keyword in nav_keywords)


def _meters_to_feet(m: float) -> float:
    return float(m) * 3.280839895013123


def main():
    global _capture_requested, _capture_reason

    robot = Robot()
    camera = robot.getDevice(CAMERA_NAME)
    if camera is None:
        print(f"[simple_controller] FATAL: camera '{CAMERA_NAME}' not found.")
        sys.exit(1)
    camera.enable(TIMESTEP)
    cam_w = camera.getWidth()
    cam_h = camera.getHeight()
    print(f"[simple_controller] Camera: {cam_w}x{cam_h}")

    bus = SceneBus()
    nao = NaoInterface(robot, TIMESTEP)
    extractor = SceneStateExtractor(robot, camera, TIMESTEP, CAMERA_NAME)

    print("[simple_controller] Loading YOLO model…")
    detector = YOLODetector(confidence_threshold=CONFIDENCE_THRESHOLD)
    stab = DetectionStabilizer.from_env(names=getattr(detector.model, "names", None))
    print("[simple_controller] YOLO ready.")

    planner = SimplePlanner()

    last_scene_state = [None]
    last_snapshot_path = [None]
    locate_object_found = [False]
    in_search_mode = [False]
    search_target_aliases: List[str] = []
    is_approaching_target = [False]

    def aliases_compatible(plan_aliases: list, target: List[str]) -> bool:
        plan_aliases_set = set(normalize_aliases([str(a) for a in (plan_aliases or [])]))
        target_aliases_set = set(normalize_aliases(list(target)))
        if not plan_aliases_set or not target_aliases_set:
            return False
        return bool(plan_aliases_set & target_aliases_set) or plan_aliases_set <= target_aliases_set or target_aliases_set <= plan_aliases_set

    def merge_object_in_view_context(base: str) -> str:
        """Append OBJECT_IN_VIEW when SEARCH_MODE sees the target (YOLO)."""
        if not in_search_mode[0] or not search_target_aliases:
            return base
        current_state = last_scene_state[0]
        matched_target = match_target_in_scene(current_state, search_target_aliases)
        if not matched_target:
            return base
        label, object_data = matched_target
        locate_object_found[0] = True
        line = format_object_in_view_line(label, object_data)
        if "OBJECT_IN_VIEW:" in base:
            return base
        return f"{base}\n{line}"

    def on_status(context: str, obj: Optional[dict]) -> None:
        nonlocal goal_active, goal_prompted
        print(f"\n[simple_controller] Status: {context}")
        if "APPROACH_CHECKPOINT" in context:
            request_capture("approach_replan")
            ctx = merge_object_in_view_context(context)
            planner.request_plan(
                last_scene_state[0] or {},
                last_snapshot_path[0],
                context=ctx,
            )
            return
        if context.startswith("FOUND:"):
            locate_object_found[0] = True
        elif context.startswith("TIMEOUT:"):
            locate_object_found[0] = False
        if context.startswith("LOST:"):
            locate_object_found[0] = False
            in_search_mode[0] = True
            is_approaching_target[0] = False
        if context.startswith("STUCK:"):
            is_approaching_target[0] = False
        if context.startswith("SUPER_CLOSE:"):
            is_approaching_target[0] = False
            ft_line = ""
            if obj:
                dm = obj.get("distance_m")
                try:
                    if dm is not None:
                        distance_meters = float(dm)
                        ft_line = f" Final RangeFinder distance ≈ {_meters_to_feet(distance_meters):.2f} ft ({distance_meters:.3f} m)."
                except (TypeError, ValueError):
                    pass
            print("[simple_controller] Final approach (SUPER_CLOSE) met. Asking LLM for next step..." + ft_line)
            
            in_search_mode[0] = False
            search_target_aliases.clear()
            locate_object_found[0] = False
            
        ctx = merge_object_in_view_context(context)
        if is_approaching_target[0] and context.startswith("STEP_DONE"):
            ctx = (
                f"APPROACH_INTERRUPT: A dodge step finished while still "
                f"navigating toward the goal object.\n{ctx}"
            )
        request_capture("status_update")
        time.sleep(0.05)
        planner.request_plan(
            last_scene_state[0] or {},
            last_snapshot_path[0],
            context=ctx,
        )

    executor = SimpleExecutor(nao, bus, on_status)

    frame_count = 0
    last_detections = []
    goal_prompted = False
    goal_active = False

    print("\n[simple_controller] All systems ready.")
    print("[simple_controller] Press SPACE in the camera window to set a goal.\n")
    if SHOW_DEPTH:
        cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_AUTOSIZE)

    while robot.step(TIMESTEP) != -1:
        raw = camera.getImage()
        if raw is None:
            frame_count += 1
            continue

        bgr = np.frombuffer(raw, dtype=np.uint8).reshape((cam_h, cam_w, 4))
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        if frame_count % DETECT_EVERY_N == 0:
            last_detections = stab.update(detector.detect(bgr))

        display = bgr.copy()
        for detection in last_detections:
            box_x, box_y, box_w, box_h = detection["box"]
            cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"{detection['label']} {detection['confidence']:.2f}",
                (box_x, max(box_y - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        status = "SPACE: set goal  |  q: quit"
        if goal_active or in_search_mode[0]:
            tag = "SEARCH" if in_search_mode[0] and executor.is_idle else "Running"
            status = f"{tag}…  |  q: quit"
        elif planner.is_planning():
            status = "LLM thinking…"
        cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.imshow("NAO — Simple Controller", display)

        if SHOW_DEPTH:
            rangefinder = extractor.range_finder
            if rangefinder is None:
                blank = np.zeros((max(120, cam_h // 4), max(200, cam_w // 2), 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "No RangeFinder (check NAO_RANGE_FINDER_NAME)",
                    (8, blank.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (80, 80, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(DEPTH_WINDOW, blank)
            else:
                try:
                    depth_array = read_range_depth_hw(rangefinder)
                    if depth_array is not None:
                        min_range = float(rangefinder.getMinRange())
                        max_range = float(rangefinder.getMaxRange())
                        visualization = depth_hw_to_bgr_vis(depth_array, min_range, max_range)
                        scale = max(2, int(round(cam_h / max(1, visualization.shape[0]))))
                        if scale > 1:
                            visualization = cv2.resize(
                                visualization,
                                (visualization.shape[1] * scale, visualization.shape[0] * scale),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        cv2.putText(
                            visualization,
                            f"Range {min_range:.2f}-{max_range:.1f} m (color = depth)",
                            (6, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(DEPTH_WINDOW, visualization)
                    else:
                        blank = np.zeros((max(120, cam_h // 4), max(200, cam_w // 2), 3), dtype=np.uint8)
                        cv2.putText(
                            blank,
                            "Range image not ready (wait 1 sampling period)",
                            (8, blank.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (200, 200, 100),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(DEPTH_WINDOW, blank)
                except Exception as exc:
                    blank = np.zeros((120, 320, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        f"Depth read error: {exc}"[:60],
                        (6, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (50, 50, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(DEPTH_WINDOW, blank)

        key = cv2.waitKey(1) & 0xFF

        space_pressed = key == ord(" ")
        if key == ord("q"):
            break

        ext_trigger = None
        with _capture_lock:
            if _capture_requested:
                ext_trigger = _capture_reason
                _capture_requested = False

        trigger = None
        if space_pressed:
            trigger = "manual_keypress"
        elif ext_trigger:
            trigger = ext_trigger
        elif goal_active or in_search_mode[0]:
            trigger = "search_live"

        if trigger is not None:
            sim_ms = int(robot.getTime() * 1000)
            save_img = trigger != "search_live"
            if trigger != "search_live":
                print(f"[simple_controller] Scene capture ({trigger}) @ {sim_ms} ms")

            state, snap = extractor.capture(
                bgr_frame=bgr,
                detections=last_detections,
                sim_time_ms=sim_ms,
                frame_count=frame_count,
                trigger=trigger,
                save_snapshot=save_img,
            )
            last_scene_state[0] = state
            if snap:
                last_snapshot_path[0] = snap
            bus.publish("scene_state", state, snap)

        if space_pressed and not goal_prompted and not goal_active and not in_search_mode[0] and not planner.is_planning():
            goal_prompted = True

            def _ask_goal():
                print("\n" + "=" * 60)
                print("  NAO SIMPLE CONTROLLER — Enter your goal for the robot")
                print("=" * 60)
                goal = input("  Goal > ").strip()
                if not goal:
                    return
                locate_object_found[0] = False
                in_search_mode[0] = False
                search_target_aliases.clear()
                planner.set_goal(goal)
                stab.reset()
                print(f"[simple_controller] Goal set: '{goal}'")
                print("[simple_controller] Calling LLM…")
                planner.request_plan(last_scene_state[0] or {}, last_snapshot_path[0])

            threading.Thread(target=_ask_goal, daemon=True).start()

        if planner.has_plan() and (executor.is_idle or executor.is_approaching):
            plan = planner.consume_plan()
            if plan is not None:
                action = plan.get("action")
                goal_txt = planner.get_goal() or ""
                nav_goal = goal_requires_locate_first(goal_txt)

                if executor.is_approaching:
                    current_aliases = executor.approach_aliases
                    if action == "move_to_object":
                        plan_aliases = plan.get("aliases", [])
                        if aliases_compatible(plan_aliases, current_aliases):
                            executor.clear_replan_pause()
                            executor.resume_approach_walk()
                            print("[simple_controller] LLM continue approach (move_to_object).")
                        else:
                            planner.request_plan(
                                last_scene_state[0] or {},
                                last_snapshot_path[0],
                                context=(
                                    "ERROR: During APPROACH_CHECKPOINT use the **same aliases** "
                                    f"as the active approach {current_aliases!r}, or turn_degrees / move_forward."
                                ),
                            )
                    elif action == "turn_degrees":
                        try:
                            degrees = float(plan.get("degrees", 25))
                        except (TypeError, ValueError):
                            degrees = 25.0
                        degrees = max(-90.0, min(90.0, degrees))
                        executor.stop_approach_soft()
                        goal_active = True
                        goal_prompted = False
                        print(f"[simple_controller] Approach dodge: turn {degrees:+.1f}°")
                        executor.start_step_turn(degrees, feedback_aliases=list(current_aliases), track_head=True)
                    elif action == "move_forward":
                        try:
                            meters = float(plan.get("meters", 0.35))
                        except (TypeError, ValueError):
                            meters = 0.35
                        meters = max(0.12, min(meters, 0.5))
                        executor.stop_approach_soft()
                        goal_active = True
                        goal_prompted = False
                        print(f"[simple_controller] Approach dodge: move_forward {meters:.2f} m")
                        executor.start_step_forward(meters, feedback_aliases=list(current_aliases), track_head=True)
                    elif action in ("look_up", "look_down"):
                        planner.request_plan(
                            last_scene_state[0] or {},
                            last_snapshot_path[0],
                            context=(
                                "ERROR: look_up and look_down are disabled. During approach, head pitch "
                                "follows the target vertically in the image automatically. Use move_to_object "
                                "(same aliases), turn_degrees, or move_forward."
                            ),
                        )
                    elif action == "done":
                        is_approaching_target[0] = False
                        executor.stop()
                        msg = plan.get("message", "Approach aborted by planner.")
                        print(f"\n[simple_controller] LLM done during approach: {msg}")
                        goal_active = False
                        goal_prompted = False
                        in_search_mode[0] = False
                        search_target_aliases.clear()
                        locate_object_found[0] = False
                        planner.set_goal("")
                    elif action == "fail":
                        is_approaching_target[0] = False
                        executor.stop()
                        reason = plan.get("reason", plan.get("message", "unspecified"))
                        print(f"\n[simple_controller] LLM fail during approach: {reason}")
                        goal_active = False
                        goal_prompted = False
                    else:
                        planner.request_plan(
                            last_scene_state[0] or {},
                            last_snapshot_path[0],
                            context=(
                                f"ERROR: During approach only move_to_object (same aliases), "
                                f"turn_degrees, move_forward (≤0.5m), done, fail — "
                                f"got action={action!r}."
                            ),
                        )

                elif executor.is_idle:
                    if (
                        action == "move_to_object"
                        and nav_goal
                        and not locate_object_found[0]
                    ):
                        print(
                            "[simple_controller] Policy: move_to_object before OBJECT_IN_VIEW "
                            "→ locate_object"
                        )
                        plan = {**plan, "action": "locate_object"}
                        action = "locate_object"

                    if action == "locate_object":
                        aliases = plan.get("aliases", [])
                        if aliases:
                            search_target_aliases.clear()
                            search_target_aliases.extend(
                                [a.lower().strip() for a in aliases if a and str(a).strip()]
                            )
                            in_search_mode[0] = True
                            locate_object_found[0] = False
                            goal_active = True
                            goal_prompted = False
                            executor.start_locate(aliases)
                            print(
                                f"[simple_controller] SEARCH_MODE + spin — aliases={search_target_aliases}"
                            )
                        else:
                            print("[simple_controller] Warning: locate_object has no aliases.")

                    elif action == "move_forward":
                        if nav_goal and not in_search_mode[0]:
                            print(
                                "[simple_controller] Rejecting move_forward before SEARCH_MODE."
                            )
                            planner.request_plan(
                                last_scene_state[0] or {},
                                last_snapshot_path[0],
                                context=(
                                    "ERROR: For navigation goals output locate_object with "
                                    "aliases first, then move_forward / turn_degrees."
                                ),
                            )
                        else:
                            try:
                                meters = float(plan.get("meters", 0.58))
                            except (TypeError, ValueError):
                                meters = 0.58
                            # Enforce a solid stride per call so search / explore does not crawl.
                            meters = max(0.48, min(meters, 2.5))
                            goal_active = True
                            goal_prompted = False
                            print(f"[simple_controller] LLM move_forward: {meters:.2f} m")
                            executor.start_step_forward(meters)
                    elif action == "crouch":
                        goal_active = True
                        goal_prompted = False
                        print("[simple_controller] LLM crouch command received.")
                        executor.start_crouch()

                    elif action == "pick_object":
                        goal_active = True
                        goal_prompted = False
                        print("[simple_controller] LLM pick_object command received.")
                        executor.start_pick()
                    elif action == "turn_degrees":
                        if nav_goal and not in_search_mode[0]:
                            print(
                                "[simple_controller] Rejecting turn_degrees before SEARCH_MODE."
                            )
                            planner.request_plan(
                                last_scene_state[0] or {},
                                last_snapshot_path[0],
                                context=(
                                    "ERROR: For navigation goals output locate_object with "
                                    "aliases first, then motion tools."
                                ),
                            )
                        else:
                            try:
                                degrees = float(plan.get("degrees", 45))
                            except (TypeError, ValueError):
                                degrees = 45.0
                            goal_active = True
                            goal_prompted = False
                            print(f"[simple_controller] LLM turn_degrees: {degrees:+.1f}°")
                            executor.start_step_turn(degrees)

                    elif action in ("look_up", "look_down"):
                        planner.request_plan(
                            last_scene_state[0] or {},
                            last_snapshot_path[0],
                            context=(
                                "ERROR: look_up and look_down are disabled. Head pitch is not an LLM tool — "
                                "it stays neutral during search and tracks the target vertically only during "
                                "move_to_object. Use locate_object, move_forward, turn_degrees, move_to_object, "
                                "done, fail, or clarify."
                            ),
                        )

                    elif action == "move_to_object":
                        aliases = plan.get("aliases", [])
                        if aliases:
                            in_search_mode[0] = False
                            goal_active = True
                            goal_prompted = False
                            is_approaching_target[0] = True
                            print(f"[simple_controller] Starting approach for: {aliases}")
                            executor.start_move(aliases)
                        else:
                            print("[simple_controller] Warning: move_to_object has no aliases.")

                    elif action == "done":
                        msg = plan.get("message", "Task complete.")
                        print(f"\n[simple_controller] ✅ LLM says: {msg}")
                        print("[simple_controller] Goal complete — press SPACE for a new goal.\n")
                        goal_active = False
                        goal_prompted = False
                        in_search_mode[0] = False
                        search_target_aliases.clear()
                        locate_object_found[0] = False
                        is_approaching_target[0] = False
                        planner.set_goal("")

                    elif action == "fail":
                        reason = plan.get("reason", plan.get("message", "unspecified"))
                        print(f"\n[simple_controller] LLM fail: {reason}")
                        ctx = merge_object_in_view_context(
                            f"FAIL: {reason}. If the goal might still be achievable, try a "
                            f"different strategy (locate_object, move_forward, turn, look). "
                            f"Only use fail again if impossible."
                        )
                        request_capture("fail_recovery")
                        time.sleep(0.15)
                        planner.request_plan(
                            last_scene_state[0] or {},
                            last_snapshot_path[0],
                            context=ctx,
                        )

                    elif action == "clarify":
                        q = plan.get("question", "?")
                        print(f"\n[simple_controller] 🤔 LLM needs clarification: {q}")

                        def _ask_clarify():
                            ans = input("  Answer > ").strip()
                            planner.request_plan(
                                last_scene_state[0] or {},
                                last_snapshot_path[0],
                                context=f"User clarification: {ans}",
                            )

                        threading.Thread(target=_ask_clarify, daemon=True).start()

                    else:
                        print(f"[simple_controller] Unknown LLM action: {plan.get('action')}")
        if (
            goal_active
            and executor.is_idle
            and not executor.is_approaching
            and not planner.has_plan()
            and not planner.is_planning()
        ):
            if not in_search_mode[0]:
                goal_active = False
                goal_prompted = False

        executor.tick(TIMESTEP / 1000.0)
        frame_count += 1

    cv2.destroyAllWindows()
    print("[simple_controller] Exited cleanly.")


if __name__ == "__main__":
    main()