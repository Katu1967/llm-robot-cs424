"""
simple_controller.py — NAO simple stack (Webots).

Wires SimplePlanner (LLM), SimpleExecutor (primitives + approach), SceneBus,
SceneStateExtractor + YOLO, and NaoInterface.

SEARCH_MODE: After ``locate_object``, the robot runs a **360° spin** while YOLO
runs each frame. The controller also appends ``OBJECT_IN_VIEW`` when the target
appears in scene state. After spin, ``FOUND:`` / ``TIMEOUT:`` drive the next LLM step;
the LLM can then use ``move_forward`` / ``turn_degrees`` / ``look_up`` / ``look_down`` (use ``look_down`` often for floor-level targets such as laptops).

On **SUPER_CLOSE** (after ``move_to_object`` final creep), the controller **clears the goal automatically** so the run ends without another LLM ``done``.

Run: make simple WEBOTS_HOME=/Applications/Webots.app

SPACE — goal prompt  |  q — quit
"""

import os
import sys
import time
import math
import threading
import logging
from typing import List, Optional

import cv2
import numpy as np
from controller import Robot

from yolo_detection import YOLODetector
from scene_state import SceneStateExtractor
from scene_bus import SceneBus
from nao_interface import NaoInterface
from simple_planner import SimplePlanner
from simple_executor import SimpleExecutor
from simple_search import format_object_in_view_line, match_target_in_scene

TIMESTEP = 32
CAMERA_NAME = "CameraTop"
DETECT_EVERY_N = 3
CONFIDENCE_THRESHOLD = 0.40

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
    g = goal.lower()
    phrases = (
        "go to", "walk to", "move to", "approach", "head to", "get to",
        "navigate to", "travel to", "reach the", "reach a ", "find the",
        "find a ", "locate ", "locate the",
    )
    return any(p in g for p in phrases)


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
    print("[simple_controller] YOLO ready.")

    planner = SimplePlanner()

    last_scene_state = [None]
    last_snapshot_path = [None]
    nav_locate_complete = [False]
    search_mode = [False]
    search_aliases: List[str] = []

    def merge_object_in_view_context(base: str) -> str:
        """Append OBJECT_IN_VIEW when SEARCH_MODE sees the target (YOLO)."""
        if not search_mode[0] or not search_aliases:
            return base
        st = last_scene_state[0]
        m = match_target_in_scene(st, search_aliases)
        if not m:
            return base
        lbl, obj = m
        nav_locate_complete[0] = True
        line = format_object_in_view_line(lbl, obj)
        if "OBJECT_IN_VIEW:" in base:
            return base
        return f"{base}\n{line}"

    def on_status(context: str, obj: Optional[dict]) -> None:
        nonlocal goal_active, goal_prompted
        print(f"\n[simple_controller] Status: {context}")
        if context.startswith("FOUND:"):
            nav_locate_complete[0] = True
        elif context.startswith("TIMEOUT:"):
            nav_locate_complete[0] = False
        if context.startswith("LOST:"):
            nav_locate_complete[0] = False
            search_mode[0] = True
        if context.startswith("SUPER_CLOSE:"):
            print(
                "[simple_controller] Final approach (SUPER_CLOSE) — "
                "vision threshold met; completing goal."
            )
            print("[simple_controller] Goal complete — press SPACE for a new goal.\n")
            goal_active = False
            goal_prompted = False
            search_mode[0] = False
            search_aliases.clear()
            nav_locate_complete[0] = False
            planner.set_goal("")
            return
        ctx = merge_object_in_view_context(context)
        request_capture("status_update")
        time.sleep(0.15)
        planner.request_plan(
            last_scene_state[0] or {},
            last_snapshot_path[0],
            context=ctx,
        )

    executor = SimpleExecutor(nao, bus, on_status)

    def replan_after_llm_head(context: str) -> None:
        request_capture("look_adjust")
        time.sleep(0.15)
        ctx = merge_object_in_view_context(context)
        planner.request_plan(
            last_scene_state[0] or {},
            last_snapshot_path[0],
            context=ctx,
        )

    frame_count = 0
    last_detections = []
    goal_prompted = False
    goal_active = False

    print("\n[simple_controller] All systems ready.")
    print("[simple_controller] Press SPACE in the camera window to set a goal.\n")

    while robot.step(TIMESTEP) != -1:
        raw = camera.getImage()
        if raw is None:
            frame_count += 1
            continue

        bgr = np.frombuffer(raw, dtype=np.uint8).reshape((cam_h, cam_w, 4))
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        if frame_count % DETECT_EVERY_N == 0:
            last_detections = detector.detect(bgr)

        display = bgr.copy()
        for d in last_detections:
            x, y, w, h = d["box"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"{d['label']} {d['confidence']:.2f}",
                (x, max(y - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        status = "SPACE: set goal  |  q: quit"
        if goal_active or search_mode[0]:
            tag = "SEARCH" if search_mode[0] and executor.is_idle else "Running"
            status = f"{tag}…  |  q: quit"
        elif planner.is_planning():
            status = "LLM thinking…"
        cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.imshow("NAO — Simple Controller", display)
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
        elif goal_active or search_mode[0]:
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

        if space_pressed and not goal_prompted and not goal_active and not search_mode[0] and not planner.is_planning():
            goal_prompted = True

            def _ask_goal():
                print("\n" + "=" * 60)
                print("  NAO SIMPLE CONTROLLER — Enter your goal for the robot")
                print("=" * 60)
                goal = input("  Goal > ").strip()
                if not goal:
                    return
                nav_locate_complete[0] = False
                search_mode[0] = False
                search_aliases.clear()
                planner.set_goal(goal)
                print(f"[simple_controller] Goal set: '{goal}'")
                print("[simple_controller] Calling LLM…")
                planner.request_plan(last_scene_state[0] or {}, last_snapshot_path[0])

            threading.Thread(target=_ask_goal, daemon=True).start()

        if planner.has_plan() and executor.is_idle:
            plan = planner.consume_plan()
            if plan is not None:
                action = plan.get("action")
                goal_txt = planner.get_goal() or ""
                nav_goal = goal_requires_locate_first(goal_txt)

                if (
                    action == "move_to_object"
                    and nav_goal
                    and not nav_locate_complete[0]
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
                        search_aliases.clear()
                        search_aliases.extend(
                            [a.lower().strip() for a in aliases if a and str(a).strip()]
                        )
                        search_mode[0] = True
                        nav_locate_complete[0] = False
                        goal_active = True
                        goal_prompted = False
                        executor.start_locate(aliases)
                        print(
                            f"[simple_controller] SEARCH_MODE + spin — aliases={search_aliases}"
                        )
                    else:
                        print("[simple_controller] Warning: locate_object has no aliases.")

                elif action == "move_forward":
                    if nav_goal and not search_mode[0]:
                        print(
                            "[simple_controller] Rejecting move_forward before SEARCH_MODE."
                        )
                        planner.request_plan(
                            last_scene_state[0] or {},
                            last_snapshot_path[0],
                            context=(
                                "ERROR: For navigation goals output locate_object with "
                                "aliases first, then move_forward / turn_degrees / look_up / look_down."
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
                        fb = list(search_aliases) if search_mode[0] else None
                        print(f"[simple_controller] LLM move_forward: {meters:.2f} m")
                        executor.start_step_forward(meters, feedback_aliases=fb)

                elif action == "turn_degrees":
                    if nav_goal and not search_mode[0]:
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
                            deg = float(plan.get("degrees", 45))
                        except (TypeError, ValueError):
                            deg = 45.0
                        goal_active = True
                        goal_prompted = False
                        fb = list(search_aliases) if search_mode[0] else None
                        print(f"[simple_controller] LLM turn_degrees: {deg:+.1f}°")
                        executor.start_step_turn(deg, feedback_aliases=fb)

                elif action == "look_up":
                    if nav_goal and not search_mode[0]:
                        planner.request_plan(
                            last_scene_state[0] or {},
                            last_snapshot_path[0],
                            context="ERROR: Output locate_object with aliases before look_up.",
                        )
                    else:
                        try:
                            deg = float(plan.get("degrees", 12))
                        except (TypeError, ValueError):
                            deg = 12.0
                        deg = max(3.0, min(30.0, deg))
                        nao.adjust_head_pitch(-math.radians(deg))
                        print(f"[simple_controller] LLM look_up: {deg:.1f}°")
                        goal_active = False
                        replan_after_llm_head(f"STEP_DONE: look_up ~{deg:.0f}°")

                elif action == "look_down":
                    if nav_goal and not search_mode[0]:
                        planner.request_plan(
                            last_scene_state[0] or {},
                            last_snapshot_path[0],
                            context="ERROR: Output locate_object with aliases before look_down.",
                        )
                    else:
                        try:
                            deg = float(plan.get("degrees", 18))
                        except (TypeError, ValueError):
                            deg = 18.0
                        deg = max(8.0, min(30.0, deg))
                        nao.adjust_head_pitch(math.radians(deg))
                        print(f"[simple_controller] LLM look_down: {deg:.1f}°")
                        goal_active = False
                        replan_after_llm_head(f"STEP_DONE: look_down ~{deg:.0f}°")

                elif action == "move_to_object":
                    aliases = plan.get("aliases", [])
                    if aliases:
                        search_mode[0] = False
                        goal_active = True
                        goal_prompted = False
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
                    search_mode[0] = False
                    search_aliases.clear()
                    nav_locate_complete[0] = False
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
            and not planner.has_plan()
            and not planner.is_planning()
        ):
            if not search_mode[0]:
                goal_active = False
                goal_prompted = False

        executor.tick(TIMESTEP / 1000.0)
        frame_count += 1

    cv2.destroyAllWindows()
    print("[simple_controller] Exited cleanly.")


if __name__ == "__main__":
    main()
