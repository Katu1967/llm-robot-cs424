
import math
import os
from typing import Callable, List, Optional, Tuple

from simple_search import format_feedback_line, match_target_in_scene


class SimpleExecutor:
    SPIN_DURATION_S = 15.0
    POLL_INTERVAL_S = 0.05
    STUCK_THRESHOLD_S = 2.0

    MOVE_CHECKPOINT_S = float(os.getenv("SIMPLE_MOVE_CHECKPOINT_S", "8.0"))
    APPROACH_REPLAN_S = float(os.getenv("SIMPLE_APPROACH_REPLAN_S", "10.0"))

    FORWARD_MPS = float(os.getenv("SIMPLE_FORWARD_MPS", "0.10"))
    TURN_DPS = float(os.getenv("SIMPLE_TURN_DPS", "50"))
    APPROACH_VX = float(os.getenv("SIMPLE_APPROACH_VX", "0.18"))
    CREEP_VX = float(os.getenv("SIMPLE_CREEP_VX", "0.09"))
    TURN_OMEGA = float(os.getenv("SIMPLE_TURN_OMEGA", "0.52"))
    MAX_PRIMITIVE_S = float(os.getenv("SIMPLE_MAX_PRIMITIVE_S", "30"))
    MIN_STEP_M = 0.05
    MAX_STEP_M = 2.5

    CREEP_ENTER_HF = float(os.getenv("SIMPLE_CREEP_ENTER_HF", "0.42"))
    SUPER_CLOSE_HF = float(os.getenv("SIMPLE_SUPER_CLOSE_HF", "0.62"))
    FINAL_CREEP_TIMEOUT_S = float(os.getenv("SIMPLE_FINAL_CREEP_TIMEOUT_S", "7.0"))
    FORCE_SUPER_CLOSE_HF = float(os.getenv("SIMPLE_FORCE_SUPER_CLOSE_HF", "0.56"))
    STOP_DISTANCE_M = float(os.getenv("SIMPLE_STOP_DISTANCE_M", "0.610"))

    _follow_look_alpha = float(os.getenv("NAO_FOLLOW_LOOK_ALPHA", "0.14"))
    _follow_pitch_gain = float(os.getenv("NAO_FOLLOW_PITCH_GAIN", "0.26"))
    _head_cy_deadband = float(os.getenv("SIMPLE_HEAD_CY_DEADBAND", "0.07"))

    def __init__(self, robot, bus, on_status: Callable[..., None]):
        self._robot = robot
        self._bus = bus
        self._on_status = on_status
        self._mode = "IDLE"
        self._aliases: list = []
        self._elapsed = 0.0
        self._last_poll = 0.0
        self._locate_seen_time = 0.0
        self._verify_timer = 0.0
        self._target_label: Optional[str] = None
        self._max_h_frac = 0.0
        self._stuck_timer = 0.0
        self._checkpoint_timer = 0.0
        self._prim_elapsed = 0.0
        self._step_target_m = 0.0
        self._step_start_xy: Optional[Tuple[float, float]] = None
        self._step_time_budget = 0.0
        self._turn_target_deg = 0.0
        self._turn_time_budget = 0.0
        self._feedback_aliases: Optional[List[str]] = None
        self._final_creep_start: Optional[float] = None
        self._replan_pause = False
        self._last_follow_cx: Optional[float] = None
        self._last_follow_cy: Optional[float] = None
        self._last_follow_pos = "center"
        self._last_follow_valid = False
        self._step_track_head = False

    @property
    def is_idle(self) -> bool:
        return self._mode == "IDLE"

    @property
    def is_approaching(self) -> bool:
        return self._mode == "MOVE"

    @property
    def approach_aliases(self) -> List[str]:
        return list(self._aliases) if self._mode == "MOVE" else []

    def clear_replan_pause(self) -> None:
        self._replan_pause = False

    def resume_approach_walk(self) -> None:
        if self._mode == "MOVE":
            self._robot.start_walk(vx=self.APPROACH_VX)

    def stop_approach_soft(self) -> None:
        if self._mode == "MOVE":
            self._replan_pause = False
            self._robot.stop_locomotion_only()
            self._mode = "IDLE"

    def start_locate(self, aliases: list) -> None:
        self._mode = "LOCATE"
        self._aliases = [a.lower().strip() for a in aliases]
        self._elapsed          = 0.0
        self._locate_seen_time = 0.0
        self._verify_timer     = 0.0
        self._target_label     = None
        self._clear_follow_memory()
        self._robot.reset_head_neutral()
        print(f"[SimpleExecutor] Locating (spin): {self._aliases}")
        self._robot.start_turn(degrees=360)

    def start_move(self, aliases: list) -> None:
        self._mode = "MOVE"
        self._aliases = [a.lower().strip() for a in aliases]
        self._elapsed          = 0.0
        self._checkpoint_timer = 0.0
        self._stuck_timer = 0.0
        self._max_h_frac = 0.0
        self._target_label     = None
        self._feedback_aliases = None
        self._final_creep_start = None
        self._replan_pause = False
        self._clear_follow_memory()
        self._step_track_head = False
        print(
            f"[SimpleExecutor] Moving toward: {self._aliases} "
            f"(LLM safety replan every {self.APPROACH_REPLAN_S:.0f}s, stop depth ≤{self.STOP_DISTANCE_M:.2f}m)"
        )
        self._robot.reset_head_neutral()
        self._robot.start_walk(vx=self.APPROACH_VX)

    def start_step_forward(
        self,
        meters: float,
        feedback_aliases: Optional[List[str]] = None,
        track_head: bool = False,
    ) -> None:
        self._mode = "STEP_FWD"
        self._prim_elapsed = 0.0
        self._step_track_head = bool(track_head)
        if self._step_track_head:
            self._feedback_aliases = (
                [a.lower().strip() for a in feedback_aliases if a and str(a).strip()]
                if feedback_aliases
                else None
            )
        else:
            self._feedback_aliases = None
            self._robot.reset_head_neutral()
        self._step_target_m = max(self.MIN_STEP_M, min(float(meters), self.MAX_STEP_M))
        self._step_start_xy = None
        gps_position = self._robot.get_gps_position()

        if gps_position is not None and len(gps_position) >= 2:
            self._step_start_xy = (float(gps_position[0]), float(gps_position[1]))
        base_t = self._step_target_m / max(self.FORWARD_MPS, 1e-6)
        self._step_time_budget = base_t * (1.35 if self._step_start_xy else 1.55)
        print(
            f"[SimpleExecutor] Step forward: target {self._step_target_m:.2f}m "
            f"(GPS={'yes' if self._step_start_xy else 'no'}, budget {self._step_time_budget:.1f}s)"
        )
        self._robot.start_walk(vx=self.APPROACH_VX)

    def start_step_turn(
        self,
        degrees: float,
        feedback_aliases: Optional[List[str]] = None,
        track_head: bool = False,
    ) -> None:
        self._mode = "STEP_TURN"
        self._prim_elapsed = 0.0
        self._step_track_head = bool(track_head)
        if self._step_track_head:
            self._feedback_aliases = (
                [a.lower().strip() for a in feedback_aliases if a and str(a).strip()]
                if feedback_aliases
                else None
            )
        else:
            self._feedback_aliases = None
            self._robot.reset_head_neutral()
        self._turn_target_deg = max(-180.0, min(180.0, float(degrees)))
        self._turn_time_budget = abs(self._turn_target_deg) / max(self.TURN_DPS, 1e-6)
        print(
            f"[SimpleExecutor] Step turn: {self._turn_target_deg:+.0f}° "
            f"(~{self._turn_time_budget:.1f}s @ {self.TURN_DPS}°/s)"
        )
        self._robot.start_turn(degrees=self._turn_target_deg)

    def stop(self) -> None:
        if self._mode != "IDLE":
            self._robot.stop_walk()
            self._mode = "IDLE"
            self._clear_follow_memory()
            self._step_track_head = False

    @staticmethod
    def _cx_abs_offset(scene_object: dict, screen_side: str) -> float:
        cx_norm = scene_object.get("cx_norm")

        if cx_norm is not None:
            return abs(float(cx_norm) - 0.5)

        if screen_side == "center":
            return 0.0

        if screen_side == "left":
            return 0.20

        return 0.20

    def _horiz_centered(self, scene_object: dict, screen_side: str, tolerance: float) -> bool:
        return self._cx_abs_offset(scene_object, screen_side) <= tolerance

    def _head_follow_bbox(self, scene_object: dict) -> None:
        screen_side = scene_object.get("position", "center")
        cy_norm = scene_object.get("cy_norm")

        if cy_norm is None:
            cy_norm = 0.65 if str(screen_side) in ("left", "right", "center") else 0.5

        smoothing = max(0.04, min(1.0, self._follow_look_alpha))

        self._robot.look_pitch_from_cy_norm(
            float(cy_norm),
            alpha=smoothing,
            pitch_gain=self._follow_pitch_gain,
            cy_deadband=self._head_cy_deadband,
        )

    def _remember_follow_target(self, scene_object: dict) -> None:
        cx_norm = scene_object.get("cx_norm")
        cy_norm = scene_object.get("cy_norm")
        screen_side = scene_object.get("position", "center")

        if cx_norm is not None and cy_norm is not None:
            self._last_follow_cx = float(cx_norm)
            self._last_follow_cy = float(cy_norm)
            self._last_follow_pos = str(screen_side)
            self._last_follow_valid = True
        else:
            self._last_follow_cx = {"left": 0.32, "center": 0.5, "right": 0.68}.get(str(screen_side), 0.5)
            self._last_follow_cy = 0.65
            self._last_follow_pos = str(screen_side)
            self._last_follow_valid = True

    def _head_follow_scene_aliases(self, aliases: Optional[List[str]]) -> None:
        if not aliases:
            return

        scene = self._latest_scene()
        if not scene:
            return

        locked_name = self._target_label if self._target_label else None
        match_pair = match_target_in_scene(scene, aliases, locked_label=locked_name)

        if match_pair:
            matched_label, scene_object = match_pair
            self._head_follow_bbox(scene_object)

    def _clear_follow_memory(self) -> None:
        self._last_follow_valid = False
        self._last_follow_cx = None
        self._last_follow_cy = None
        self._last_follow_pos = "center"

    def tick(self, dt: float) -> None:
        if self._mode == "IDLE":
            return

        self._elapsed += dt
        self._last_poll += dt

        if self._last_poll >= self.POLL_INTERVAL_S:
            self._last_poll = 0.0
            self._update(dt)

    def _update(self, dt: float):
        if self._mode == "STEP_FWD":
            self._update_step_forward()
            return

        if self._mode == "STEP_TURN":
            self._update_step_turn()
            return

        scene_match = self._check_scene()

        if self._mode == "LOCATE":
            if scene_match:
                self._locate_seen_time += self.POLL_INTERVAL_S

                if self._locate_seen_time >= 0.2:
                    matched_label, scene_object = scene_match
                    self._target_label = matched_label
                    self._robot.stop_locomotion_only()
                    self._mode = "VERIFY_LOCATE"
                    self._verify_timer = 0.0
                    print(f"[SimpleExecutor] Potential '{matched_label}' spotted. Stopping to verify...")
            else:
                self._locate_seen_time = 0.0

            if self._elapsed >= self.SPIN_DURATION_S:
                self.stop()
                self._on_status("TIMEOUT: Object not found during spin", None)

        elif self._mode == "VERIFY_LOCATE":
            self._verify_timer += self.POLL_INTERVAL_S

            if self._verify_timer >= 1.0:
                if scene_match:
                    matched_label, scene_object = scene_match
                    self.stop()
                    self._on_status(f"FOUND: {matched_label}", scene_object)
                else:
                    print(
                        f"[SimpleExecutor] False alarm on '{self._target_label}'. "
                        "Resuming spin..."
                    )
                    self._mode = "LOCATE"
                    self._target_label = None
                    self._locate_seen_time = 0.0
                    self._robot.reset_head_neutral()
                    self._robot.start_turn(degrees=360)

        elif self._mode == "MOVE":
            if not self._replan_pause:
                self._checkpoint_timer += self.POLL_INTERVAL_S

            if not scene_match:
                self._stuck_timer += self.POLL_INTERVAL_S

                if self._last_follow_valid and self._last_follow_cx is not None and self._last_follow_cy is not None:
                    self._head_follow_bbox(
                        {
                            "cx_norm": self._last_follow_cx,
                            "cy_norm": self._last_follow_cy,
                            "position": self._last_follow_pos,
                        }
                    )

                if self._stuck_timer >= self.STUCK_THRESHOLD_S:
                    self.stop()
                    self._on_status(
                        f"LOST: Target {self._target_label or self._aliases} disappeared",
                        None,
                    )

                return

            matched_label, scene_object = scene_match
            self._target_label = matched_label
            self._stuck_timer = 0.0
            self._remember_follow_target(scene_object)
            self._head_follow_bbox(scene_object)

            distance_bucket = scene_object.get("distance", "far")
            depth_meters = scene_object.get("distance_m")
            screen_side = scene_object.get("position", "center")
            height_frac_raw = scene_object.get("height_frac")
            height_frac = (
                float(height_frac_raw)
                if height_frac_raw is not None
                else _height_frac_from_bucket(distance_bucket)
            )

            centered_loose = self._horiz_centered(scene_object, screen_side, 0.17)
            centered_tight = self._horiz_centered(scene_object, screen_side, 0.12)

            if height_frac < self.CREEP_ENTER_HF - 0.06:
                self._final_creep_start = None
            elif centered_loose and self._final_creep_start is None:
                self._final_creep_start = self._elapsed

            creep_age_seconds = (
                (self._elapsed - self._final_creep_start)
                if self._final_creep_start is not None
                else 0.0
            )

            is_super_close = False

            if depth_meters is not None and centered_loose:
                try:
                    if float(depth_meters) <= self.STOP_DISTANCE_M:
                        is_super_close = True
                except (TypeError, ValueError):
                    pass

            if not is_super_close and depth_meters is None and self._elapsed >= 0.35:
                if height_frac >= 0.66 and self._horiz_centered(scene_object, screen_side, 0.22):
                    is_super_close = True
                elif height_frac >= self.SUPER_CLOSE_HF and centered_loose:
                    is_super_close = True
                elif height_frac >= 0.56 and distance_bucket == "very_near" and centered_tight:
                    is_super_close = True
                elif (
                    self._final_creep_start is not None
                    and creep_age_seconds >= self.FINAL_CREEP_TIMEOUT_S
                    and height_frac >= self.FORCE_SUPER_CLOSE_HF
                    and centered_loose
                ):
                    is_super_close = True

            if is_super_close:
                self._replan_pause = False
                self._head_follow_bbox(scene_object)

                feet_hint = ""
                if depth_meters is not None:
                    try:
                        depth_float = float(depth_meters)
                        feet_hint = f" (~{depth_float * 3.280839895013123:.2f} ft, {depth_float:.3f} m RangeFinder)"
                    except (TypeError, ValueError):
                        feet_hint = ""

                self.stop()

                distance_message = (
                    f"distance_m={float(depth_meters):.3f} m{feet_hint}"
                    if depth_meters is not None
                    else f"distance={distance_bucket}"
                )

                self._on_status(
                    f"SUPER_CLOSE: {matched_label} (height_frac={height_frac:.2f}, {distance_message}) "
                    "final approach threshold met.",
                    scene_object,
                )

                return

            current_height_frac = (
                float(scene_object.get("height_frac"))
                if scene_object.get("height_frac") is not None
                else _height_frac_from_bucket(scene_object.get("distance", "far"))
            )

            if current_height_frac > self._max_h_frac:
                self._max_h_frac = current_height_frac
                self._stuck_timer = 0.0
            else:
                self._stuck_timer += self.POLL_INTERVAL_S

            stuck_budget_seconds = float(os.getenv("SIMPLE_APPROACH_STUCK_S", "8.0"))

            if self._stuck_timer >= stuck_budget_seconds:
                self._replan_pause = False
                self.stop()

                height_frac_display = scene_object.get("height_frac")
                height_frac_text = f"{height_frac_display}" if height_frac_display is not None else "?"

                self._on_status(
                    f"STUCK: {matched_label} | height_frac={height_frac_text} distance={distance_bucket} position={screen_side} "
                    f"(no bbox growth ~{stuck_budget_seconds:.0f}s)",
                    scene_object,
                )

                return

            if self._checkpoint_timer >= self.APPROACH_REPLAN_S:
                self._checkpoint_timer = 0.0
                self._replan_pause = True
                self._robot.stop_locomotion_only()

                scene_dict = self._latest_scene() or {}
                sonar_dict = scene_dict.get("sonar", {})
                sonar_left_m = sonar_dict.get("left_m", "?")
                sonar_right_m = sonar_dict.get("right_m", "?")

                depth_clause = (
                    f"distance_m={depth_meters}"
                    if depth_meters is not None
                    else "distance_m=n/a"
                )

                self._on_status(
                    f"APPROACH_CHECKPOINT: Feet paused (head still tracking). Approaching '{matched_label}'. "
                    f"{depth_clause}, height_frac={scene_object.get('height_frac')}, bbox position={screen_side}. "
                    f"Sonar left_m={sonar_left_m}, right_m={sonar_right_m}. "
                    f"Goal: approach until distance_m <= {self.STOP_DISTANCE_M:.2f} m. "
                    "Return **move_to_object** with the **same aliases** to continue, "
                    "or **turn_degrees** / **move_forward** (small) to dodge. "
                    "Head pitch follows the target vertically automatically (not LLM) - one action per response.",
                    scene_object,
                )

                return

            if self._replan_pause:
                return

            if screen_side == "left":
                self._robot.start_walk(vx=0.0, omega=self.TURN_OMEGA)
            elif screen_side == "right":
                self._robot.start_walk(vx=0.0, omega=-self.TURN_OMEGA)
            else:
                use_creep_speed = height_frac >= self.CREEP_ENTER_HF and centered_loose
                forward_speed = self.CREEP_VX if use_creep_speed else self.APPROACH_VX
                self._robot.start_walk(vx=forward_speed, omega=0.0)

    def _latest_scene(self) -> Optional[dict]:
        latest = self._bus.get_latest("scene_state") if self._bus else None
        if not latest:
            return None
        state = latest[0] if isinstance(latest, (tuple, list)) else latest
        return state if isinstance(state, dict) else None

    def _update_step_forward(self) -> None:
        self._prim_elapsed += self.POLL_INTERVAL_S

        if self._step_track_head:
            self._head_follow_scene_aliases(self._feedback_aliases)

        step_finished = False

        if self._step_start_xy is not None:
            gps_position = self._robot.get_gps_position()

            if gps_position is not None and len(gps_position) >= 2:
                distance_traveled_m = math.hypot(
                    float(gps_position[0]) - self._step_start_xy[0],
                    float(gps_position[1]) - self._step_start_xy[1],
                )

                if distance_traveled_m >= self._step_target_m * 0.9:
                    step_finished = True

        if not step_finished and self._prim_elapsed >= self._step_time_budget:
            step_finished = True

        if self._prim_elapsed >= self.MAX_PRIMITIVE_S:
            self.stop()
            self._emit_step_done(
                f"STEP_ABORT: move_forward exceeded {self.MAX_PRIMITIVE_S:.0f}s",
            )
            return

        if step_finished:
            self.stop()
            self._emit_step_done(f"STEP_DONE: move_forward ~{self._step_target_m:.2f}m")

    def _update_step_turn(self) -> None:
        self._prim_elapsed += self.POLL_INTERVAL_S
        if self._step_track_head:
            self._head_follow_scene_aliases(self._feedback_aliases)
        if self._prim_elapsed >= self.MAX_PRIMITIVE_S:
            self.stop()
            self._emit_step_done(
                f"STEP_ABORT: turn_degrees exceeded {self.MAX_PRIMITIVE_S:.0f}s",
            )
            return
        if self._prim_elapsed >= self._turn_time_budget:
            self.stop()
            self._emit_step_done(f"STEP_DONE: turn_degrees {self._turn_target_deg:+.0f}°")

    def _emit_step_done(self, base: str) -> None:
        scene = self._latest_scene()

        if self._feedback_aliases:
            feedback_line = format_feedback_line(scene, self._feedback_aliases)
            self._on_status(f"{base}\n{feedback_line}", None)
        else:
            self._on_status(base, None)

        self._feedback_aliases = None

    def _check_scene(self):
        scene = self._latest_scene()

        if not scene:
            return None

        objects = scene.get("objects", [])

        for scene_object in objects:
            object_label_lower = scene_object.get("label", "").lower()

            if self._target_label and object_label_lower != self._target_label:
                continue

            for alias in self._aliases:
                if alias in object_label_lower or object_label_lower in alias:
                    return object_label_lower, scene_object

        return None


def _height_frac_from_bucket(distance_bucket: str) -> float:
    bucket_to_height_hint = {"very_near": 0.45, "near": 0.26, "medium": 0.10, "far": 0.04}

    return bucket_to_height_hint.get(distance_bucket, 0.04)
