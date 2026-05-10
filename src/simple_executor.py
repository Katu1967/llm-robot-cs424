import math
import os
from typing import Callable, List, Optional, Tuple

from simple_search import format_feedback_line, match_target_in_scene


class SimpleExecutor:
    """
    locate_object: 360° spin + YOLO verify (FOUND / TIMEOUT); **head stays neutral** during spin and verify (no bbox tracking).
    move_to_object: vision-guided walk; **depth** ``distance_m`` ends the approach at **STOP_DISTANCE_M**
    (~2 ft default) when the RangeFinder reports depth. Without depth, bbox / creep heuristics approximate "close enough".
    Periodic **APPROACH_CHECKPOINT** (~10 s) pauses feet (no Stand) for LLM safety replanning.
    Head **continuously tracks** the target bbox (including brief occlusion, verify pause, and dodge steps).

    LLM primitives: move_forward, turn_degrees.

    Controller also merges OBJECT_IN_VIEW from scene polling during SEARCH_MODE.
    """

    SPIN_DURATION_S       = 15.0    # full rotation search budget
    POLL_INTERVAL_S       = 0.05
    STUCK_THRESHOLD_S     = 2.0

    MOVE_CHECKPOINT_S = float(os.getenv("SIMPLE_MOVE_CHECKPOINT_S", "8.0"))
    # During move_to_object: pause feet + LLM replan on this cadence (seconds).
    APPROACH_REPLAN_S = float(os.getenv("SIMPLE_APPROACH_REPLAN_S", "10.0"))

    FORWARD_MPS = float(os.getenv("SIMPLE_FORWARD_MPS", "0.10"))
    TURN_DPS = float(os.getenv("SIMPLE_TURN_DPS", "50"))
    APPROACH_VX = float(os.getenv("SIMPLE_APPROACH_VX", "0.18"))
    CREEP_VX = float(os.getenv("SIMPLE_CREEP_VX", "0.09"))
    TURN_OMEGA = float(os.getenv("SIMPLE_TURN_OMEGA", "0.52"))
    MAX_PRIMITIVE_S = float(os.getenv("SIMPLE_MAX_PRIMITIVE_S", "30"))
    MIN_STEP_M = 0.05
    MAX_STEP_M = 2.5

    # Final approach during move_to_object: slow creep until bbox is very large + centered.
    CREEP_ENTER_HF = float(os.getenv("SIMPLE_CREEP_ENTER_HF", "0.42"))
    SUPER_CLOSE_HF = float(os.getenv("SIMPLE_SUPER_CLOSE_HF", "0.62"))
    FINAL_CREEP_TIMEOUT_S = float(os.getenv("SIMPLE_FINAL_CREEP_TIMEOUT_S", "7.0"))
    FORCE_SUPER_CLOSE_HF = float(os.getenv("SIMPLE_FORCE_SUPER_CLOSE_HF", "0.56"))
    # ~2 feet (0.6096 m) — RangeFinder depth at bbox when roughly centered ends approach.
    STOP_DISTANCE_M = float(os.getenv("SIMPLE_STOP_DISTANCE_M", "0.610"))

    # Lower alpha / gain = smoother pitch; bbox cy jitters less on the motors.
    _follow_look_alpha = float(os.getenv("NAO_FOLLOW_LOOK_ALPHA", "0.14"))
    _follow_pitch_gain = float(os.getenv("NAO_FOLLOW_PITCH_GAIN", "0.26"))
    _head_cy_deadband = float(os.getenv("SIMPLE_HEAD_CY_DEADBAND", "0.07"))

    def __init__(self, robot, bus, on_status: Callable[..., None]):
        self._robot      = robot
        self._bus        = bus
        self._on_status  = on_status

        self._mode:       str   = "IDLE"   # LOCATE, VERIFY_LOCATE, MOVE, STEP_*, IDLE
        self._aliases:    list  = []
        self._elapsed:    float = 0.0
        self._last_poll:  float = 0.0

        self._locate_seen_time: float = 0.0
        self._verify_timer:    float = 0.0

        self._target_label:     Optional[str] = None
        self._max_h_frac:       float         = 0.0
        self._stuck_timer:      float         = 0.0
        self._checkpoint_timer: float         = 0.0

        self._prim_elapsed:       float = 0.0
        self._step_target_m:      float = 0.0
        self._step_start_xy:     Optional[Tuple[float, float]] = None
        self._step_time_budget:  float = 0.0
        self._turn_target_deg:   float = 0.0
        self._turn_time_budget:  float = 0.0
        self._feedback_aliases: Optional[List[str]] = None
        self._final_creep_start: Optional[float] = None
        self._replan_pause: bool = False
        self._last_follow_cx: Optional[float] = None
        self._last_follow_cy: Optional[float] = None
        self._last_follow_pos: str = "center"
        self._last_follow_valid: bool = False
        self._step_track_head: bool = False

    @property
    def is_idle(self) -> bool:
        return self._mode == "IDLE"

    @property
    def is_approaching(self) -> bool:
        return self._mode == "MOVE"

    @property
    def approach_aliases(self) -> List[str]:
        """Aliases for the current ``move_to_object`` (MOVE mode), else empty."""
        return list(self._aliases) if self._mode == "MOVE" else []

    def clear_replan_pause(self) -> None:
        """Resume walking after an LLM replan approved continue (controller calls this)."""
        self._replan_pause = False

    def resume_approach_walk(self) -> None:
        """Restart forward approach motion while still in MOVE mode."""
        if self._mode == "MOVE":
            self._robot.start_walk(vx=self.APPROACH_VX)

    def stop_approach_soft(self) -> None:
        """Leave MOVE without Stand/rest — for dodge primitives (turn / short move)."""
        if self._mode == "MOVE":
            self._replan_pause = False
            self._robot.stop_locomotion_only()
            self._mode = "IDLE"

    def start_locate(self, aliases: list) -> None:
        """Spin ~360° while polling scene for target; FOUND after verify or TIMEOUT."""
        self._mode             = "LOCATE"
        self._aliases          = [a.lower().strip() for a in aliases]
        self._elapsed          = 0.0
        self._locate_seen_time = 0.0
        self._verify_timer     = 0.0
        self._target_label     = None
        self._clear_follow_memory()
        self._robot.reset_head_neutral()
        print(f"[SimpleExecutor] Locating (spin): {self._aliases}")
        self._robot.start_turn(degrees=360)

    def start_move(self, aliases: list) -> None:
        self._mode             = "MOVE"
        self._aliases          = [a.lower().strip() for a in aliases]
        self._elapsed          = 0.0
        self._checkpoint_timer = 0.0
        self._stuck_timer      = 0.0
        self._max_h_frac       = 0.0
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
        # One neutral head at approach start; pitch then tracks bbox (no per-frame HeadYaw writes).
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
        pos = self._robot.get_gps_position()
        if pos is not None and len(pos) >= 2:
            self._step_start_xy = (float(pos[0]), float(pos[1]))
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
    def _cx_abs_offset(obj: dict, pos: str) -> float:
        c = obj.get("cx_norm")
        if c is not None:
            return abs(float(c) - 0.5)
        if pos == "center":
            return 0.0
        if pos == "left":
            return 0.20
        return 0.20

    def _horiz_centered(self, obj: dict, pos: str, tol: float) -> bool:
        return self._cx_abs_offset(obj, pos) <= tol

    def _head_follow_bbox(self, obj: dict) -> None:
        """Pitch only from bbox vertical position vs image center; yaw fixed at start of approach."""
        pos = obj.get("position", "center")
        cy = obj.get("cy_norm")
        if cy is None:
            cy = 0.65 if str(pos) in ("left", "right", "center") else 0.5
        a = max(0.04, min(1.0, self._follow_look_alpha))
        self._robot.look_pitch_from_cy_norm(
            float(cy),
            alpha=a,
            pitch_gain=self._follow_pitch_gain,
            cy_deadband=self._head_cy_deadband,
        )

    def _remember_follow_target(self, obj: dict) -> None:
        cx, cy = obj.get("cx_norm"), obj.get("cy_norm")
        pos = obj.get("position", "center")
        if cx is not None and cy is not None:
            self._last_follow_cx = float(cx)
            self._last_follow_cy = float(cy)
            self._last_follow_pos = str(pos)
            self._last_follow_valid = True
        else:
            self._last_follow_cx = {"left": 0.32, "center": 0.5, "right": 0.68}.get(str(pos), 0.5)
            self._last_follow_cy = 0.65
            self._last_follow_pos = str(pos)
            self._last_follow_valid = True

    def _head_follow_scene_aliases(self, aliases: Optional[List[str]]) -> None:
        """During dodge steps, keep head on the approach target if still visible."""
        if not aliases:
            return
        scene = self._latest_scene()
        if not scene:
            return
        lock = self._target_label if self._target_label else None
        m = match_target_in_scene(scene, aliases, locked_label=lock)
        if m:
            _lb, obj = m
            self._head_follow_bbox(obj)

    def _clear_follow_memory(self) -> None:
        self._last_follow_valid = False
        self._last_follow_cx = None
        self._last_follow_cy = None
        self._last_follow_pos = "center"

    def tick(self, dt: float) -> None:
        if self._mode == "IDLE":
            return

        self._elapsed   += dt
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

        match = self._check_scene()

        if self._mode == "LOCATE":
            if match:
                self._locate_seen_time += self.POLL_INTERVAL_S
                if self._locate_seen_time >= 0.2:
                    lbl, obj = match
                    self._target_label = lbl
                    self._robot.stop_locomotion_only()
                    self._mode = "VERIFY_LOCATE"
                    self._verify_timer = 0.0
                    print(f"[SimpleExecutor] Potential '{lbl}' spotted. Stopping to verify...")
            else:
                self._locate_seen_time = 0.0

            if self._elapsed >= self.SPIN_DURATION_S:
                self.stop()
                self._on_status("TIMEOUT: Object not found during spin", None)

        elif self._mode == "VERIFY_LOCATE":
            self._verify_timer += self.POLL_INTERVAL_S
            if self._verify_timer >= 1.0:
                if match:
                    lbl, obj = match
                    self.stop()
                    self._on_status(f"FOUND: {lbl}", obj)
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

            if not match:
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

            lbl, obj = match
            self._target_label = lbl
            self._stuck_timer = 0.0
            self._remember_follow_target(obj)
            self._head_follow_bbox(obj)

            dist = obj.get("distance", "far")
            real_dist = obj.get("distance_m")
            pos = obj.get("position", "center")
            hf_obj = obj.get("height_frac")
            hf_val = float(hf_obj) if hf_obj is not None else _height_frac_from_bucket(dist)
            centered_loose = self._horiz_centered(obj, pos, 0.17)
            centered_tight = self._horiz_centered(obj, pos, 0.12)

            if hf_val < self.CREEP_ENTER_HF - 0.06:
                self._final_creep_start = None
            elif centered_loose and self._final_creep_start is None:
                self._final_creep_start = self._elapsed

            creep_age = (
                (self._elapsed - self._final_creep_start)
                if self._final_creep_start is not None
                else 0.0
            )

            super_close = False
            # When RangeFinder depth is available, **only** depth vs STOP_DISTANCE_M (~2 ft)
            # ends the approach. Bbox / creep heuristics alone caused early completion (large
            # bbox but still >2 ft away).
            if real_dist is not None and centered_loose:
                try:
                    if float(real_dist) <= self.STOP_DISTANCE_M:
                        super_close = True
                except (TypeError, ValueError):
                    pass

            if not super_close and real_dist is None and self._elapsed >= 0.35:
                if hf_val >= 0.66 and self._horiz_centered(obj, pos, 0.22):
                    super_close = True
                elif hf_val >= self.SUPER_CLOSE_HF and centered_loose:
                    super_close = True
                elif hf_val >= 0.56 and dist == "very_near" and centered_tight:
                    super_close = True
                elif (
                    self._final_creep_start is not None
                    and creep_age >= self.FINAL_CREEP_TIMEOUT_S
                    and hf_val >= self.FORCE_SUPER_CLOSE_HF
                    and centered_loose
                ):
                    super_close = True

            if super_close:
                self._replan_pause = False
                self._head_follow_bbox(obj)
                ft_s = ""
                if real_dist is not None:
                    try:
                        m = float(real_dist)
                        ft_s = f" (~{m * 3.280839895013123:.2f} ft, {m:.3f} m RangeFinder)"
                    except (TypeError, ValueError):
                        ft_s = ""
                self.stop()
                dist_msg = (
                    f"distance_m={float(real_dist):.3f} m{ft_s}"
                    if real_dist is not None
                    else f"distance={dist}"
                )
                self._on_status(
                    f"SUPER_CLOSE: {lbl} (height_frac={hf_val:.2f}, {dist_msg}) "
                    "— final approach threshold met.",
                    obj,
                )
                return

            curr_h = float(obj.get("height_frac")) if obj.get("height_frac") is not None else _height_frac_from_bucket(
                obj.get("distance", "far")
            )

            if curr_h > self._max_h_frac:
                self._max_h_frac = curr_h
                self._stuck_timer = 0.0
            else:
                self._stuck_timer += self.POLL_INTERVAL_S

            stuck_budget = float(os.getenv("SIMPLE_APPROACH_STUCK_S", "8.0"))
            if self._stuck_timer >= stuck_budget:
                self._replan_pause = False
                self.stop()
                hf = obj.get("height_frac")
                hf_s = f"{hf}" if hf is not None else "?"
                self._on_status(
                    f"STUCK: {lbl} | height_frac={hf_s} distance={dist} position={pos} "
                    f"(no bbox growth ~{stuck_budget:.0f}s)",
                    obj,
                )
                return

            if self._checkpoint_timer >= self.APPROACH_REPLAN_S:
                self._checkpoint_timer = 0.0
                self._replan_pause = True
                self._robot.stop_locomotion_only()
                scene = self._latest_scene() or {}
                son = scene.get("sonar", {})
                left = son.get("left_m", "?")
                right = son.get("right_m", "?")
                dm_p = (
                    f"distance_m={real_dist}"
                    if real_dist is not None
                    else "distance_m=n/a"
                )
                self._on_status(
                    f"APPROACH_CHECKPOINT: Feet paused (head still tracking). Approaching '{lbl}'. "
                    f"{dm_p}, height_frac={obj.get('height_frac')}, bbox position={pos}. "
                    f"Sonar left_m={left}, right_m={right}. "
                    "Goal: stay within ~2 ft (~0.61 m) depth when centered. "
                    "Return **move_to_object** with the **same aliases** to continue, "
                    "or **turn_degrees** / **move_forward** (small) to dodge. "
                    "Head pitch follows the target vertically automatically (not LLM) — one action per response.",
                    obj,
                )
                return

            if self._replan_pause:
                return

            if pos == "left":
                self._robot.start_walk(vx=0.0, omega=self.TURN_OMEGA)
            elif pos == "right":
                self._robot.start_walk(vx=0.0, omega=-self.TURN_OMEGA)
            else:
                use_creep = hf_val >= self.CREEP_ENTER_HF and centered_loose
                vx = self.CREEP_VX if use_creep else self.APPROACH_VX
                self._robot.start_walk(vx=vx, omega=0.0)

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
        done = False
        if self._step_start_xy is not None:
            pos = self._robot.get_gps_position()
            if pos is not None and len(pos) >= 2:
                d = math.hypot(
                    float(pos[0]) - self._step_start_xy[0],
                    float(pos[1]) - self._step_start_xy[1],
                )
                if d >= self._step_target_m * 0.9:
                    done = True
        if not done and self._prim_elapsed >= self._step_time_budget:
            done = True
        if self._prim_elapsed >= self.MAX_PRIMITIVE_S:
            self.stop()
            self._emit_step_done(
                f"STEP_ABORT: move_forward exceeded {self.MAX_PRIMITIVE_S:.0f}s",
            )
            return
        if done:
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
            fb = format_feedback_line(scene, self._feedback_aliases)
            self._on_status(f"{base}\n{fb}", None)
        else:
            self._on_status(base, None)
        self._feedback_aliases = None

    def _check_scene(self):
        scene = self._latest_scene()
        if not scene:
            return None
        objects = scene.get("objects", [])
        for obj in objects:
            lbl = obj.get("label", "").lower()
            if self._target_label and lbl != self._target_label:
                continue
            for alias in self._aliases:
                if alias in lbl or lbl in alias:
                    return lbl, obj
        return None


def _height_frac_from_bucket(bucket: str) -> float:
    h_map = {"very_near": 0.45, "near": 0.26, "medium": 0.10, "far": 0.04}
    return h_map.get(bucket, 0.04)
