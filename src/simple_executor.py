import math
import os
from typing import Callable, List, Optional, Tuple

from simple_search import format_feedback_line


class SimpleExecutor:
    """
    locate_object: 360° spin + YOLO verify.
    move_to_object: approach target using YOLO + RangeFinder distance.
    """

    SPIN_DURATION_S       = 15.0
    POLL_INTERVAL_S       = 0.05
    STUCK_THRESHOLD_S     = 4.0

    MOVE_CHECKPOINT_S = float(os.getenv("SIMPLE_MOVE_CHECKPOINT_S", "8.0"))

    FORWARD_MPS = float(os.getenv("SIMPLE_FORWARD_MPS", "0.10"))
    TURN_DPS = float(os.getenv("SIMPLE_TURN_DPS", "50"))
    APPROACH_VX = float(os.getenv("SIMPLE_APPROACH_VX", "0.18"))
    CREEP_VX = float(os.getenv("SIMPLE_CREEP_VX", "0.035"))
    TURN_OMEGA = float(os.getenv("SIMPLE_TURN_OMEGA", "0.25"))
    MAX_PRIMITIVE_S = float(os.getenv("SIMPLE_MAX_PRIMITIVE_S", "30"))
    MIN_STEP_M = 0.05
    MAX_STEP_M = 2.5

    CREEP_ENTER_HF = float(os.getenv("SIMPLE_CREEP_ENTER_HF", "0.42"))
    SUPER_CLOSE_HF = float(os.getenv("SIMPLE_SUPER_CLOSE_HF", "0.78"))
    FINAL_CREEP_TIMEOUT_S = float(os.getenv("SIMPLE_FINAL_CREEP_TIMEOUT_S", "12.0"))
    FORCE_SUPER_CLOSE_HF = float(os.getenv("SIMPLE_FORCE_SUPER_CLOSE_HF", "0.88"))

    # Stop for pickup distance. With RangeFinder depth available, this is the main criterion.
    SUPER_CLOSE_DISTANCE_M = float(os.getenv("SIMPLE_SUPER_CLOSE_DISTANCE_M", "0.30"))

    _follow_look_alpha = float(os.getenv("NAO_FOLLOW_LOOK_ALPHA", "0.22"))
    _follow_pitch_gain = float(os.getenv("NAO_FOLLOW_PITCH_GAIN", "0.62"))
    _follow_floor_boost = float(os.getenv("NAO_FOLLOW_FLOOR_PITCH_BOOST", "0.14"))

    def __init__(self, robot, bus, on_status: Callable[..., None]):
        self._robot      = robot
        self._bus        = bus
        self._on_status  = on_status

        self._mode:       str   = "IDLE"
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
        self._last_floor_pitch_nudge: float = -999.0
        self._feedback_aliases: Optional[List[str]] = None
        self._final_creep_start: Optional[float] = None

    @property
    def is_idle(self) -> bool:
        return self._mode == "IDLE"

    def start_locate(self, aliases: list) -> None:
        self._mode             = "LOCATE"
        self._aliases          = [a.lower().strip() for a in aliases]
        self._elapsed          = 0.0
        self._locate_seen_time = 0.0
        self._verify_timer     = 0.0
        self._target_label     = None
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
        self._last_floor_pitch_nudge = -999.0
        self._feedback_aliases = None
        self._final_creep_start = None
        print(
            f"[SimpleExecutor] Moving toward: {self._aliases} "
            f"(pickup stop distance {self.SUPER_CLOSE_DISTANCE_M:.2f}m)"
        )
        self._robot.start_walk(vx=self.APPROACH_VX)

    def start_step_forward(self, meters: float, feedback_aliases: Optional[List[str]] = None) -> None:
        self._mode = "STEP_FWD"
        self._prim_elapsed = 0.0
        self._feedback_aliases = (
            [a.lower().strip() for a in feedback_aliases if a and str(a).strip()]
            if feedback_aliases
            else None
        )
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

    def start_step_turn(self, degrees: float, feedback_aliases: Optional[List[str]] = None) -> None:
        self._mode = "STEP_TURN"
        self._prim_elapsed = 0.0
        self._feedback_aliases = (
            [a.lower().strip() for a in feedback_aliases if a and str(a).strip()]
            if feedback_aliases
            else None
        )
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

    @staticmethod
    def _cx_abs_offset(obj: dict, pos: str) -> float:
        c = obj.get("cx_norm")
        if c is not None:
            return abs(float(c) - 0.5)
        if pos == "center":
            return 0.0
        return 0.20

    def _horiz_centered(self, obj: dict, pos: str, tol: float) -> bool:
        return self._cx_abs_offset(obj, pos) <= tol

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
                    self._robot.stop_walk()
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
                    self._robot.start_turn(degrees=360)

        elif self._mode == "MOVE":
            self._checkpoint_timer += self.POLL_INTERVAL_S

            if not match:
                self._stuck_timer += self.POLL_INTERVAL_S

                # If we recently got close, do NOT spin-recover.
                # The object is probably below the camera, so stop and look down.
                if self._max_h_frac >= 0.45:
                    self._robot.stop_walk()
                    self._robot.set_head_pitch(0.48)

                    if self._stuck_timer >= 4.0:
                        self.stop()
                        self._on_status(
                            f"SUPER_CLOSE: {self._target_label or self._aliases} "
                            f"(lost below camera after close approach) — assuming pickup range.",
                            None,
                        )
                    return

                if self._stuck_timer >= self.STUCK_THRESHOLD_S:
                    self.stop()
                    self._on_status(
                        f"LOST: Target {self._target_label or self._aliases} disappeared",
                        None,
                    )
                return

            lbl, obj = match
            self._target_label = lbl
            self._stuck_timer  = 0.0

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

            # ----------------------------------------------------------
            # IMPORTANT:
            # If RangeFinder depth exists, ONLY distance can trigger SUPER_CLOSE.
            # This prevents stopping too early just because the YOLO box is large.
            # ----------------------------------------------------------
            super_close = False

            if self._elapsed >= 0.35:
                if real_dist is not None:
                    try:
                        super_close = float(real_dist) <= self.SUPER_CLOSE_DISTANCE_M
                    except (TypeError, ValueError):
                        super_close = False
                else:
                    # Fallback only when depth is unavailable.
                    if hf_val >= self.SUPER_CLOSE_HF and centered_loose:
                        super_close = True
                    elif hf_val >= self.FORCE_SUPER_CLOSE_HF and centered_loose:
                        super_close = True
                    elif (
                        self._final_creep_start is not None
                        and creep_age >= self.FINAL_CREEP_TIMEOUT_S
                        and hf_val >= self.FORCE_SUPER_CLOSE_HF
                        and centered_loose
                    ):
                        super_close = True

            if super_close:
                self.stop()
                dist_msg = (
                    f"distance_m={float(real_dist):.3f}m"
                    if real_dist is not None
                    else f"distance={dist}"
                )
                self._on_status(
                    f"SUPER_CLOSE: {lbl} (height_frac={hf_val:.2f}, {dist_msg}) "
                    "— pickup distance threshold met.",
                    obj,
                )
                return

            if pos == "left":
                self._robot.start_walk(vx=0.02, omega=self.TURN_OMEGA)
            elif pos == "right":
                self._robot.start_walk(vx=0.02, omega=-self.TURN_OMEGA)
            else:
                use_creep = hf_val >= self.CREEP_ENTER_HF and centered_loose
                vx = self.CREEP_VX if use_creep else self.APPROACH_VX
                self._robot.start_walk(vx=vx, omega=0.0)

            cx = obj.get("cx_norm")
            cy = obj.get("cy_norm")
            if cx is None or cy is None:
                cx = {"left": 0.32, "center": 0.5, "right": 0.68}.get(pos, 0.5)
                cy = 0.65

            self._robot.look_at_normalised(
                cx,
                cy,
                alpha=self._follow_look_alpha,
                pitch_gain=self._follow_pitch_gain,
                floor_pitch_boost=self._follow_floor_boost,
            )

            cy_val = float(cy) if cy is not None else 0.5
            if cy_val >= 0.72 and (self._elapsed - self._last_floor_pitch_nudge) >= 0.35:
                extra = min(0.09, 0.038 + (cy_val - 0.72) * 0.45)
                self._robot.adjust_head_pitch(extra)
                self._last_floor_pitch_nudge = self._elapsed

            curr_h = (
                float(obj.get("height_frac"))
                if obj.get("height_frac") is not None
                else _height_frac_from_bucket(obj.get("distance", "far"))
            )

            if curr_h > self._max_h_frac:
                self._max_h_frac = curr_h
                self._stuck_timer = 0.0
            else:
                self._stuck_timer += self.POLL_INTERVAL_S

            if self._checkpoint_timer >= self.MOVE_CHECKPOINT_S:
                self.stop()
                hf = obj.get("height_frac")
                hf_s = f"{hf}" if hf is not None else "?"
                dist = obj.get("distance", "?")
                pos = obj.get("position", "?")
                real_dist_msg = (
                    f" distance_m={float(real_dist):.3f}m"
                    if real_dist is not None
                    else ""
                )

                if self._stuck_timer >= self.STUCK_THRESHOLD_S:
                    self._on_status(
                        f"STUCK: {lbl} | height_frac={hf_s} distance={dist} "
                        f"position={pos}{real_dist_msg}",
                        obj,
                    )
                else:
                    self._on_status(
                        f"PROGRESS: Approaching {lbl} | height_frac={hf_s} "
                        f"distance={dist} position={pos}{real_dist_msg}",
                        obj,
                    )

    def _latest_scene(self) -> Optional[dict]:
        latest = self._bus.get_latest("scene_state") if self._bus else None
        if not latest:
            return None
        state = latest[0] if isinstance(latest, (tuple, list)) else latest
        return state if isinstance(state, dict) else None

    def _update_step_forward(self) -> None:
        self._prim_elapsed += self.POLL_INTERVAL_S
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