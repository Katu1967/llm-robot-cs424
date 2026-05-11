import math
import os
from typing import Callable, List, Optional, Tuple

from simple_search import format_feedback_line, match_target_in_scene


class SimpleExecutor:
    """
    Executor for NAO robot navigation and object approach.
    
    locate_object: 360° spin + YOLO verify (FOUND / TIMEOUT).
    move_to_object: vision-guided walk toward target, stops at STOP_DISTANCE_M by RangeFinder depth.
    Periodic APPROACH_CHECKPOINT pauses for LLM safety replanning (~10s).
    Head continuously tracks target bbox during approach.
    """

    SPIN_DURATION_S       = 15.0
    POLL_INTERVAL_S       = 0.05
    STUCK_THRESHOLD_S     = 2.0

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
    STOP_DISTANCE_M = float(os.getenv("SIMPLE_STOP_DISTANCE_M", "0.22"))

    _follow_look_alpha = float(os.getenv("NAO_FOLLOW_LOOK_ALPHA", "0.14"))
    _follow_pitch_gain = float(os.getenv("NAO_FOLLOW_PITCH_GAIN", "0.26"))
    _head_cy_deadband = float(os.getenv("SIMPLE_HEAD_CY_DEADBAND", "0.07"))

    def __init__(self, robot, bus, on_status: Callable[..., None]):
        self._robot      = robot
        self._bus        = bus
        self._on_status  = on_status

        self._mode:       str   = "IDLE"
        self._aliases:    list  = []
        self._elapsed:    float = 0.0
        self._last_poll:  float = 0.0

        self._locate_seen_time: float = 0.0
        self._verify_timer:     float = 0.0

        self._target_label:     Optional[str] = None
        self._max_h_frac:       float         = 0.0
        self._stuck_timer:      float         = 0.0
        self._checkpoint_timer: float         = 0.0
        self._recovery_timer:   float         = 0.0

        self._prim_elapsed:      float = 0.0
        self._step_target_m:     float = 0.0
        self._step_start_xy:     Optional[Tuple[float, float]] = None
        self._step_time_budget:  float = 0.0
        self._turn_target_deg:   float = 0.0
        self._turn_time_budget:  float = 0.0
        self._feedback_aliases:  Optional[List[str]] = None
        self._final_creep_start: Optional[float] = None
        self._replan_pause:      bool = False
        
        self._last_known_dist_m: Optional[float] = None
        self._last_follow_cx:    Optional[float] = None
        self._last_follow_cy:    Optional[float] = None
        self._last_follow_pos:   str = "center"
        self._last_follow_valid: bool = False
        self._step_track_head:   bool = False
        self._blind_move_timer:  float = 0.0
        self._blind_move_budget: float = 0.0

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
        self._recovery_timer   = 0.0
        self._max_h_frac       = 0.0
        self._target_label     = None
        self._feedback_aliases = None
        self._final_creep_start = None
        self._replan_pause     = False
        self._clear_follow_memory()
        self._step_track_head  = False
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
                if feedback_aliases else None
            )
        else:
            self._feedback_aliases = None
            self._robot.reset_head_neutral()
            
        self._step_target_m = max(self.MIN_STEP_M, min(float(meters), self.MAX_STEP_M))
        self._step_start_xy = None
        
        current_gps = self._robot.get_gps_position()
        if current_gps is not None and len(current_gps) >= 2:
            self._step_start_xy = (float(current_gps[0]), float(current_gps[1]))
            
        base_time_seconds = self._step_target_m / max(self.FORWARD_MPS, 1e-6)
        self._step_time_budget = base_time_seconds * (1.35 if self._step_start_xy else 1.55)
        
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
                if feedback_aliases else None
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
    def _cx_abs_offset(obj: dict, position: str) -> float:
        center_x_norm = obj.get("cx_norm")
        if center_x_norm is not None:
            return abs(float(center_x_norm) - 0.5)
        if position == "center":
            return 0.0
        return 0.20

    def _horiz_centered(self, obj: dict, position: str, tolerance: float) -> bool:
        return self._cx_abs_offset(obj, position) <= tolerance

    def _head_follow_bbox(self, obj: dict) -> None:
        position = obj.get("position", "center")
        center_y_norm = obj.get("cy_norm")
        if center_y_norm is None:
            center_y_norm = 0.65 if str(position) in ("left", "right", "center") else 0.5
            
        alpha = max(0.04, min(1.0, self._follow_look_alpha))
        self._robot.look_pitch_from_cy_norm(
            float(center_y_norm),
            alpha=alpha,
            pitch_gain=self._follow_pitch_gain,
            cy_deadband=self._head_cy_deadband,
        )

    def start_crouch(self) -> None:
        self._mode = "CROUCH"
        self._prim_elapsed = 0.0

    def start_pick(self) -> None:
        self._mode = "PICK"
        self._prim_elapsed = 0.0

    def _remember_follow_target(self, obj: dict) -> None:
        center_x_norm = obj.get("cx_norm")
        center_y_norm = obj.get("cy_norm")
        position = obj.get("position", "center")
        
        if center_x_norm is not None and center_y_norm is not None:
            self._last_follow_cx = float(center_x_norm)
            self._last_follow_cy = float(center_y_norm)
            self._last_follow_pos = str(position)
            self._last_follow_valid = True
        else:
            self._last_follow_cx = {"left": 0.32, "center": 0.5, "right": 0.68}.get(str(position), 0.5)
            self._last_follow_cy = 0.65
            self._last_follow_pos = str(position)
            self._last_follow_valid = True

    def _head_follow_scene_aliases(self, aliases: Optional[List[str]]) -> None:
        if not aliases:
            return
        scene = self._latest_scene()
        if not scene:
            return
            
        locked_label = self._target_label if self._target_label else None
        matched_target = match_target_in_scene(scene, aliases, locked_label=locked_label)
        if matched_target:
            _label, detected_object = matched_target
            self._head_follow_bbox(detected_object)

    def _clear_follow_memory(self) -> None:
        self._last_follow_valid = False
        self._last_follow_cx = None
        self._last_follow_cy = None
        self._last_follow_pos = "center"

    def tick(self, delta_time: float) -> None:
        if self._mode == "IDLE":
            return

        self._elapsed   += delta_time
        self._last_poll += delta_time

        if self._last_poll >= self.POLL_INTERVAL_S:
            self._last_poll = 0.0
            self._update(delta_time)

    def _update_crouch(self) -> None:
        self._prim_elapsed += self.POLL_INTERVAL_S
        elapsed_time = self._prim_elapsed

        if elapsed_time <= self.POLL_INTERVAL_S * 2:
            self._robot.set_joint("LHipPitch",   -0.70)
            self._robot.set_joint("RHipPitch",   -0.70)
            self._robot.set_joint("LKneePitch",   1.30)
            self._robot.set_joint("RKneePitch",   1.30)
            self._robot.set_joint("LAnklePitch", -0.55)
            self._robot.set_joint("RAnklePitch", -0.55)

        if elapsed_time >= 1.8:
            self.stop()
            self._on_status("ACTION_DONE: Crouch complete. What next?", None)

    def _update_pick(self) -> None:
        self._prim_elapsed += self.POLL_INTERVAL_S
        elapsed_time = self._prim_elapsed

        if elapsed_time <= 0.8:
            if elapsed_time <= self.POLL_INTERVAL_S * 2:
                self._robot.set_head_pitch(0.5)
                self._robot.set_joint("LHand", 1.0)
                self._robot.set_joint("RHand", 1.0)

        elif 0.8 < elapsed_time <= 2.0:
            if 0.8 < elapsed_time <= 0.8 + self.POLL_INTERVAL_S * 2:
                self._robot.set_joint("LHipPitch",    -0.70)
                self._robot.set_joint("RHipPitch",    -0.70)
                self._robot.set_joint("LKneePitch",    1.30)
                self._robot.set_joint("RKneePitch",    1.30)
                self._robot.set_joint("LAnklePitch",  -0.55)
                self._robot.set_joint("RAnklePitch",  -0.55)
                self._robot.set_joint("LShoulderPitch",  1.80)
                self._robot.set_joint("RShoulderPitch",  1.80)
                self._robot.set_joint("LShoulderRoll",   0.15)
                self._robot.set_joint("RShoulderRoll",  -0.15)
                self._robot.set_joint("LElbowYaw",      -1.50)
                self._robot.set_joint("RElbowYaw",       1.50)
                self._robot.set_joint("LElbowRoll",     -0.10)
                self._robot.set_joint("RElbowRoll",      0.10)

        elif 2.0 < elapsed_time <= 2.6:
            if 2.0 < elapsed_time <= 2.0 + self.POLL_INTERVAL_S * 2:
                self._robot.set_joint("LHand", 0.0)
                self._robot.set_joint("RHand", 0.0)

        elif 2.6 < elapsed_time <= 4.0:
            if 2.6 < elapsed_time <= 2.6 + self.POLL_INTERVAL_S * 2:
                self._robot.set_joint("LHipPitch",   -0.45)
                self._robot.set_joint("RHipPitch",   -0.45)
                self._robot.set_joint("LKneePitch",   0.87)
                self._robot.set_joint("RKneePitch",   0.87)
                self._robot.set_joint("LAnklePitch", -0.41)
                self._robot.set_joint("RAnklePitch", -0.41)
                self._robot.set_joint("LShoulderPitch",  1.20)
                self._robot.set_joint("RShoulderPitch",  1.20)
                self._robot.set_joint("LShoulderRoll",   0.20)
                self._robot.set_joint("RShoulderRoll",  -0.20)
                self._robot.set_joint("LElbowYaw",      -1.50)
                self._robot.set_joint("RElbowYaw",       1.50)
                self._robot.set_joint("LElbowRoll",     -0.80)
                self._robot.set_joint("RElbowRoll",      0.80)
                self._robot.set_head_pitch(0.0)

        if elapsed_time >= 4.5:
            self.stop()
            self._on_status("ACTION_DONE: Pick object complete. What next?", None)

    def _update(self, delta_time: float):
        if self._mode == "CROUCH":
            self._update_crouch()
            return
        if self._mode == "PICK":
            self._update_pick()
            return
        if self._mode == "STEP_FWD":
            self._update_step_forward()
            return
        if self._mode == "STEP_TURN":
            self._update_step_turn()
            return

        matched_target = self._check_scene()

        if self._mode == "LOCATE":
            if matched_target:
                self._locate_seen_time += self.POLL_INTERVAL_S
                if self._locate_seen_time >= 0.2:
                    label, detected_object = matched_target
                    self._target_label = label
                    self._robot.stop_locomotion_only()
                    self._mode = "VERIFY_LOCATE"
                    self._verify_timer = 0.0
                    print(f"[SimpleExecutor] Potential '{label}' spotted. Stopping to verify...")
            else:
                self._locate_seen_time = 0.0

            if self._elapsed >= self.SPIN_DURATION_S:
                self.stop()
                self._on_status("TIMEOUT: Object not found during spin", None)

        elif self._mode == "VERIFY_LOCATE":
            self._verify_timer += self.POLL_INTERVAL_S
            if self._verify_timer >= 1.0:
                if matched_target:
                    label, detected_object = matched_target
                    self.stop()
                    self._on_status(f"FOUND: {label}", detected_object)
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

            if not matched_target:
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
                    print("[SimpleExecutor] Target lost. Initiating vertical look recovery...")
                    self._robot.stop_locomotion_only()
                    self._mode = "RECOVERY_LOOK"
                    self._recovery_timer = 0.0
                return

            label, detected_object = matched_target
            self._target_label = label
            self._stuck_timer = 0.0
            self._remember_follow_target(detected_object)
            self._head_follow_bbox(detected_object)

            distance_bucket = detected_object.get("distance", "far")
            real_distance_m = detected_object.get("distance_m")
            position = detected_object.get("position", "center")
            
            if real_distance_m is not None:
                try:
                    self._last_known_dist_m = float(real_distance_m)
                except (TypeError, ValueError):
                    pass
                    
            height_frac_obj = detected_object.get("height_frac")
            height_frac_val = float(height_frac_obj) if height_frac_obj is not None else _height_frac_from_bucket(distance_bucket)
            centered_loose = self._horiz_centered(detected_object, position, 0.17)
            centered_tight = self._horiz_centered(detected_object, position, 0.12)

            if height_frac_val < self.CREEP_ENTER_HF - 0.06:
                self._final_creep_start = None
            elif centered_loose and self._final_creep_start is None:
                self._final_creep_start = self._elapsed

            creep_age = (self._elapsed - self._final_creep_start) if self._final_creep_start is not None else 0.0

            super_close = False
            if real_distance_m is not None and centered_loose:
                try:
                    if float(real_distance_m) <= self.STOP_DISTANCE_M:
                        super_close = True
                except (TypeError, ValueError):
                    pass

            if not super_close and real_distance_m is None and self._elapsed >= 0.35:
                if height_frac_val >= 0.66 and self._horiz_centered(detected_object, position, 0.22):
                    super_close = True
                elif height_frac_val >= self.SUPER_CLOSE_HF and centered_loose:
                    super_close = True
                elif height_frac_val >= 0.56 and distance_bucket == "very_near" and centered_tight:
                    super_close = True
                elif (
                    self._final_creep_start is not None
                    and creep_age >= self.FINAL_CREEP_TIMEOUT_S
                    and height_frac_val >= self.FORCE_SUPER_CLOSE_HF
                    and centered_loose
                ):
                    super_close = True

            if super_close:
                self._replan_pause = False
                self._head_follow_bbox(detected_object)
                
                distance_feet_str = ""
                if real_distance_m is not None:
                    try:
                        distance_meters = float(real_distance_m)
                        distance_feet_str = f" (~{distance_meters * 3.280839895013123:.2f} ft, {distance_meters:.3f} m RangeFinder)"
                    except (TypeError, ValueError):
                        pass
                        
                self.stop()
                distance_msg = (
                    f"distance_m={float(real_distance_m):.3f} m{distance_feet_str}"
                    if real_distance_m is not None
                    else f"distance={distance_bucket}"
                )
                self._on_status(
                    f"SUPER_CLOSE: {label} (height_frac={height_frac_val:.2f}, {distance_msg}) "
                    "— final approach threshold met.",
                    detected_object,
                )
                return

            current_height_frac = float(detected_object.get("height_frac")) if detected_object.get("height_frac") is not None else _height_frac_from_bucket(
                detected_object.get("distance", "far")
            )

            if current_height_frac > self._max_h_frac:
                self._max_h_frac = current_height_frac
                self._stuck_timer = 0.0
            else:
                self._stuck_timer += self.POLL_INTERVAL_S

            stuck_budget = float(os.getenv("SIMPLE_APPROACH_STUCK_S", "8.0"))
            if self._stuck_timer >= stuck_budget:
                self._replan_pause = False
                self.stop()
                height_frac = detected_object.get("height_frac")
                height_frac_str = f"{height_frac}" if height_frac is not None else "?"
                self._on_status(
                    f"STUCK: {label} | height_frac={height_frac_str} distance={distance_bucket} position={position} "
                    f"(no bbox growth ~{stuck_budget:.0f}s)",
                    detected_object,
                )
                return

            if self._checkpoint_timer >= self.APPROACH_REPLAN_S:
                self._checkpoint_timer = 0.0
                self._replan_pause = True
                self._robot.stop_locomotion_only()
                scene = self._latest_scene() or {}
                sonar = scene.get("sonar", {})
                left_m = sonar.get("left_m", "?")
                right_m = sonar.get("right_m", "?")
                distance_m_display = f"distance_m={real_distance_m}" if real_distance_m is not None else "distance_m=n/a"
                
                self._on_status(
                    f"APPROACH_CHECKPOINT: Feet paused (head still tracking). Approaching '{label}'. "
                    f"{distance_m_display}, height_frac={detected_object.get('height_frac')}, bbox position={position}. "
                    f"Sonar left_m={left_m}, right_m={right_m}. "
                    f"Goal: approach until distance_m <= {self.STOP_DISTANCE_M:.2f} m. "
                    "Return **move_to_object** with the **same aliases** to continue, "
                    "or **turn_degrees** / **move_forward** (small) to dodge. "
                    "Head pitch follows the target vertically automatically (not LLM) — one action per response.",
                    detected_object,
                )
                return

            if self._replan_pause:
                return

            if position == "left":
                self._robot.start_walk(vx=0.0, omega=self.TURN_OMEGA)
            elif position == "right":
                self._robot.start_walk(vx=0.0, omega=-self.TURN_OMEGA)
            else:
                use_creep = height_frac_val >= self.CREEP_ENTER_HF and centered_loose
                velocity_x = self.CREEP_VX if use_creep else self.APPROACH_VX
                self._robot.start_walk(vx=velocity_x, omega=0.0)

        elif self._mode == "RECOVERY_LOOK":
            self._recovery_timer += self.POLL_INTERVAL_S
            
            if matched_target:
                label, detected_object = matched_target
                print(f"[SimpleExecutor] Target '{label}' re-acquired during sweep! Resuming approach.")
                self._target_label = label
                self._stuck_timer = 0.0
                self._mode = "MOVE"
                self._remember_follow_target(detected_object)
                self._robot.start_walk(vx=self.APPROACH_VX)
                return
            
            if self._recovery_timer < 1.5:
                self._robot.set_head_pitch(-0.4) 
            elif self._recovery_timer < 3.5:
                self._robot.set_head_pitch(0.5)  
            elif self._recovery_timer < 4.5:
                self._robot.set_head_pitch(0.0)  
            else:
                if self._last_known_dist_m is not None and self._last_known_dist_m < 1.5:
                    remaining_dist = max(0.0, self._last_known_dist_m - self.STOP_DISTANCE_M)
                    
                    # Add startup friction buffer to the blind move
                    base_time_seconds = remaining_dist / max(self.APPROACH_VX, 0.01)
                    self._blind_move_budget = max(1.5, base_time_seconds * 1.5)
                    
                    print(f"[SimpleExecutor] Sweep failed, but last known distance was {self._last_known_dist_m:.2f}m. Initiating blind approach for {remaining_dist:.2f}m (~{self._blind_move_budget:.1f}s).")
                    
                    self._robot.reset_head_neutral()
                    self._robot.start_walk(vx=self.APPROACH_VX)
                    self._mode = "BLIND_APPROACH"
                    self._blind_move_timer = 0.0
                else:
                    self.stop()
                    self._on_status(
                        f"LOST: Target {self._target_label or self._aliases} disappeared",
                        None,
                    )
                    
        elif self._mode == "BLIND_APPROACH":
            self._blind_move_timer += self.POLL_INTERVAL_S
            
            if matched_target:
                label, detected_object = matched_target
                print(f"[SimpleExecutor] Target '{label}' re-acquired during blind approach! Resuming normal tracking.")
                self._target_label = label
                self._stuck_timer = 0.0
                self._mode = "MOVE"
                self._remember_follow_target(detected_object)
                self._robot.start_walk(vx=self.APPROACH_VX)
                return
            
            if self._blind_move_timer >= self._blind_move_budget:
                self.stop()
                self._on_status(
                    f"SUPER_CLOSE: {self._target_label or self._aliases} (Blind approach complete) — final approach threshold met.",
                    None,
                )

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
            
        target_reached = False
        if self._step_start_xy is not None:
            current_pos = self._robot.get_gps_position()
            if current_pos is not None and len(current_pos) >= 2:
                distance_traveled = math.hypot(
                    float(current_pos[0]) - self._step_start_xy[0],
                    float(current_pos[1]) - self._step_start_xy[1],
                )
                if distance_traveled >= self._step_target_m * 0.9:
                    target_reached = True
                    
        if not target_reached and self._prim_elapsed >= self._step_time_budget:
            target_reached = True
            
        if self._prim_elapsed >= self.MAX_PRIMITIVE_S:
            self.stop()
            self._emit_step_done(
                f"STEP_ABORT: move_forward exceeded {self.MAX_PRIMITIVE_S:.0f}s",
            )
            return
            
        if target_reached:
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

    def _emit_step_done(self, base_status: str) -> None:
        scene = self._latest_scene()
        if self._feedback_aliases:
            feedback_message = format_feedback_line(scene, self._feedback_aliases)
            self._on_status(f"{base_status}\n{feedback_message}", None)
        else:
            self._on_status(base_status, None)
        self._feedback_aliases = None

    def _check_scene(self):
        scene = self._latest_scene()
        if not scene:
            return None
        objects = scene.get("objects", [])
        for detected_object in objects:
            label = detected_object.get("label", "").lower()
            if self._target_label and label != self._target_label:
                continue
            for alias in self._aliases:
                if alias in label or label in alias:
                    return label, detected_object
        return None


def _height_frac_from_bucket(bucket: str) -> float:
    height_mapping = {"very_near": 0.45, "near": 0.26, "medium": 0.10, "far": 0.04}
    return height_mapping.get(bucket, 0.04)