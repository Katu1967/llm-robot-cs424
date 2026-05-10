"""
PlanExecutor — NAO robot execution layer.
"""

import time
import hashlib
import json
import logging
from enum import Enum, auto
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class ExecutorState(Enum):
    IDLE    = auto()
    RUNNING = auto()
    PAUSED  = auto()
    DONE    = auto()
    FAILED  = auto()


class PlanExecutor:
    def __init__(self, robot_interface, scene_bus=None):
        self.robot = robot_interface
        self.bus   = scene_bus
        self.state = ExecutorState.IDLE

        self._plan:        list                 = []
        self._step_index:  int                  = 0
        self._active_gen:  Optional[Generator]  = None
        self._plan_hash:   str                  = ""
        self._step_result: str                  = "ok"

        self._handlers = {
            "turn_left":          self._act_turn,
            "turn_right":         self._act_turn,
            "adjust_orientation": self._act_turn,
            "move_forward":       self._act_move_forward,
            "move_backward":      self._act_move_backward,
            "stop":               self._act_stop,
            "set_head_yaw":       self._act_set_head_yaw,
            "set_head_pitch":     self._act_set_head_pitch,
            "wave":               self._act_wave,
            "move_toward_object": self._act_move_toward_object,
            "center_on_object":   self._act_center_on_object,
            "look_for_object":    self._act_look_for_object,
            "pick_object":        self._act_pick_object,
            "place_object":       self._act_place_object,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_plan(self, plan_json: dict) -> bool:
        steps = plan_json.get("plan", [])
        if not steps:
            logger.warning("PlanExecutor: empty or missing 'plan' key — skipping")
            return False

        plan_hash = _hash_plan(steps)
        if plan_hash == self._plan_hash and self.state == ExecutorState.RUNNING:
            logger.debug("PlanExecutor: plan unchanged and still running — not reloading")
            return False

        logger.info(f"PlanExecutor: loading plan with {len(steps)} step(s)")
        self._plan       = steps
        self._step_index = 0
        self._plan_hash  = plan_hash
        self._active_gen = None
        self.state       = ExecutorState.RUNNING
        self._advance()
        return True

    def tick(self):
        if self.state != ExecutorState.RUNNING:
            return
        if self._active_gen is None:
            return

        try:
            next(self._active_gen)
        except StopIteration as e:
            result = e.value if e.value else "ok"
            logger.debug(f"PlanExecutor: step {self._step_index} finished → {result}")
            self._on_step_done(result)
        except Exception as exc:
            import traceback
            logger.error(f"PlanExecutor: step {self._step_index} raised: {exc}")
            traceback.print_exc()
            self._on_step_done("failed")

    def reset(self):
        self._active_gen = None
        self._plan       = []
        self._step_index = 0
        self._plan_hash  = ""
        self.state       = ExecutorState.IDLE
        self.robot.stop_walk()

    @property
    def is_idle(self) -> bool:
        return self.state in (ExecutorState.IDLE, ExecutorState.DONE, ExecutorState.FAILED)

    @property
    def current_step(self) -> Optional[dict]:
        if 0 <= self._step_index < len(self._plan):
            return self._plan[self._step_index]
        return None

    # ------------------------------------------------------------------
    # Step machine
    # ------------------------------------------------------------------

    def _advance(self):
        if self._step_index >= len(self._plan):
            print("[PlanExecutor] ✓ Plan complete!")
            self.state = ExecutorState.DONE
            self.robot.stop_walk()
            return

        step   = self._plan[self._step_index]
        action = step.get("action", "").lower()
        params = step.get("parameters", {})

        print(f"[PlanExecutor] Step {self._step_index + 1}/{len(self._plan)}: {action} {params}")

        handler = self._handlers.get(action)
        if handler is None:
            print(f"[PlanExecutor] WARNING: unknown action '{action}' — skipping")
            self._on_step_done("skipped")
            return

        self._active_gen = handler(params)

    def _on_step_done(self, result: str):
        self._active_gen = None

        if result == "failed":
            self.state = ExecutorState.FAILED
            print(f"[PlanExecutor] ✗ FAILED at step {self._step_index + 1}")
            return

        self._step_index += 1
        self._advance()

    # ------------------------------------------------------------------
    # Primitive actions
    # ------------------------------------------------------------------

    def _act_turn(self, params: dict) -> Generator:
        degrees = float(params.get("degrees", 30))
        action  = self.current_step.get("action", "turn_left")

        degrees = -abs(degrees) if action == "turn_right" else abs(degrees)

        duration_s = abs(degrees) / 45.0
        frames     = max(1, int(duration_s / self.robot.timestep_s))

        print(f"[PlanExecutor] turn: {degrees:.1f}° over {duration_s:.1f}s ({frames} frames)")

        self.robot.start_turn(degrees)

        for _ in range(frames):
            yield

        self.robot.stop_walk()
        return "ok"

    def _act_move_forward(self, params: dict) -> Generator:
        meters = float(params.get("meters", params.get("distance_m", 0.5)))
        speed  = 0.12
        frames = max(1, int(meters / speed / self.robot.timestep_s))

        print(f"[PlanExecutor] move_forward: {meters}m ({frames} frames)")

        self.robot.start_walk(vx=speed, vy=0, omega=0)

        for _ in range(frames):
            yield

        self.robot.stop_walk()
        return "ok"

    def _act_move_backward(self, params: dict) -> Generator:
        meters = float(params.get("meters", params.get("distance_m", 0.3)))
        speed  = 0.10
        frames = max(1, int(meters / speed / self.robot.timestep_s))

        print(f"[PlanExecutor] move_backward: {meters}m ({frames} frames)")

        self.robot.start_walk(vx=-speed, vy=0, omega=0)

        for _ in range(frames):
            yield

        self.robot.stop_walk()
        return "ok"

    def _act_stop(self, params: dict) -> Generator:
        self.robot.stop_walk()
        yield
        return "ok"

    def _act_set_head_yaw(self, params: dict) -> Generator:
        angle  = float(params.get("angle", 0.0))
        frames = int(1.0 / self.robot.timestep_s)

        self.robot.set_head_yaw(angle)

        for _ in range(frames):
            yield

        return "ok"

    def _act_set_head_pitch(self, params: dict) -> Generator:
        angle  = float(params.get("angle", 0.0))
        frames = int(1.0 / self.robot.timestep_s)

        self.robot.set_head_pitch(angle)

        for _ in range(frames):
            yield

        return "ok"

    def _act_wave(self, params: dict) -> Generator:
        import math

        frames_per_cycle = int(0.5 / self.robot.timestep_s)
        cycles = int(params.get("cycles", 2))

        for c in range(cycles * 2):
            angle = math.pi / 4 if c % 2 == 0 else 0.0

            self.robot.set_joint("RShoulderPitch", -angle)
            self.robot.set_joint("RShoulderRoll",   angle)
            self.robot.set_joint("RElbowRoll",      angle * 1.5)

            for _ in range(frames_per_cycle):
                yield

        self.robot.set_joint("RShoulderPitch", 1.4)
        self.robot.set_joint("RShoulderRoll", -0.3)
        self.robot.set_joint("RElbowRoll",     0.5)

        for _ in range(frames_per_cycle):
            yield

        return "ok"

    # ------------------------------------------------------------------
    # Semantic actions
    # ------------------------------------------------------------------

    def _act_center_on_object(self, params: dict) -> Generator:
        label     = params.get("label", "")
        tolerance = float(params.get("tolerance", 0.10))
        timeout_s = float(params.get("timeout_s", 5.0))
        deadline  = time.time() + timeout_s

        print(f"[PlanExecutor] center_on_object: '{label}'")

        while time.time() < deadline:
            obj = self._find_object(label)

            if obj is None:
                yield
                continue

            cx_norm = obj["cx_norm"]
            error   = cx_norm - 0.5

            if abs(error) < tolerance:
                print(f"[PlanExecutor] center_on_object: '{label}' centred ✓")
                return "ok"

            head_yaw = self.robot.get_head_yaw()

            if abs(head_yaw) > 1.5:
                self.robot.start_turn(-20.0 if error > 0 else 20.0)
            else:
                self.robot.stop_walk()
                self.robot.adjust_head_yaw(-error * 0.6)

            yield

        print(f"[PlanExecutor] center_on_object: timeout for '{label}'")
        return "failed"

    def _act_move_toward_object(self, params: dict) -> Generator:
        """
        Walk toward an object using live YOLO and scene feedback.

        Important behavior:
        - If the object is visible, keep trying even after timeout_s.
        - Only fail from timeout if the object is no longer visible.
        - Stop when vision / proximity heuristics indicate arrival (see executor params).
        """

        label           = params.get("label", "")
        timeout_s       = float(params.get("timeout_s", 30.0))
        stop_distance_m = float(params.get("stop_distance_m", 0.40))
        deadline        = time.time() + timeout_s

        print(
            f"[PlanExecutor] move_toward_object: '{label}' "
            f"stop_distance={stop_distance_m}m timeout={timeout_s}s "
            f"(timeout only fails if object is lost)"
        )

        lost_count   = 0
        last_seen_cx = 0.5
        last_seen_cy = 0.6
        seen_once    = False
        tick         = 0

        while True:
            tick += 1
            obj = self._find_object(label)

            # ----------------------------------------------------------
            # Object visible
            # ----------------------------------------------------------
            if obj is not None:
                last_seen_cx = obj["cx_norm"]
                last_seen_cy = obj["cy_norm"]
                lost_count   = 0
                seen_once    = True

            # ----------------------------------------------------------
            # Track head toward last known target position
            # ----------------------------------------------------------
            try:
                self.robot.look_at_normalised(last_seen_cx, last_seen_cy)
            except Exception:
                pass

            # ----------------------------------------------------------
            # Debug print
            # ----------------------------------------------------------
            if tick % 30 == 0:
                try:
                    gps = self.robot.get_gps_position()
                    gps_str = f" gps={gps}"
                except Exception:
                    gps_str = ""

                time_left = deadline - time.time()

                if obj is not None:
                    print(
                        f"[move_toward] tick={tick} "
                        f"cx={obj['cx_norm']:.2f} "
                        f"h={obj['h_norm']:.3f} "
                        f"dist={obj.get('distance_m')} "
                        f"stop={stop_distance_m} "
                        f"lost={lost_count} "
                        f"time_left={time_left:.1f}s"
                        f"{gps_str}"
                    )
                else:
                    print(
                        f"[move_toward] tick={tick} "
                        f"OBJECT NOT VISIBLE "
                        f"lost_count={lost_count} "
                        f"last_cx={last_seen_cx:.2f} "
                        f"time_left={time_left:.1f}s"
                        f"{gps_str}"
                    )

            # ----------------------------------------------------------
            # If object is lost, search/recover.
            # Only fail if timeout has passed AND object is still lost.
            # ----------------------------------------------------------
            if obj is None:
                lost_count += 1

                if time.time() >= deadline:
                    self.robot.stop_walk()
                    print(
                        f"[move_toward] TIMEOUT for '{label}' after {timeout_s}s "
                        f"— object is not visible"
                    )
                    return "failed"

                if seen_once and lost_count > 15:
                    turn_dir = 20.0 if last_seen_cx < 0.5 else -20.0
                    self.robot.start_turn(turn_dir)
                    print(
                        f"[move_toward] lost — turning "
                        f"{'left' if turn_dir > 0 else 'right'} to recover"
                    )
                else:
                    self.robot.start_walk(vx=0.08, vy=0, omega=0)

                yield
                continue

            # ----------------------------------------------------------
            # Object visible: do NOT fail just because timeout passed.
            # Keep going until close enough.
            # ----------------------------------------------------------
            distance_m = obj.get("distance_m")

            if distance_m is not None and distance_m <= stop_distance_m:
                self.robot.stop_walk()
                print(
                    f"[move_toward] REACHED '{label}' "
                    f"— distance={distance_m:.3f}m <= {stop_distance_m:.3f}m ✓"
                )
                return "ok"

            # ----------------------------------------------------------
            # Steering control
            # ----------------------------------------------------------
            error = last_seen_cx - 0.5

            # If object is pretty centered, walk forward.
            # If it is off-center, turn in place first.
            omega = -error * 1.0

            self.robot.start_walk(vx=0.15, vy=0, omega=omega)

            yield

    def _act_look_for_object(self, params: dict) -> Generator:
        label     = params.get("label", "")
        timeout_s = float(params.get("timeout_s", 10.0))
        deadline  = time.time() + timeout_s

        print(f"[PlanExecutor] look_for_object: '{label}'")

        self.robot.start_turn(degrees=360)

        while time.time() < deadline:
            obj = self._find_object(label)

            if obj is not None:
                self.robot.stop_walk()
                print(f"[PlanExecutor] look_for_object: found '{label}' ✓")
                return "ok"

            yield

        self.robot.stop_walk()
        print(f"[PlanExecutor] look_for_object: '{label}' not found (timeout)")
        return "failed"

    def _act_pick_object(self, params: dict) -> Generator:
        label = params.get("label", "")
        print(f"[PlanExecutor] pick_object: grasping '{label}'")

        f = self.robot.timestep_s

        def frames(secs):
            return max(1, int(secs / f))

        # 1. Look down
        self.robot.set_head_pitch(0.4)

        for _ in range(frames(0.5)):
            yield

        # 2. Reach arms forward
        self.robot.set_joint("LShoulderPitch",  1.5)
        self.robot.set_joint("RShoulderPitch",  1.5)
        self.robot.set_joint("LShoulderRoll",   0.1)
        self.robot.set_joint("RShoulderRoll",  -0.1)
        self.robot.set_joint("LElbowYaw",      -0.5)
        self.robot.set_joint("RElbowYaw",       0.5)
        self.robot.set_joint("LElbowRoll",     -0.3)
        self.robot.set_joint("RElbowRoll",      0.3)

        for _ in range(frames(0.6)):
            yield

        # 3. Crouch
        self.robot.set_joint("LHipPitch",   -0.7)
        self.robot.set_joint("RHipPitch",   -0.7)
        self.robot.set_joint("LKneePitch",   1.2)
        self.robot.set_joint("RKneePitch",   1.2)
        self.robot.set_joint("LAnklePitch", -0.5)
        self.robot.set_joint("RAnklePitch", -0.5)
        self.robot.set_joint("LShoulderPitch",  1.9)
        self.robot.set_joint("RShoulderPitch",  1.9)

        for _ in range(frames(0.8)):
            yield

        # 4. Hold simulated grasp
        print("[PlanExecutor] pick_object: grasping…")

        for _ in range(frames(0.5)):
            yield

        # 5. Stand back up, carry position
        self.robot.set_joint("LHipPitch",   -0.45)
        self.robot.set_joint("RHipPitch",   -0.45)
        self.robot.set_joint("LKneePitch",   0.87)
        self.robot.set_joint("RKneePitch",   0.87)
        self.robot.set_joint("LAnklePitch", -0.41)
        self.robot.set_joint("RAnklePitch", -0.41)
        self.robot.set_joint("LShoulderPitch",  1.2)
        self.robot.set_joint("RShoulderPitch",  1.2)
        self.robot.set_joint("LElbowRoll",     -0.8)
        self.robot.set_joint("RElbowRoll",      0.8)
        self.robot.set_head_pitch(0.0)

        for _ in range(frames(1.0)):
            yield

        print("[PlanExecutor] pick_object: complete ✓")
        return "ok"

    def _act_place_object(self, params: dict) -> Generator:
        print("[PlanExecutor] place_object")

        f = self.robot.timestep_s

        def frames(secs):
            return max(1, int(secs / f))

        self.robot.set_joint("LShoulderPitch",  1.7)
        self.robot.set_joint("RShoulderPitch",  1.7)
        self.robot.set_joint("LElbowRoll",     -0.3)
        self.robot.set_joint("RElbowRoll",      0.3)

        for _ in range(frames(0.6)):
            yield

        self.robot.set_joint("LShoulderPitch",  1.4)
        self.robot.set_joint("RShoulderPitch",  1.4)
        self.robot.set_joint("LShoulderRoll",   0.3)
        self.robot.set_joint("RShoulderRoll",  -0.3)
        self.robot.set_joint("LElbowRoll",     -0.5)
        self.robot.set_joint("RElbowRoll",      0.5)

        for _ in range(frames(0.5)):
            yield

        print("[PlanExecutor] place_object: done ✓")
        return "ok"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_object(self, label: str) -> Optional[dict]:
        """
        Query SceneBus for a detected object matching `label` and normalize both
        supported scene schemas:
          - simple schema: state["objects"]
          - compatibility schema: state["scene"]["objects"]
        Returns a dict with cx_norm, cy_norm, h_norm, and distance_m.
        """
        if self.bus is None:
            return None

        scene = self.bus.get_latest("scene_state")
        if not scene:
            return None

        if isinstance(scene, (tuple, list)) and scene:
            scene = scene[0]
        if not isinstance(scene, dict):
            return None

        if isinstance(scene.get("objects"), list):
            raw_dets = scene.get("objects", [])
        elif isinstance(scene.get("detections"), list):
            raw_dets = scene.get("detections", [])
        else:
            raw_dets = scene.get("scene", {}).get("objects", [])

        if not raw_dets:
            return None

        cam = scene.get("camera", {}) if isinstance(scene, dict) else {}
        res = cam.get("resolution", {}) if isinstance(cam, dict) else {}
        fw = res.get("width") or scene.get("frame_width") or 640
        fh = res.get("height") or scene.get("frame_height") or 480

        target = label.lower().strip()
        matches = []

        for d in raw_dets:
            lbl = str(d.get("label", "")).lower().strip()
            if not lbl:
                continue
            if lbl != target and target not in lbl and lbl not in target:
                continue

            bb = d.get("bounding_box", {}) if isinstance(d.get("bounding_box"), dict) else {}
            x = bb.get("x", d.get("x", 0))
            y = bb.get("y", d.get("y", 0))
            w = bb.get("width", d.get("w", d.get("width", 0)))
            h = bb.get("height", d.get("h", d.get("height", 0)))

            sp = d.get("screen_position", {}) if isinstance(d.get("screen_position"), dict) else {}
            cx_norm = d.get("cx_norm", sp.get("x_norm"))
            cy_norm = d.get("cy_norm", sp.get("y_norm"))
            if cx_norm is None:
                cx_norm = (x + w / 2) / fw if fw else 0.5
            if cy_norm is None:
                cy_norm = (y + h / 2) / fh if fh else 0.5

            h_norm = d.get("h_norm", d.get("height_frac"))
            w_norm = d.get("w_norm", d.get("width_frac"))
            if h_norm is None:
                h_norm = h / fh if fh else 0.0
            if w_norm is None:
                w_norm = w / fw if fw else 0.0

            distance_m = (
                d.get("depth_distance_m")
                if d.get("depth_distance_m") is not None
                else d.get("distance_m")
                if d.get("distance_m") is not None
                else d.get("estimated_distance_m")
            )

            matches.append({
                "label": lbl,
                "cx_norm": float(cx_norm),
                "cy_norm": float(cy_norm),
                "w_norm": float(w_norm),
                "h_norm": float(h_norm),
                "confidence": float(d.get("confidence", 0.0)),
                "bearing_deg": d.get("horizontal_angle_deg"),
                "distance_m": distance_m,
                "raw": d,
            })

        if not matches:
            return None

        return max(matches, key=lambda d: d.get("confidence", 0.0))


def _hash_plan(steps: list) -> str:
    raw = json.dumps(steps, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]