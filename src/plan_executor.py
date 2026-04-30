"""
PlanExecutor — NAO robot execution layer.

Sits between the LLM planner and raw Webots motor control.
Executes plans step-by-step, non-blocking, inside the Webots simulation loop.

Design:
  - Each action is a generator that yields control each timestep
  - PlanExecutor drives the active action one tick at a time
  - Actions can query the SceneBus for live vision feedback
  - Plan caching prevents re-executing the same plan
  - A simple state machine tracks: IDLE → RUNNING → DONE / FAILED
"""

import time
import hashlib
import json
import logging
from enum import Enum, auto
from typing import Generator, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  State machine states
# ─────────────────────────────────────────────

class ExecutorState(Enum):
    IDLE    = auto()   # No active plan
    RUNNING = auto()   # Executing a step
    PAUSED  = auto()   # Waiting (e.g. for sensor confirmation)
    DONE    = auto()   # Plan complete
    FAILED  = auto()   # Unrecoverable error


# ─────────────────────────────────────────────
#  PlanExecutor
# ─────────────────────────────────────────────

class PlanExecutor:
    """
    Drives plan execution one simulation tick at a time.

    Usage inside Webots loop:
        executor = PlanExecutor(robot_interface, scene_bus)

        while robot.step(timestep) != -1:
            # update perception / planning ...
            if new_plan:
                executor.load_plan(new_plan)
            executor.tick()
    """

    def __init__(self, robot_interface, scene_bus=None):
        """
        Args:
            robot_interface: NaoInterface instance (wraps Webots motor APIs)
            scene_bus:       SceneBus instance for live vision queries (optional)
        """
        self.robot   = robot_interface
        self.bus     = scene_bus
        self.state   = ExecutorState.IDLE

        self._plan:        list  = []          # List of step dicts from LLM
        self._step_index:  int   = 0           # Which step we're on
        self._active_gen:  Optional[Generator] = None   # Running action generator
        self._plan_hash:   str   = ""          # Hash of current plan (for caching)
        self._step_result: str   = "ok"        # "ok" | "failed"

        # Registry: action name → handler method
        self._handlers = {
            # ── Primitive (motor-level) ──────────────────
            "turn_left":       self._act_turn,
            "turn_right":      self._act_turn,
            "adjust_orientation":  self._act_turn, 
            "move_forward":    self._act_move_forward,
            "move_backward":   self._act_move_backward,
            "stop":            self._act_stop,
            "set_head_yaw":    self._act_set_head_yaw,
            "set_head_pitch":  self._act_set_head_pitch,
            "wave":            self._act_wave,
            # ── Semantic (vision-guided) ─────────────────
            "move_toward_object":  self._act_move_toward_object,
            "center_on_object":    self._act_center_on_object,
            "look_for_object":     self._act_look_for_object,
            "pick_object":         self._act_pick_object,
            "place_object":        self._act_place_object,
        }

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def load_plan(self, plan_json: dict) -> bool:
        """
        Load a new plan.  Returns False if plan is identical to current one
        (caching) or malformed.

        Args:
            plan_json: Full LLM output dict, e.g.:
                {
                  "plan": [
                    {"step": 1, "action": "turn_left",  "parameters": {"degrees": 30}},
                    {"step": 2, "action": "move_forward","parameters": {"meters": 1.0}},
                  ]
                }
        """
        steps = plan_json.get("plan", [])
        if not steps:
            logger.warning("PlanExecutor: empty or missing 'plan' key — skipping")
            return False

        plan_hash = _hash_plan(steps)
        # Skip reload only if plan is identical AND we are actively running it
        if plan_hash == self._plan_hash and self.state == ExecutorState.RUNNING:
            logger.debug("PlanExecutor: plan unchanged and still running — not reloading")
            return False

        logger.info(f"PlanExecutor: loading plan with {len(steps)} step(s)")
        self._plan       = steps
        self._step_index = 0
        self._plan_hash  = plan_hash
        self._active_gen = None
        self.state       = ExecutorState.RUNNING
        self._advance()   # prime first action
        return True

    def tick(self):
        """
        Call once per Webots simulation step.
        Advances the active action by one frame.
        """
        if self.state != ExecutorState.RUNNING:
            return

        if self._active_gen is None:
            return

        try:
            next(self._active_gen)              # advance action one frame
        except StopIteration as e:
            result = e.value if e.value else "ok"
            logger.debug(f"PlanExecutor: step {self._step_index} finished → {result}")
            self._on_step_done(result)
        except Exception as exc:
            logger.error(f"PlanExecutor: step {self._step_index} raised {exc}")
            self._on_step_done("failed")

    def reset(self):
        """Abandon current plan and return to IDLE."""
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

    # ──────────────────────────────────────────
    #  Internal step-machine helpers
    # ──────────────────────────────────────────

    def _advance(self):
        """Start the next step, or mark plan as DONE."""
        if self._step_index >= len(self._plan):
            logger.info("PlanExecutor: plan complete ✓")
            self.state = ExecutorState.DONE
            self.robot.stop_walk()
            return

        step   = self._plan[self._step_index]
        action = step.get("action", "").lower()
        params = step.get("parameters", {})

        logger.info(f"PlanExecutor: step {self._step_index + 1}/{len(self._plan)} → {action} {params}")

        handler = self._handlers.get(action)
        if handler is None:
            logger.warning(f"PlanExecutor: unknown action '{action}' — skipping")
            self._on_step_done("skipped")
            return

        self._active_gen = handler(params)

    def _on_step_done(self, result: str):
        self._active_gen = None
        if result == "failed":
            self.state = ExecutorState.FAILED
            logger.error(f"PlanExecutor: FAILED at step {self._step_index}")
            return
        self._step_index += 1
        self._advance()

    # ──────────────────────────────────────────
    #  ── PRIMITIVE ACTIONS ────────────────────
    #  Each is a generator: yields each frame, returns "ok"/"failed"
    # ──────────────────────────────────────────

    def _act_turn(self, params: dict) -> Generator:
        """
        Turn in place by `degrees` (positive = left, negative = right).
        For turn_right the sign is flipped automatically by checking the
        action name stored in the current step.
        """
        degrees = float(params.get("degrees", 30))
        action  = self.current_step.get("action", "turn_left")
        if action == "turn_right":
            degrees = -abs(degrees)
        else:
            degrees = abs(degrees)

        duration_s  = abs(degrees) / 45.0      # ~45 °/s turn speed
        frames      = max(1, int(duration_s / self.robot.timestep_s))

        self.robot.start_turn(degrees)
        for _ in range(frames):
            yield
        self.robot.stop_walk()
        return "ok"

    def _act_move_forward(self, params: dict) -> Generator:
        meters = float(params.get("meters", params.get("distance_m", 0.5)))
        speed      = 0.12                       # conservative NAO gait estimate
        frames     = max(1, int(meters / speed / self.robot.timestep_s))

        self.robot.start_walk(vx=speed, vy=0, omega=0)
        for _ in range(frames):
            yield
        self.robot.stop_walk()
        return "ok"

    def _act_move_backward(self, params: dict) -> Generator:
        meters = float(params.get("meters", params.get("distance_m", 0.3)))
        speed  = 0.10
        frames = max(1, int(meters / speed / self.robot.timestep_s))

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
        angle  = float(params.get("angle", 0.0))      # radians
        frames = int(1.0 / self.robot.timestep_s)      # ~1 s motion

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
        """Simple wave gesture using right shoulder/elbow joints."""
        import math
        frames_per_cycle = int(0.5 / self.robot.timestep_s)   # 0.5 s per half-wave
        cycles = int(params.get("cycles", 2))

        for c in range(cycles * 2):
            angle = math.pi / 4 if c % 2 == 0 else 0.0
            self.robot.set_joint("RShoulderPitch", -angle)
            self.robot.set_joint("RShoulderRoll",   angle)
            self.robot.set_joint("RElbowRoll",      angle * 1.5)
            for _ in range(frames_per_cycle):
                yield

        # Return arm to rest
        self.robot.set_joint("RShoulderPitch", 1.4)
        self.robot.set_joint("RShoulderRoll", -0.3)
        self.robot.set_joint("RElbowRoll",     0.5)
        for _ in range(frames_per_cycle):
            yield
        return "ok"

    # ──────────────────────────────────────────
    #  ── SEMANTIC ACTIONS ─────────────────────
    #  Vision-guided; use SceneBus for feedback
    # ──────────────────────────────────────────

    def _act_center_on_object(self, params: dict) -> Generator:
        """
        Rotate head (and body if needed) until target object is centred
        in the camera frame.

        params:
            label       – YOLO class name, e.g. "cup"
            tolerance   – fraction of frame width to accept as "centred" (default 0.1)
            timeout_s   – give up after this many seconds (default 5)
        """
        label      = params.get("label", "")
        tolerance  = float(params.get("tolerance", 0.10))    # 10 % of frame
        timeout_s  = float(params.get("timeout_s", 5.0))
        deadline   = time.time() + timeout_s

        logger.info(f"center_on_object: looking for '{label}'")

        while time.time() < deadline:
            obj = self._find_object(label)
            if obj is None:
                logger.debug(f"center_on_object: '{label}' not visible — waiting")
                yield
                continue

            cx_norm = obj["cx_norm"]    # 0 = left edge, 1 = right edge, 0.5 = centre
            error   = cx_norm - 0.5     # negative → object left of centre

            if abs(error) < tolerance:
                logger.info(f"center_on_object: '{label}' centred ✓")
                return "ok"

            # Proportional head yaw correction
            correction = -error * 0.8   # radians, scaled
            self.robot.adjust_head_yaw(correction)
            yield

        logger.warning(f"center_on_object: timeout waiting for '{label}'")
        return "failed"

    def _act_move_toward_object(self, params: dict) -> Generator:
        """
        Walk toward an object until its bounding-box height reaches
        a threshold fraction of the frame (proxy for distance).

        params:
            label           – YOLO class name
            target_height   – bbox height as fraction of frame when "close enough" (default 0.35)
            timeout_s       – (default 10)
        """
        label         = params.get("label", "")
        target_height = float(params.get("target_height", 0.35))
        timeout_s     = float(params.get("timeout_s", 10.0))
        deadline      = time.time() + timeout_s
        tolerance     = 0.10   # centering tolerance

        logger.info(f"move_toward_object: heading for '{label}'")

        while time.time() < deadline:
            obj = self._find_object(label)

            if obj is None:
                # Object lost — stop and wait one tick
                self.robot.stop_walk()
                yield
                continue

            cx_norm   = obj["cx_norm"]
            box_h     = obj["h_norm"]     # 0→1 relative frame height
            error     = cx_norm - 0.5

            # Close enough?
            if box_h >= target_height:
                self.robot.stop_walk()
                logger.info(f"move_toward_object: reached '{label}' ✓")
                return "ok"

            # Steer while walking
            omega = -error * 1.2    # yaw correction (rad/s)
            self.robot.start_walk(vx=0.25, vy=0, omega=omega)
            yield

        self.robot.stop_walk()
        logger.warning(f"move_toward_object: timeout for '{label}'")
        return "failed"

    def _act_look_for_object(self, params: dict) -> Generator:
        """
        Rotate in place scanning for a labelled object.

        params:
            label       – YOLO class name
            timeout_s   – (default 8)
        """
        label     = params.get("label", "")
        timeout_s = float(params.get("timeout_s", 8.0))
        deadline  = time.time() + timeout_s

        logger.info(f"look_for_object: scanning for '{label}'")
        self.robot.start_turn(degrees=360)   # slow continuous rotation

        while time.time() < deadline:
            obj = self._find_object(label)
            if obj is not None:
                self.robot.stop_walk()
                logger.info(f"look_for_object: found '{label}' ✓")
                return "ok"
            yield

        self.robot.stop_walk()
        logger.warning(f"look_for_object: '{label}' not found within timeout")
        return "failed"

    def _act_pick_object(self, params: dict) -> Generator:
        """
        Simplified pick: lower torso + close right hand.
        Assumes robot is already positioned close to the object.
        Extend this with IK or Webots Motion objects for real grasping.
        """
        import math
        label = params.get("label", "")
        logger.info(f"pick_object: attempting to grasp '{label}'")

        settle_frames = int(0.5 / self.robot.timestep_s)

        # Lean forward slightly
        self.robot.set_joint("LShoulderPitch",  1.8)
        self.robot.set_joint("RShoulderPitch",  1.8)
        self.robot.set_joint("LElbowRoll",     -1.0)
        self.robot.set_joint("RElbowRoll",      1.0)
        for _ in range(settle_frames * 2):
            yield

        # Close hands
        self.robot.set_joint("LHand", 0.0)
        self.robot.set_joint("RHand", 0.0)
        for _ in range(settle_frames):
            yield

        logger.info("pick_object: grasp complete (simulated) ✓")
        return "ok"

    def _act_place_object(self, params: dict) -> Generator:
        """Open hand to release object."""
        logger.info("place_object: releasing object")
        settle_frames = int(0.5 / self.robot.timestep_s)

        self.robot.set_joint("LHand", 1.0)
        self.robot.set_joint("RHand", 1.0)
        for _ in range(settle_frames):
            yield

        # Return arms to rest
        self.robot.set_joint("LShoulderPitch", 1.4)
        self.robot.set_joint("RShoulderPitch", 1.4)
        for _ in range(settle_frames):
            yield

        logger.info("place_object: done ✓")
        return "ok"

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    def _find_object(self, label: str) -> Optional[dict]:
        """
        Query SceneBus for the most-confident detection matching `label`.

        Returns a dict with:
            cx_norm  – horizontal centre 0..1
            cy_norm  – vertical centre 0..1
            w_norm   – width  0..1
            h_norm   – height 0..1
            confidence
        or None if not found.
        """
        if self.bus is None:
            return None

        scene = self.bus.get_latest("scene_state")
        if not scene:
            return None

        detections = scene.get("detections", [])
        matches = [d for d in detections if d.get("label", "").lower() == label.lower()]
        if not matches:
            return None

        # Pick highest-confidence match
        best = max(matches, key=lambda d: d.get("confidence", 0))

        # Normalise bounding box to 0..1 (expects keys x, y, w, h in pixels
        # and frame_width / frame_height in scene state, or pre-normalised)
        fw = scene.get("frame_width",  640)
        fh = scene.get("frame_height", 480)
        x, y, w, h = best["x"], best["y"], best["w"], best["h"]

        return {
            "cx_norm":    (x + w / 2) / fw,
            "cy_norm":    (y + h / 2) / fh,
            "w_norm":     w / fw,
            "h_norm":     h / fh,
            "confidence": best.get("confidence", 0),
        }


# ─────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────

def _hash_plan(steps: list) -> str:
    raw = json.dumps(steps, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]