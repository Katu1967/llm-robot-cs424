"""
NaoInterface — thin wrapper around Webots NAO motor APIs.

Provides a clean, unit-testable surface for PlanExecutor.
All raw Webots calls (getDevice, setVelocity, etc.) live here.

NAO joint reference
-------------------
Head:    HeadYaw, HeadPitch
Arms:    LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, LHand
         R* variants for right arm
Legs:    LHipYawPitch, LHipRoll, LHipPitch, LKneePitch, LAnklePitch, LAnkleRoll
         R* variants
Walk:    controlled via Webots Motion files (shipped with Webots NAO model)
"""

import os
import math
import logging

logger = logging.getLogger(__name__)

# All 25 NAO joints (Webots device names)
NAO_JOINTS = [
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
]

# Safe rest angles (radians) — robot stands upright, arms at sides
REST_POSE = {
    "HeadYaw":         0.0,
    "HeadPitch":       0.0,
    "LShoulderPitch":  1.4,   "RShoulderPitch":  1.4,
    "LShoulderRoll":   0.3,   "RShoulderRoll":  -0.3,
    "LElbowYaw":      -1.2,   "RElbowYaw":       1.2,
    "LElbowRoll":     -0.5,   "RElbowRoll":      0.5,
    "LWristYaw":       0.0,   "RWristYaw":       0.0,
    "LHand":           0.5,   "RHand":           0.5,
    "LHipYawPitch":    0.0,   "RHipYawPitch":    0.0,
    "LHipRoll":        0.0,   "RHipRoll":        0.0,
    "LHipPitch":      -0.45,  "RHipPitch":      -0.45,
    "LKneePitch":      0.87,  "RKneePitch":      0.87,
    "LAnklePitch":    -0.41,  "RAnklePitch":    -0.41,
    "LAnkleRoll":      0.0,   "RAnkleRoll":      0.0,
}

# Motion file names (without .motion extension) — shipped with the Webots NAO model
# or copied into the repo under worlds/candy_world/motions/.
MOTION_FILES = {
    "forward":    "Forwards",
    "backward":   "Backwards",
    "turn_left":  "TurnLeft60",
    "turn_right": "TurnRight60",
}


class NaoInterface:
    """
    Wraps Webots Robot object and exposes high-level motor commands.

    Walk is implemented via Webots Motion files (the real NAO gait).
    Head and arm joints are controlled directly via position motors.

    Args:
        robot:      Webots Robot instance (from `from controller import Robot`)
        timestep:   Simulation timestep in ms (typically 32 or 64)
    """

    def __init__(self, robot, timestep: int = 32):
        self.robot      = robot
        self.timestep   = timestep
        self.timestep_s = timestep / 1000.0

        # ── Head position tracking ───────────────────
        self._head_yaw   = 0.0
        self._head_pitch = 0.0

        # ── Initialise position motors (head + arms) ─
        self._motors: dict = {}
        for name in NAO_JOINTS:
            try:
                motor = robot.getDevice(name)
                if motor is not None:
                    self._motors[name] = motor
                    if name in REST_POSE:
                        try:
                            motor.setPosition(REST_POSE[name])
                            motor.setVelocity(motor.getMaxVelocity())
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning(f"NaoInterface: could not init joint '{name}': {exc}")

        # ── Load Motion files ────────────────────────
        self._motions: dict = {}
        self._active_motion = None
        self._load_motions()

        # Put the robot in a safe rest pose immediately.
        self.go_to_rest()

        logger.info(
            f"NaoInterface: ready — {len(self._motors)} joints, "
            f"{len(self._motions)} motion files loaded"
        )

    # ──────────────────────────────────────────
    #  Motion file loading
    # ──────────────────────────────────────────

    def _load_motions(self):
        """
        Locate and load Webots NAO Motion files.

        Searches in order:
          1. WEBOTS_HOME env var  → .../projects/robots/softbank/nao/motions/
          2. Common Windows path  → C:/Program Files/Webots/...
          3. Common Linux path    → /usr/local/webots/...
          4. src/motions/ next to this file (if you copied motions locally)
        """
        from controller import Motion

        candidate_dirs = []

        webots_home = os.environ.get("WEBOTS_HOME", "")
        if webots_home:
            candidate_dirs.append(
                os.path.join(webots_home, "projects", "robots", "softbank", "nao", "motions")
            )

        candidate_dirs += [
            r"C:\Program Files\Webots\projects\robots\softbank\nao\motions",
            r"C:\Program Files (x86)\Webots\projects\robots\softbank\nao\motions",
            "/usr/local/webots/projects/robots/softbank/nao/motions",
            "/usr/share/webots/projects/robots/softbank/nao/motions",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions"),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "worlds",
                "candy_world",
                "motions",
            ),
        ]

        motions_dir = None
        for d in candidate_dirs:
            normalized = d.replace("\\", "/")
            if os.path.isdir(normalized) or os.path.isdir(d):
                motions_dir = d
                break

        if motions_dir is None:
            logger.error(
                "NaoInterface: could not find NAO motions directory. "
                "Set WEBOTS_HOME env var or copy motion files to src/motions/. "
                "Walking will NOT work."
            )
            return

        logger.info(f"NaoInterface: loading motions from {motions_dir}")

        for key, filename in MOTION_FILES.items():
            path = os.path.join(motions_dir, filename + ".motion")
            if os.path.isfile(path):
                try:
                    m = Motion(path)
                    self._motions[key] = m
                    logger.info(f"NaoInterface: loaded '{key}' ← {filename}.motion")
                except Exception as exc:
                    logger.warning(f"NaoInterface: failed to load '{filename}': {exc}")
            else:
                logger.warning(f"NaoInterface: motion file not found: {path}")

    # ──────────────────────────────────────────
    #  Walk control (Motion-based)
    # ──────────────────────────────────────────

    def start_walk(self, vx: float = 0.0, vy: float = 0.0, omega: float = 0.0):
        """
        Start walking. Selects the appropriate Motion file based on
        the dominant velocity component.

        Args:
            vx:    forward / backward speed  (positive = forward)
            vy:    lateral speed             (unused — NAO has no strafe motion)
            omega: yaw rate                  (positive = left turn)
        """
        if abs(omega) > 0.1:
            key = "turn_left" if omega > 0 else "turn_right"
        elif vx >= 0:
            key = "forward"
        else:
            key = "backward"

        self._play_motion(key)

    def start_turn(self, degrees: float):
        """
        Start turning in place.
        Positive degrees = left turn, negative = right turn.
        The executor controls timing/duration.
        """
        key = "turn_left" if degrees > 0 else "turn_right"
        self._play_motion(key)

    def stop_walk(self):
        """Stop locomotion and return to stand pose."""
        if self._active_motion is not None:
            self._active_motion.stop()
            self._active_motion = None
        self.go_to_rest()

    def _play_motion(self, key: str):
        """
        Stop the current motion and start the named one.
        Locomotion motions loop.
        """
        if key == "stand":
            self.stop_walk()
            return

        motion = self._motions.get(key)
        if motion is None:
            logger.warning(f"NaoInterface: motion '{key}' not loaded — skipping")
            return

        # Stop previous motion if switching to a different one
        if self._active_motion is not None and self._active_motion is not motion:
            self._active_motion.stop()

        is_locomotion = key in ("forward", "backward", "turn_left", "turn_right")
        motion.setLoop(is_locomotion)
        motion.play()
        self._active_motion = motion if is_locomotion else None

    # ──────────────────────────────────────────
    #  Head control
    # ──────────────────────────────────────────

    def set_head_yaw(self, angle: float):
        """Set absolute head yaw (radians). Clamped to NAO limits ±2.0857."""
        angle = max(-2.08, min(2.08, angle))
        self._head_yaw = angle
        self._set("HeadYaw", angle)

    def set_head_pitch(self, angle: float):
        """Set absolute head pitch (radians). Clamped ±0.5."""
        angle = max(-0.5, min(0.5, angle))
        self._head_pitch = angle
        self._set("HeadPitch", angle)

    def adjust_head_yaw(self, delta: float):
        """Rotate head by delta radians from its current yaw."""
        self.set_head_yaw(self._head_yaw + delta)

    def look_at_normalised(self, cx_norm: float, cy_norm: float):
        """
        Point head toward a normalised screen coordinate.
        cx_norm=0 → left edge, 1 → right edge.
        cy_norm=0 → top,       1 → bottom.
        """
        yaw_error   = (cx_norm - 0.5) * math.pi * 0.5
        pitch_error = (cy_norm - 0.5) * math.pi * 0.3
        self.set_head_yaw(-yaw_error)
        self.set_head_pitch(pitch_error)

    # ──────────────────────────────────────────
    #  Joint control
    # ──────────────────────────────────────────

    def set_joint(self, name: str, angle: float):
        """Set a named joint to a target angle (radians)."""
        self._set(name, angle)

    def go_to_rest(self):
        """Return head and arm joints to the rest pose."""
        for name, angle in REST_POSE.items():
            self._set(name, angle)

    # ──────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────

    def _set(self, name: str, angle: float):
        motor = self._motors.get(name)
        if motor:
            try:
                motor.setPosition(angle)
            except Exception as exc:
                logger.warning(f"NaoInterface: setPosition failed for {name}: {exc}")
        else:
            logger.debug(f"NaoInterface: joint '{name}' not found — skipping")