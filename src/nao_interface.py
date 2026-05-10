"""
NaoInterface — NAO control wrapper for Webots.

Locomotion uses the built-in Webots NAO Motion files when available:
  - Forwards.motion
  - Backwards.motion
  - TurnLeft60.motion
  - TurnRight60.motion

Head and arm joints are controlled directly with setPosition().
Also exposes get_gps_position() for debugging whether the robot is physically moving.
"""

import os
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


NAO_JOINTS = [
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
]


REST_POSE = {
    "HeadYaw": 0.0,          "HeadPitch": 0.0,

    "LShoulderPitch": 1.4,   "RShoulderPitch": 1.4,
    "LShoulderRoll": 0.3,    "RShoulderRoll": -0.3,
    "LElbowYaw": -1.2,       "RElbowYaw": 1.2,
    "LElbowRoll": -0.5,      "RElbowRoll": 0.5,
    "LWristYaw": 0.0,        "RWristYaw": 0.0,
    "LHand": 0.5,            "RHand": 0.5,

    "LHipYawPitch": 0.0,     "RHipYawPitch": 0.0,
    "LHipRoll": 0.0,         "RHipRoll": 0.0,
    "LHipPitch": -0.45,      "RHipPitch": -0.45,
    "LKneePitch": 0.87,      "RKneePitch": 0.87,
    "LAnklePitch": -0.41,    "RAnklePitch": -0.41,
    "LAnkleRoll": 0.0,       "RAnkleRoll": 0.0,
}


class NaoInterface:
    def __init__(self, robot, timestep: int = 32):
        self.robot = robot
        self.timestep = timestep
        self.timestep_s = timestep / 1000.0

        self._head_yaw = 0.0
        self._head_pitch = 0.0
        # Skip tiny setPosition deltas to reduce head jitter (rad).
        self._head_cmd_min = float(os.getenv("NAO_HEAD_CMD_MIN_RAD", "0.012"))
        # HeadPitch joint limits from Webots (avoid "too big requested position" warnings).
        self._head_pitch_min = -0.67
        self._head_pitch_max = 0.51

        self._motors = {}
        for name in NAO_JOINTS:
            try:
                motor = robot.getDevice(name)
                if motor is not None:
                    self._motors[name] = motor
                    try:
                        motor.setVelocity(motor.getMaxVelocity())
                    except Exception:
                        pass
                    if name == "HeadPitch":
                        try:
                            self._head_pitch_min = float(motor.getMinPosition())
                            self._head_pitch_max = float(motor.getMaxPosition())
                        except Exception:
                            pass
                    if name in REST_POSE:
                        motor.setPosition(REST_POSE[name])
            except Exception as exc:
                logger.warning(f"NaoInterface: could not init '{name}': {exc}")

        self._gps = None
        try:
            self._gps = robot.getDevice("gps")
            if self._gps is not None:
                self._gps.enable(timestep)
        except Exception as exc:
            print(f"[NaoInterface] GPS unavailable: {exc}")

        self._motions = {}
        self._active_motion_key = None
        self._active_motion = None
        self._walk_fast_vx_thr = float(os.environ.get("NAO_WALK_FAST_VX_THRESHOLD", "0.08"))
        self._try_load_motions()

        print(
            f"[NaoInterface] ready — {len(self._motors)} joints, "
            f"{len(self._motions)} motion files "
            f"(HeadPitch clamp [{self._head_pitch_min:.3f}, {self._head_pitch_max:.3f}] rad)"
        )

    # ------------------------------------------------------------------
    # Motion file loading
    # ------------------------------------------------------------------

    def _try_load_motions(self):
        try:
            from controller import Motion
        except ImportError:
            print("[NaoInterface] Could not import Webots Motion class")
            return

        candidate_dirs = []

        webots_home = os.environ.get("WEBOTS_HOME", "")
        if webots_home:
            candidate_dirs.append(
                os.path.join(
                    webots_home,
                    "Contents",
                    "projects",
                    "robots",
                    "softbank",
                    "nao",
                    "motions",
                )
            )
            candidate_dirs.append(
                os.path.join(
                    webots_home,
                    "projects",
                    "robots",
                    "softbank",
                    "nao",
                    "motions",
                )
            )

        candidate_dirs += [
            "/Applications/Webots.app/Contents/projects/robots/softbank/nao/motions",
            "/usr/local/webots/projects/robots/softbank/nao/motions",
            r"C:\Program Files\Webots\projects\robots\softbank\nao\motions",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions"),
        ]

        motions_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)

        if not motions_dir:
            print("[NaoInterface] ERROR: NAO motion directory not found")
            print("[NaoInterface] Checked:")
            for d in candidate_dirs:
                print(f"  - {d}")
            return

        print(f"[NaoInterface] Using motion directory: {motions_dir}")

        motion_files = {
            "forward": "Forwards.motion",
            "forward_fast": "Forwards50.motion",
            "backward": "Backwards.motion",
            "turn_left": "TurnLeft60.motion",
            "turn_right": "TurnRight60.motion",
            "stand": "Stand.motion",
        }

        for key, filename in motion_files.items():
            path = os.path.join(motions_dir, filename)
            if not os.path.isfile(path):
                print(f"[NaoInterface] WARNING: missing motion file: {path}")
                continue

            try:
                self._motions[key] = Motion(path)
            except Exception as exc:
                print(f"[NaoInterface] WARNING: failed to load {filename}: {exc}")

        print(f"[NaoInterface] Motion files loaded: {sorted(self._motions.keys())}")

        ff = "forward_fast" in self._motions
        std = os.environ.get("NAO_FORWARD_PROFILE", "standard").lower() != "fast"
        print(
            f"[NaoInterface] Forward walk: "
            f"{'Forwards50 (fast)' if ff and not std else 'Forwards (default)'} "
            f"(set NAO_FORWARD_PROFILE=fast for Forwards50.motion)"
        )

    # ------------------------------------------------------------------
    # GPS debug
    # ------------------------------------------------------------------

    def get_gps_position(self):
        try:
            if self._gps is None:
                return None
            values = self._gps.getValues()
            return tuple(round(float(v), 4) for v in values)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Locomotion
    # ------------------------------------------------------------------

    def _play_motion(self, key: str, loop: bool = True):
        motion = self._motions.get(key)

        if motion is None:
            print(f"[NaoInterface] missing motion '{key}'")
            return False

        if self._active_motion_key == key:
            return True

        if self._active_motion is not None:
            try:
                self._active_motion.stop()
            except Exception:
                pass

        self._active_motion_key = key
        self._active_motion = motion

        try:
            motion.setLoop(loop)
        except Exception:
            pass

        try:
            motion.play()
            print(f"[NaoInterface] motion -> {key}")
            return True
        except Exception as exc:
            print(f"[NaoInterface] failed to play motion '{key}': {exc}")
            return False

    def start_walk(self, vx: float = 0.0, vy: float = 0.0, omega: float = 0.0):
        if abs(omega) > 0.30:
            if omega > 0:
                self._play_motion("turn_left", loop=True)
            else:
                self._play_motion("turn_right", loop=True)
            return

        if vx < -0.01:
            self._play_motion("backward", loop=True)
            return

        use_fast = (
            os.environ.get("NAO_FORWARD_PROFILE", "standard").lower() == "fast"
            and float(vx) >= self._walk_fast_vx_thr
            and "forward_fast" in self._motions
        )
        if use_fast:
            self._play_motion("forward_fast", loop=True)
        elif "forward" in self._motions:
            self._play_motion("forward", loop=True)
        elif "forward_fast" in self._motions:
            self._play_motion("forward_fast", loop=True)

    def start_turn(self, degrees: float):
        if degrees >= 0:
            self._play_motion("turn_left", loop=True)
        else:
            self._play_motion("turn_right", loop=True)

    def stop_locomotion_only(self):
        """Stop walk/turn motions only — no Stand / go_to_rest (keeps head pose)."""
        if self._active_motion is not None:
            try:
                self._active_motion.stop()
            except Exception:
                pass
        self._active_motion = None
        self._active_motion_key = None

    def stop_walk(self):
        self.stop_locomotion_only()
        if "stand" in self._motions:
            try:
                self._motions["stand"].setLoop(False)
                self._motions["stand"].play()
            except Exception:
                self.go_to_rest()
        else:
            self.go_to_rest()

        print("[NaoInterface] stopped")

    # ------------------------------------------------------------------
    # Head control
    # ------------------------------------------------------------------

    def reset_head_neutral(self):
        """Head straight ahead / level — use during search before the target is confirmed."""
        self.set_head_yaw(0.0)
        self.set_head_pitch(0.0)

    def set_head_yaw(self, angle: float):
        angle = max(-2.08, min(2.08, float(angle)))
        if abs(angle - self._head_yaw) < self._head_cmd_min:
            self._head_yaw = angle
            return
        self._head_yaw = angle
        self._set("HeadYaw", angle)

    def set_head_pitch(self, angle: float):
        mn, mx = self._head_pitch_min, self._head_pitch_max
        angle = max(mn, min(mx, float(angle)))
        if abs(angle - self._head_pitch) < self._head_cmd_min:
            self._head_pitch = angle
            return
        self._head_pitch = angle
        self._set("HeadPitch", angle)

    def adjust_head_yaw(self, delta: float):
        self.set_head_yaw(self._head_yaw + float(delta))

    def adjust_head_pitch(self, delta: float):
        """Positive delta tilts head down (camera toward floor); negative looks up."""
        self.set_head_pitch(self._head_pitch + float(delta))

    def get_head_pitch(self) -> float:
        return self._head_pitch

    def get_head_yaw(self) -> float:
        return self._head_yaw

    def look_pitch_from_cy_norm(
        self,
        cy_norm: float,
        alpha: Optional[float] = None,
        pitch_gain: Optional[float] = None,
        cy_deadband: Optional[float] = None,
    ) -> None:
        """
        Adjust head **pitch only** from vertical image position (0=top, 1=bottom).
        Yaw is unchanged. When ``cy_norm`` is within ``cy_deadband`` of image center (0.5),
        target pitch is level (0 rad). Pitch is clamped to HeadPitch joint limits.
        """
        cy_norm = max(0.0, min(1.0, float(cy_norm)))
        db = 0.04 if cy_deadband is None else max(0.0, float(cy_deadband))
        pg = 0.3 if pitch_gain is None else max(0.05, float(pitch_gain))
        if abs(cy_norm - 0.5) <= db:
            desired_pitch = 0.0
        else:
            desired_pitch = (cy_norm - 0.5) * math.pi * pg

        mn, mx = self._head_pitch_min, self._head_pitch_max
        desired_pitch = max(mn, min(mx, desired_pitch))

        a = 0.08 if alpha is None else max(0.01, min(1.0, float(alpha)))
        new_pitch = self._head_pitch + a * (desired_pitch - self._head_pitch)
        new_pitch = max(mn, min(mx, new_pitch))
        self.set_head_pitch(new_pitch)

    def look_at_normalised(
        self,
        cx_norm: float,
        cy_norm: float,
        alpha: Optional[float] = None,
        pitch_gain: Optional[float] = None,
        floor_pitch_boost: Optional[float] = None,
    ):
        """
        Point head toward normalized image coords (0–1). x: left=0, right=1; y: top=0, bottom=1.
        Low y = object high in frame (look up); high y = object low / on floor (look down).

        ``pitch_gain`` scales vertical aiming (default 0.3; use ~0.55–0.75 when following ground objects).
        ``floor_pitch_boost`` extra downward bias when cy_norm > 0.5 (rad, added to desired pitch).
        """
        cx_norm = max(0.0, min(1.0, float(cx_norm)))
        cy_norm = max(0.0, min(1.0, float(cy_norm)))

        desired_yaw = -((cx_norm - 0.5) * math.pi * 0.5)
        pg = 0.3 if pitch_gain is None else max(0.05, float(pitch_gain))
        desired_pitch = (cy_norm - 0.5) * math.pi * pg
        if floor_pitch_boost and cy_norm > 0.5:
            desired_pitch += float(floor_pitch_boost) * (cy_norm - 0.5) * 2.0

        mn, mx = self._head_pitch_min, self._head_pitch_max
        desired_pitch = max(mn, min(mx, desired_pitch))

        a = 0.08 if alpha is None else max(0.01, min(1.0, float(alpha)))

        new_yaw = self._head_yaw + a * (desired_yaw - self._head_yaw)
        new_pitch = self._head_pitch + a * (desired_pitch - self._head_pitch)
        new_pitch = max(mn, min(mx, new_pitch))

        self.set_head_yaw(new_yaw)
        self.set_head_pitch(new_pitch)

    # ------------------------------------------------------------------
    # Joint / pose control
    # ------------------------------------------------------------------

    def set_joint(self, name: str, angle: float):
        self._set(name, angle)

    def go_to_rest(self):
        for name, angle in REST_POSE.items():
            self._set(name, angle)

    def _set(self, name: str, angle: float):
        motor = self._motors.get(name)
        if motor is None:
            logger.debug(f"NaoInterface: joint '{name}' not found")
            return

        try:
            motor.setPosition(float(angle))
        except Exception as exc:
            logger.warning(f"NaoInterface: setPosition failed for {name}: {exc}")