"""
scene_state.py — NAO Scene State Extractor (simplified)

Produces a minimal, LLM-friendly JSON snapshot:
  - Detected objects with screen position and distance bucket
  - Sonar readings (obstacle proximity)
  - GPS position and heading

The image snapshot is saved separately and passed to the vision model directly.

Usage:
    extractor = SceneStateExtractor(robot, camera, TIMESTEP)
    state, snapshot_path = extractor.capture(bgr_frame, detections, sim_time_ms)
"""

import os
import math
import time
import cv2
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(_SRC_DIR, "snapshots")

# NAO CameraTop field of view
NAO_HFOV_DEG = 60.9
NAO_VFOV_DEG = 47.6

# Sensors
SONAR_SENSORS = ["Sonar/Left", "Sonar/Right"]

NAO_JOINTS = [
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
]

TOUCH_SENSORS = [
    "Head/Touch/Front", "Head/Touch/Middle", "Head/Touch/Rear",
    "LFoot/Bumper/Left", "LFoot/Bumper/Right",
    "RFoot/Bumper/Left", "RFoot/Bumper/Right",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_enable(robot, name: str, timestep: int):
    dev = robot.getDevice(name)
    if dev is not None:
        try:
            dev.enable(timestep)
        except Exception:
            pass
    return dev


def _safe_read(dev, method: str, fallback=None):
    if dev is None:
        return fallback
    try:
        return getattr(dev, method)()
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# SceneStateExtractor
# ---------------------------------------------------------------------------

class SceneStateExtractor:
    """
    Collects sensor data and YOLO detections into a compact,
    LLM-friendly scene state dict. Also saves a camera snapshot.
    """

    def __init__(self, robot, camera, timestep: int, camera_name: str = "CameraTop"):
        self._robot       = robot
        self._camera      = camera
        self._timestep    = timestep
        self._camera_name = camera_name

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        # Sensors we actually use in the simplified payload
        self._imu  = _safe_enable(robot, "inertial unit", timestep)
        self._gps  = _safe_enable(robot, "gps",           timestep)

        self._sonar: dict = {}
        for name in SONAR_SENSORS:
            dev = _safe_enable(robot, name, timestep)
            if dev is not None:
                self._sonar[name] = dev

        # Keep touch/joints enabled (needed by NaoInterface) but don't
        # include them in the LLM payload.
        self._touch: dict = {}
        for name in TOUCH_SENSORS:
            dev = _safe_enable(robot, name, timestep)
            if dev is not None:
                self._touch[name] = dev

        self._joints: dict = {}
        for joint_name in NAO_JOINTS:
            motor = robot.getDevice(joint_name)
            if motor is None:
                continue
            try:
                ps = motor.getPositionSensor()
                if ps is not None:
                    ps.enable(timestep)
                    self._joints[joint_name] = ps
            except Exception:
                pass

        print(
            f"[SceneState] Sensors ready — "
            f"IMU={'✓' if self._imu else '✗'}  "
            f"GPS={'✓' if self._gps else '✗'}  "
            f"Sonar={len(self._sonar)}  "
            f"Touch={len(self._touch)}  "
            f"Joints={len(self._joints)}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _screen_label(self, cx_px: int) -> str:
        """Describe horizontal position as 'left', 'center', or 'right'."""
        w = self._camera.getWidth()
        frac = cx_px / w
        if frac < 0.38:
            return "left"
        elif frac > 0.62:
            return "right"
        return "center"

    def _distance_bucket(self, box_h_frac: float) -> str:
        """Coarse distance label based on bounding-box height fraction.

        Thresholds biased so ``very_near`` / ``near`` align with “within arm’s reach”
        for typical COCO-sized objects in the NAO camera.
        """
        if box_h_frac > 0.40:
            return "very_near"
        elif box_h_frac > 0.18:
            return "near"
        elif box_h_frac > 0.07:
            return "medium"
        return "far"

    def _read_sonar(self) -> dict:
        out = {}
        for name, dev in self._sonar.items():
            val = _safe_read(dev, "getValue")
            key = "left_m" if "Left" in name else "right_m"
            out[key] = round(val, 2) if val is not None else None
        return out

    def _read_gps(self) -> Optional[dict]:
        v = _safe_read(self._gps, "getValues")
        if v is None:
            return None
        return {"x_m": round(v[0], 2), "y_m": round(v[1], 2)}

    def _read_heading(self) -> Optional[float]:
        """Yaw in degrees from the IMU (north = 0, clockwise positive)."""
        v = _safe_read(self._imu, "getRollPitchYaw")
        if v is None:
            return None
        return round(math.degrees(v[2]), 1)

    def _build_objects(self, detections: list) -> list:
        """
        Convert YOLO detections into simple position descriptors.
        Each entry has: label, position (left/center/right), distance bucket.
        """
        cam_w = self._camera.getWidth()
        cam_h = self._camera.getHeight()
        result = []

        for det in detections:
            x, y, w, h = det["box"]
            cx = x + w // 2
            cy = y + h // 2
            h_frac = h / cam_h if cam_h > 0 else 0.0
            cx_norm = cx / cam_w if cam_w > 0 else 0.5
            cy_norm = cy / cam_h if cam_h > 0 else 0.5

            result.append({
                "label":        det["label"],
                "position":     self._screen_label(cx),
                "distance":     self._distance_bucket(h_frac),
                "height_frac":  round(h_frac, 3),
                "cx_norm":      round(max(0.0, min(1.0, cx_norm)), 3),
                "cy_norm":      round(max(0.0, min(1.0, cy_norm)), 3),
            })

        return result

    def _save_snapshot(self, bgr_frame: np.ndarray, sim_time_ms: int) -> str:
        filename = f"snapshot_{sim_time_ms:010d}_{int(time.time())}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, bgr_frame)
        return path

    # ------------------------------------------------------------------
    # Public API — also expose joint/touch sensors for NaoInterface use
    # ------------------------------------------------------------------

    def get_joint_sensor(self, name: str):
        return self._joints.get(name)

    def get_touch_sensor(self, name: str):
        return self._touch.get(name)

    def capture(
        self,
        bgr_frame:   np.ndarray,
        detections:  list,
        sim_time_ms: int,
        frame_count: int = 0,
        trigger:     str = "manual",
        save_snapshot: bool = True,
    ) -> tuple:
        """
        Build and return (state_dict, snapshot_path).

        state_dict is a minimal JSON-ready payload:
        {
          "objects":     [{"label": ..., "position": ..., "distance": ...}],
          "sonar":       {"left_m": ..., "right_m": ...},
          "gps":         {"x_m": ..., "y_m": ...},
          "heading_deg": ...,
          "trigger":     "manual" | "periodic" | ...
        }
        """
        snapshot_path = None
        if save_snapshot:
            snapshot_path = self._save_snapshot(bgr_frame, sim_time_ms)

        state = {
            "objects":     self._build_objects(detections),
            "sonar":       self._read_sonar(),
            "gps":         self._read_gps(),
            "heading_deg": self._read_heading(),
            "trigger":     trigger,
        }

        return state, snapshot_path
