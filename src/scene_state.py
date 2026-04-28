"""
scene_state.py — NAO Scene State Extractor

Collects all available sensor data from the NAO robot in Webots and
packages it into a structured JSON-ready dict.  Also saves a camera
snapshot to disk.

Usage (inside a Webots controller loop):

    from scene_state import SceneStateExtractor

    extractor = SceneStateExtractor(robot, camera, TIMESTEP)
    # ... inside the step loop:
    state, snapshot_path = extractor.capture(bgr_frame, yolo_detections, sim_time_ms)
"""

import os
import math
import time
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR  = os.path.join(_SRC_DIR, "snapshots")

# Approximate NAO CameraTop horizontal field of view (degrees).
# Used for screen-angle and rough distance estimates.
NAO_HFOV_DEG = 60.9
NAO_VFOV_DEG = 47.6

# "Known" object heights (metres) for focal-length distance estimation.
# Only used when the label is in the table; otherwise falls back to a
# bucketed heuristic.
KNOWN_HEIGHTS_M: dict[str, float] = {
    "person":       1.70,
    "chair":        0.90,
    "dining table": 0.75,
    "bottle":       0.25,
    "cup":          0.10,
    "laptop":       0.35,
    "tv":           0.60,
    "cell phone":   0.15,
    "book":         0.24,
    "dog":          0.55,
    "cat":          0.30,
    "ball":         0.22,
}

# ---------------------------------------------------------------------------
# NAO device names
# ---------------------------------------------------------------------------

# All standard NAO joints (Webots Nao.proto, 25-DOF variant).
# Missing joints are silently skipped.
NAO_JOINTS = [
    # Head
    "HeadYaw", "HeadPitch",
    # Left arm
    "LShoulderPitch", "LShoulderRoll",
    "LElbowYaw",      "LElbowRoll",
    "LWristYaw",      "LHand",
    # Right arm
    "RShoulderPitch", "RShoulderRoll",
    "RElbowYaw",      "RElbowRoll",
    "RWristYaw",      "RHand",
    # Left leg
    "LHipYawPitch", "LHipRoll", "LHipPitch",
    "LKneePitch",
    "LAnklePitch",  "LAnkleRoll",
    # Right leg
    "RHipYawPitch", "RHipRoll", "RHipPitch",
    "RKneePitch",
    "RAnklePitch",  "RAnkleRoll",
]

TOUCH_SENSORS = [
    "Head/Touch/Front",
    "Head/Touch/Middle",
    "Head/Touch/Rear",
    "LFoot/Bumper/Left",
    "LFoot/Bumper/Right",
    "RFoot/Bumper/Left",
    "RFoot/Bumper/Right",
]

SONAR_SENSORS = ["Sonar/Left", "Sonar/Right"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_enable(robot, name: str, timestep: int):
    """
    Try to get and enable a named device.  Returns the device object on
    success, or None if the device is not present in this world.
    """
    dev = robot.getDevice(name)
    if dev is not None:
        try:
            dev.enable(timestep)
        except Exception:
            pass
    return dev


def _safe_read(dev, method: str, fallback=None):
    """Call dev.<method>() safely, returning fallback on any error."""
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
    Initialise once after creating the Robot instance.
    Call capture() every time you want a full scene snapshot.

    Parameters
    ----------
    robot     : controller.Robot
    camera    : the already-enabled Camera device
    timestep  : simulation timestep in ms
    camera_name : name of the camera device (for metadata)
    """

    def __init__(self, robot, camera, timestep: int, camera_name: str = "CameraTop"):
        self._robot       = robot
        self._camera      = camera
        self._timestep    = timestep
        self._camera_name = camera_name

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        # --- Inertial sensors ---
        self._accel = _safe_enable(robot, "accelerometer", timestep)
        self._gyro  = _safe_enable(robot, "gyro",          timestep)
        self._imu   = _safe_enable(robot, "inertial unit", timestep)
        self._gps   = _safe_enable(robot, "gps",           timestep)

        # --- Sonar ---
        self._sonar: dict[str, object] = {}
        for name in SONAR_SENSORS:
            dev = _safe_enable(robot, name, timestep)
            if dev is not None:
                self._sonar[name] = dev

        # --- Touch sensors ---
        self._touch: dict[str, object] = {}
        for name in TOUCH_SENSORS:
            dev = _safe_enable(robot, name, timestep)
            if dev is not None:
                self._touch[name] = dev

        # --- Joint position sensors ---
        self._joints: dict[str, object] = {}
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
    # Camera geometry helpers
    # ------------------------------------------------------------------

    def _focal_length_px(self) -> float:
        """Approximate horizontal focal length in pixels."""
        w = self._camera.getWidth()
        return w / (2.0 * math.tan(math.radians(NAO_HFOV_DEG / 2.0)))

    def _estimate_distance(self, label: str, box_h_px: int) -> float | None:
        """
        Estimate object distance in metres using the pinhole camera model.
            distance = (real_height * focal_length) / box_height_px
        Returns None if we have no known height for this label.
        """
        real_h = KNOWN_HEIGHTS_M.get(label.lower())
        if real_h is None or box_h_px < 1:
            return None
        fh = (self._camera.getHeight()
              / (2.0 * math.tan(math.radians(NAO_VFOV_DEG / 2.0))))
        return round((real_h * fh) / box_h_px, 2)

    def _distance_bucket(self, box_h_frac: float) -> str:
        """Rough distance label based on bbox height as fraction of frame."""
        if box_h_frac > 0.5:
            return "very_near"
        elif box_h_frac > 0.25:
            return "near"
        elif box_h_frac > 0.10:
            return "medium"
        else:
            return "far"

    def _screen_angle_deg(self, cx_px: int) -> float:
        """
        Horizontal angle from camera centre to object centre (degrees).
        Negative = left of centre, positive = right.
        """
        w = self._camera.getWidth()
        fl = self._focal_length_px()
        return round(math.degrees(math.atan2(cx_px - w / 2.0, fl)), 1)

    # ------------------------------------------------------------------
    # Sensor read helpers
    # ------------------------------------------------------------------

    def _read_accel(self) -> dict:
        v = _safe_read(self._accel, "getValues")
        if v is None:
            return {}
        return {"x_ms2": round(v[0], 4), "y_ms2": round(v[1], 4), "z_ms2": round(v[2], 4)}

    def _read_gyro(self) -> dict:
        v = _safe_read(self._gyro, "getValues")
        if v is None:
            return {}
        return {"x_rads": round(v[0], 4), "y_rads": round(v[1], 4), "z_rads": round(v[2], 4)}

    def _read_imu(self) -> dict:
        v = _safe_read(self._imu, "getRollPitchYaw")
        if v is None:
            return {}
        return {
            "roll_deg":  round(math.degrees(v[0]), 2),
            "pitch_deg": round(math.degrees(v[1]), 2),
            "yaw_deg":   round(math.degrees(v[2]), 2),
        }

    def _read_gps(self) -> dict | None:
        v = _safe_read(self._gps, "getValues")
        if v is None:
            return None
        return {"x_m": round(v[0], 4), "y_m": round(v[1], 4), "z_m": round(v[2], 4)}

    def _read_sonar(self) -> dict:
        out = {}
        for name, dev in self._sonar.items():
            val = _safe_read(dev, "getValue")
            key = "left_m" if "Left" in name else "right_m"
            out[key] = round(val, 3) if val is not None else None
        return out

    def _read_touch(self) -> dict:
        out = {}
        for name, dev in self._touch.items():
            val = _safe_read(dev, "getValue")
            # touch sensors return 0.0 or 1.0
            out[name] = bool(val) if val is not None else None
        return out

    def _read_joints(self) -> dict:
        out = {}
        for name, ps in self._joints.items():
            val = _safe_read(ps, "getValue")
            out[name] = round(val, 4) if val is not None else None
        return out

    # ------------------------------------------------------------------
    # Object analysis from YOLO detections
    # ------------------------------------------------------------------

    def _analyse_objects(self, detections: list[dict]) -> list[dict]:
        """
        Enrich each YOLO detection with spatial reasoning fields.
        """
        cam_w = self._camera.getWidth()
        cam_h = self._camera.getHeight()
        enriched = []

        for det in detections:
            x, y, w, h = det["box"]
            cx = x + w // 2
            cy = y + h // 2

            h_frac = h / cam_h if cam_h > 0 else 0.0
            est_dist_m = self._estimate_distance(det["label"], h)

            enriched.append({
                "label":            det["label"],
                "confidence":       round(det["confidence"], 3),
                "bounding_box":     {"x": x, "y": y, "width": w, "height": h},
                "center_px":        {"x": cx, "y": cy},
                # Relative screen position (0=left/top, 1=right/bottom)
                "screen_position":  {
                    "x_norm": round(cx / cam_w, 3),
                    "y_norm": round(cy / cam_h, 3),
                },
                # Horizontal angle from camera centre
                "horizontal_angle_deg": self._screen_angle_deg(cx),
                # Distance estimate
                "estimated_distance_m":   est_dist_m,
                "relative_distance":      self._distance_bucket(h_frac),
                # Is the object roughly centred in frame?
                "centred_in_frame": abs(cx - cam_w / 2) < cam_w * 0.15,
            })

        # Sort nearest first (objects with known distance estimate first,
        # then by bounding-box height descending as a proxy)
        enriched.sort(
            key=lambda o: (
                o["estimated_distance_m"] if o["estimated_distance_m"] is not None else 999
            )
        )
        return enriched

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(self, bgr_frame: np.ndarray, sim_time_ms: int) -> str:
        """
        Save the current camera frame as a JPEG.
        Returns the absolute path to the saved file.
        """
        filename = f"snapshot_{sim_time_ms:010d}_{int(time.time())}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, bgr_frame)
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(
        self,
        bgr_frame:   np.ndarray,
        detections:  list[dict],
        sim_time_ms: int,
        frame_count: int = 0,
        trigger:     str = "manual",
    ) -> tuple[dict, str]:
        """
        Collect all sensor data and produce a JSON-ready state dict.

        Parameters
        ----------
        bgr_frame   : current camera BGR frame
        detections  : YOLO detection dicts from YOLODetector.detect()
        sim_time_ms : current simulation time in ms (robot.getTime() * 1000)
        frame_count : number of frames elapsed since start
        trigger     : reason string — 'manual', 'periodic', etc.

        Returns
        -------
        state       : dict  (JSON-serialisable scene state)
        snapshot_path : str  absolute path to the saved JPEG
        """
        cam_w = self._camera.getWidth()
        cam_h = self._camera.getHeight()

        snapshot_path = self._save_snapshot(bgr_frame, sim_time_ms)

        state = {
            "meta": {
                "trigger":       trigger,
                "sim_time_ms":   sim_time_ms,
                "wall_time":     time.strftime("%Y-%m-%dT%H:%M:%S"),
                "frame_count":   frame_count,
                "snapshot_path": snapshot_path,
            },
            "camera": {
                "device":     self._camera_name,
                "resolution": {"width": cam_w, "height": cam_h},
                "fov_deg":    {"horizontal": NAO_HFOV_DEG, "vertical": NAO_VFOV_DEG},
            },
            "robot": {
                "orientation":     self._read_imu(),
                "acceleration":    self._read_accel(),
                "angular_velocity": self._read_gyro(),
                "gps_position":    self._read_gps(),
                "joint_positions": self._read_joints(),
            },
            "sensors": {
                "sonar": self._read_sonar(),
                "touch":  self._read_touch(),
            },
            "scene": {
                "object_count": len(detections),
                "objects":      self._analyse_objects(detections),
            },
        }

        return state, snapshot_path
