"""
scene_state.py — NAO Scene State Extractor

Produces both:
  1. simple top-level payload: state["objects"]
  2. compatibility payload: state["scene"]["objects"]

Distance behavior:
  - Prefer RangeFinder depth (device ``NAO_RANGE_FINDER_NAME``, default ``HeadRangeFinder``)
    inside the YOLO bounding box.
  - Fall back to a rough bounding-box bucket when depth is unavailable.
"""

import os
import math
import time
import cv2
import numpy as np
from typing import Optional

from range_finder_util import (
    default_range_finder_name,
    median_depth_in_camera_box,
    read_range_depth_hw,
)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(_SRC_DIR, "snapshots")

NAO_HFOV_DEG = 60.9
NAO_VFOV_DEG = 47.6
RANGEFINDER_NAME = default_range_finder_name()
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


class SceneStateExtractor:
    def __init__(self, robot, camera, timestep: int, camera_name: str = "CameraTop"):
        self._robot = robot
        self._camera = camera
        self._timestep = timestep
        self._camera_name = camera_name

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        self._imu = _safe_enable(robot, "inertial unit", timestep)
        self._gps = _safe_enable(robot, "gps", timestep)
        self._rangefinder = _safe_enable(robot, RANGEFINDER_NAME, timestep)
        print(f"[SceneState] RangeFinder={'✓' if self._rangefinder else '✗'}")

        self._sonar = {}
        for name in SONAR_SENSORS:
            dev = _safe_enable(robot, name, timestep)
            if dev is not None:
                self._sonar[name] = dev

        self._touch = {}
        for name in TOUCH_SENSORS:
            dev = _safe_enable(robot, name, timestep)
            if dev is not None:
                self._touch[name] = dev

        self._joints = {}
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
            f"RangeFinder={'✓' if self._rangefinder else '✗'}  "
            f"Sonar={len(self._sonar)}  "
            f"Touch={len(self._touch)}  "
            f"Joints={len(self._joints)}"
        )

    @property
    def range_finder(self):
        """Webots RangeFinder device (or ``None``) — same instance used for ``distance_m``."""
        return self._rangefinder

    def _screen_label(self, center_x_px: int) -> str:
        camera_width = self._camera.getWidth()
        horizontal_fraction = center_x_px / camera_width if camera_width > 0 else 0.5
        if horizontal_fraction < 0.38:
            return "left"
        if horizontal_fraction > 0.62:
            return "right"
        return "center"

    def _distance_bucket(self, box_height_fraction: float, distance_m: Optional[float] = None) -> str:
        if distance_m is not None:
            if distance_m <= 0.40:
                return "very_near"
            if distance_m <= 0.85:
                return "near"
            if distance_m <= 1.75:
                return "medium"
            return "far"
        if box_height_fraction > 0.40:
            return "very_near"
        if box_height_fraction > 0.18:
            return "near"
        if box_height_fraction > 0.07:
            return "medium"
        return "far"

    def _screen_angle_deg(self, center_x_px: int) -> float:
        camera_width = self._camera.getWidth()
        focal_length = camera_width / (2.0 * math.tan(math.radians(NAO_HFOV_DEG / 2.0)))
        return round(math.degrees(math.atan2(center_x_px - camera_width / 2.0, focal_length)), 1)

    def _read_sonar(self) -> dict:
        sonar_readings = {}
        for sensor_name, sensor_device in self._sonar.items():
            sensor_value = _safe_read(sensor_device, "getValue")
            reading_key = "left_m" if "Left" in sensor_name else "right_m"
            sonar_readings[reading_key] = round(sensor_value, 2) if sensor_value is not None else None
        return sonar_readings

    def _read_gps(self) -> Optional[dict]:
        gps_values = _safe_read(self._gps, "getValues")
        if gps_values is None:
            return None
        return {"x_m": round(gps_values[0], 3), "y_m": round(gps_values[1], 3), "z_m": round(gps_values[2], 3)}

    def _read_heading(self) -> Optional[float]:
        roll_pitch_yaw = _safe_read(self._imu, "getRollPitchYaw")
        if roll_pitch_yaw is None:
            return None
        return round(math.degrees(roll_pitch_yaw[2]), 1)

    def _depth_for_box(self, box_x: int, box_y: int, box_w: int, box_h: int) -> Optional[float]:
        if self._rangefinder is None:
            return None
        try:
            depth_array = read_range_depth_hw(self._rangefinder)
            if depth_array is None:
                return None
            min_range = float(self._rangefinder.getMinRange())
            max_range = float(self._rangefinder.getMaxRange())
            camera_width = self._camera.getWidth()
            camera_height = self._camera.getHeight()
            return median_depth_in_camera_box(
                depth_array, camera_width, camera_height, box_x, box_y, box_w, box_h, min_m=min_range + 1e-4, max_m=max_range - 1e-3
            )
        except Exception as exc:
            print(f"[SceneState] RangeFinder depth error: {exc}")
            return None

    def _build_objects(self, detections: list) -> list:
        camera_width = self._camera.getWidth()
        camera_height = self._camera.getHeight()
        result = []
        for detection in detections:
            box_x, box_y, box_w, box_h = detection["box"]
            center_x = box_x + box_w // 2
            center_y = box_y + box_h // 2
            height_fraction = box_h / camera_height if camera_height > 0 else 0.0
            width_fraction = box_w / camera_width if camera_width > 0 else 0.0
            center_x_normalized = center_x / camera_width if camera_width > 0 else 0.5
            center_y_normalized = center_y / camera_height if camera_height > 0 else 0.5
            depth_m = self._depth_for_box(box_x, box_y, box_w, box_h)
            distance_bucket = self._distance_bucket(height_fraction, depth_m)
            obj = {
                "label": detection["label"],
                "confidence": round(float(detection.get("confidence", 0.0)), 3),
                "position": self._screen_label(center_x),
                "distance": distance_bucket,
                "height_frac": round(height_fraction, 3),
                "width_frac": round(width_fraction, 3),
                "cx_norm": round(max(0.0, min(1.0, center_x_normalized)), 3),
                "cy_norm": round(max(0.0, min(1.0, center_y_normalized)), 3),
                "distance_m": depth_m,
                "depth_distance_m": depth_m,
                "estimated_distance_m": depth_m,
                "bounding_box": {"x": int(box_x), "y": int(box_y), "width": int(box_w), "height": int(box_h)},
                "center_px": {"x": int(center_x), "y": int(center_y)},
                "screen_position": {"x_norm": round(max(0.0, min(1.0, center_x_normalized)), 3), "y_norm": round(max(0.0, min(1.0, center_y_normalized)), 3)},
                "horizontal_angle_deg": self._screen_angle_deg(center_x),
                "relative_distance": distance_bucket,
                "centred_in_frame": abs(center_x - camera_width / 2) < camera_width * 0.15,
            }
            result.append(obj)
        result.sort(key=lambda obj_item: (obj_item["distance_m"] if obj_item["distance_m"] is not None else 999.0, -obj_item["height_frac"]))
        return result

    def _save_snapshot(self, bgr_frame: np.ndarray, sim_time_ms: int) -> str:
        filename = f"snapshot_{sim_time_ms:010d}_{int(time.time())}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, bgr_frame)
        return path

    def get_joint_sensor(self, name: str):
        return self._joints.get(name)

    def get_touch_sensor(self, name: str):
        return self._touch.get(name)

    def capture(self, bgr_frame: np.ndarray, detections: list, sim_time_ms: int, frame_count: int = 0, trigger: str = "manual", save_snapshot: bool = True) -> tuple:
        snapshot_path = self._save_snapshot(bgr_frame, sim_time_ms) if save_snapshot else None
        objects = self._build_objects(detections)
        sonar = self._read_sonar()
        gps = self._read_gps()
        heading = self._read_heading()
        state = {
            "objects": objects,
            "sonar": sonar,
            "gps": gps,
            "heading_deg": heading,
            "trigger": trigger,
            "meta": {"trigger": trigger, "sim_time_ms": sim_time_ms, "frame_count": frame_count, "snapshot_path": snapshot_path},
            "camera": {"device": self._camera_name, "resolution": {"width": self._camera.getWidth(), "height": self._camera.getHeight()}, "hfov_deg": NAO_HFOV_DEG, "vfov_deg": NAO_VFOV_DEG},
            "robot": {"gps_position": gps, "heading_deg": heading},
            "sensors": {"sonar": sonar, "rangefinder": {"enabled": self._rangefinder is not None, "name": RANGEFINDER_NAME}},
            "scene": {"objects": objects},
        }
        return state, snapshot_path
