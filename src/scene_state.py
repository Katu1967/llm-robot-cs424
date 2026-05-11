

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
            pass  # device exists but enable failed
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
        print(f"[SceneState] RangeFinder={'yes' if self._rangefinder else 'no'}")

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
            f"IMU={'yes' if self._imu else 'no'}  "
            f"GPS={'yes' if self._gps else 'no'}  "
            f"RangeFinder={'yes' if self._rangefinder else 'no'}  "
            f"Sonar={len(self._sonar)}  "
            f"Touch={len(self._touch)}  "
            f"Joints={len(self._joints)}"
        )

    @property
    def range_finder(self):
        return self._rangefinder

    def _screen_label(self, center_x_px: int) -> str:
        image_width = self._camera.getWidth()
        horizontal_fraction = center_x_px / image_width if image_width > 0 else 0.5

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
        image_width = self._camera.getWidth()
        focal_length_px = image_width / (2.0 * math.tan(math.radians(NAO_HFOV_DEG / 2.0)))

        return round(math.degrees(math.atan2(center_x_px - image_width / 2.0, focal_length_px)), 1)

    def _read_sonar(self) -> dict:
        readings = {}

        for device_name, device in self._sonar.items():
            raw_meters = _safe_read(device, "getValue")
            json_key = "left_m" if "Left" in device_name else "right_m"
            readings[json_key] = round(raw_meters, 2) if raw_meters is not None else None

        return readings

    def _read_gps(self) -> Optional[dict]:
        xyz = _safe_read(self._gps, "getValues")

        if xyz is None:
            return None

        return {"x_m": round(xyz[0], 3), "y_m": round(xyz[1], 3), "z_m": round(xyz[2], 3)}

    def _read_heading(self) -> Optional[float]:
        roll_pitch_yaw = _safe_read(self._imu, "getRollPitchYaw")

        if roll_pitch_yaw is None:
            return None

        return round(math.degrees(roll_pitch_yaw[2]), 1)

    def _depth_for_box(self, x: int, y: int, w: int, h: int) -> Optional[float]:
        if self._rangefinder is None:
            return None

        try:
            depth_hw = read_range_depth_hw(self._rangefinder)

            if depth_hw is None:
                return None

            min_range_m = float(self._rangefinder.getMinRange())
            max_range_m = float(self._rangefinder.getMaxRange())
            cam_w = self._camera.getWidth()
            cam_h = self._camera.getHeight()

            return median_depth_in_camera_box(
                depth_hw, cam_w, cam_h, x, y, w, h, min_m=min_range_m + 1e-4, max_m=max_range_m - 1e-3
            )
        except Exception as exc:
            print(f"[SceneState] RangeFinder depth error: {exc}")
            return None

    def _build_objects(self, detections: list) -> list:
        cam_w = self._camera.getWidth()
        cam_h = self._camera.getHeight()
        objects_out = []

        for det in detections:
            box_x, box_y, box_w, box_h = det["box"]
            center_x = box_x + box_w // 2
            center_y = box_y + box_h // 2
            height_fraction = box_h / cam_h if cam_h > 0 else 0.0
            width_fraction = box_w / cam_w if cam_w > 0 else 0.0
            cx_norm = center_x / cam_w if cam_w > 0 else 0.5
            cy_norm = center_y / cam_h if cam_h > 0 else 0.5
            depth_m = self._depth_for_box(box_x, box_y, box_w, box_h)
            distance_bucket = self._distance_bucket(height_fraction, depth_m)

            row = {
                "label": det["label"],
                "confidence": round(float(det.get("confidence", 0.0)), 3),
                "position": self._screen_label(center_x),
                "distance": distance_bucket,
                "height_frac": round(height_fraction, 3),
                "width_frac": round(width_fraction, 3),
                "cx_norm": round(max(0.0, min(1.0, cx_norm)), 3),
                "cy_norm": round(max(0.0, min(1.0, cy_norm)), 3),
                "distance_m": depth_m,
                "depth_distance_m": depth_m,
                "estimated_distance_m": depth_m,
                "bounding_box": {"x": int(box_x), "y": int(box_y), "width": int(box_w), "height": int(box_h)},
                "center_px": {"x": int(center_x), "y": int(center_y)},
                "screen_position": {"x_norm": round(max(0.0, min(1.0, cx_norm)), 3), "y_norm": round(max(0.0, min(1.0, cy_norm)), 3)},
                "horizontal_angle_deg": self._screen_angle_deg(center_x),
                "relative_distance": distance_bucket,
                "centred_in_frame": abs(center_x - cam_w / 2) < cam_w * 0.15,
            }
            objects_out.append(row)

        objects_out.sort(
            key=lambda row: (row["distance_m"] if row["distance_m"] is not None else 999.0, -row["height_frac"])
        )

        return objects_out

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
