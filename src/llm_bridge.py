"""
LLM decision bridge.

Takes a scene state dict and a snapshot path. For now this only prints debug text.
Later it can call a vision plus text API and return an action plan.

Usage:
    from llm_bridge import LLMBridge
    bridge = LLMBridge()
    response = bridge.send(state_dict, snapshot_path)
"""

import json
import os


class LLMBridge:
    """
    Connects scene state from the extractor to the planning layer.

    verbose: if True print full JSON. If False print a shorter summary only.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        print("[LLMBridge] Initialised in stub mode. No LLM calls yet.")

    def send(self, state: dict, snapshot_path: str) -> dict:
        """
        Handle one decision point.

        state is the dict from SceneStateExtractor.capture().
        snapshot_path is the absolute path to the saved snapshot image.

        Returns a stub dict until a real LLM is wired in.
        """
        self._print_state(state, snapshot_path)
        self._print_snapshot_info(snapshot_path)

        # Placeholder until the real LLM is connected.
        # Future steps will encode the image, build the prompt, call the API, and parse actions.
        response = {
            "status": "stub",
            "message": "LLM not yet connected.",
            "actions": [],
        }
        print(f"[LLMBridge] Response stub: {response}\n")
        return response

    def _print_state(self, state: dict, snapshot_path: str) -> None:
        print("\n[LLMBridge] SCENE STATE\n")

        # When and why this capture happened.
        meta = state.get("meta", {})
        print(f"Trigger: {meta.get('trigger', '?')}")
        print(f"Sim time: {meta.get('sim_time_ms', 0) / 1000:.2f} s")
        print(f"Wall time: {meta.get('wall_time', '?')}")
        print(f"Frame: {meta.get('frame_count', '?')}")

        print("\n[LLMBridge] Robot pose and motion\n")

        robot = state.get("robot", {})
        ori = robot.get("orientation", {})
        if ori:
            print(
                f"  Orientation roll={ori.get('roll_deg', '?'):>7.2f} deg "
                f"pitch={ori.get('pitch_deg', '?'):>7.2f} deg "
                f"yaw={ori.get('yaw_deg', '?'):>7.2f} deg"
            )

        acc = robot.get("acceleration", {})
        if acc:
            print(
                f"  Accel m per s squared x={acc.get('x_ms2', 0):>8.4f} "
                f"y={acc.get('y_ms2', 0):>8.4f} z={acc.get('z_ms2', 0):>8.4f}"
            )

        gyro = robot.get("angular_velocity", {})
        if gyro:
            print(
                f"  Gyro rad per s x={gyro.get('x_rads', 0):>8.4f} "
                f"y={gyro.get('y_rads', 0):>8.4f} z={gyro.get('z_rads', 0):>8.4f}"
            )

        gps = robot.get("gps_position")
        if gps:
            print(
                f"  GPS meters x={gps.get('x_m', 0):>8.4f} "
                f"y={gps.get('y_m', 0):>8.4f} z={gps.get('z_m', 0):>8.4f}"
            )

        # Joint angles in radians.
        joints = robot.get("joint_positions", {})
        if joints:
            print("\n  Joint positions in radians:")
            items = list(joints.items())
            for i in range(0, len(items), 3):
                row = items[i : i + 3]
                print("  " + "   ".join(f"{k:<20} {v:>7.4f}" for k, v in row))

        print("\n[LLMBridge] Sensors\n")

        sensors = state.get("sensors", {})
        sonar = sensors.get("sonar", {})
        if sonar:
            print(
                f"  Sonar left={sonar.get('left_m', 'N/A')} m "
                f"right={sonar.get('right_m', 'N/A')} m"
            )

        touch = sensors.get("touch", {})
        if touch:
            active = [k for k, v in touch.items() if v]
            print(f"  Touch active sensors: {active if active else 'none'}")

        print("\n[LLMBridge] Detected objects\n")

        scene = state.get("scene", {})
        objects = scene.get("objects", [])
        print(f"Total objects: {len(objects)}")

        if not objects:
            print("  No objects detected.")
        else:
            for i, obj in enumerate(objects):
                if obj.get("estimated_distance_m") is not None:
                    dist_str = f"{obj['estimated_distance_m']:.2f} m"
                else:
                    dist_str = str(obj.get("relative_distance", "unknown"))

                # True if the box is near the horizontal center of the image.
                centred = "yes centred" if obj.get("centred_in_frame") else ""

                print(
                    f"  [{i + 1}] {obj['label']:<20} "
                    f"conf={obj['confidence']:.2f} "
                    f"dist about {dist_str:<10} "
                    f"angle={obj['horizontal_angle_deg']:>6.1f} deg "
                    f"{centred}"
                )

        if self.verbose:
            print("\n[LLMBridge] Full JSON payload\n")
            print(json.dumps(state, indent=2, default=str))

    def _print_snapshot_info(self, snapshot_path: str) -> None:
        print("\n[LLMBridge] Snapshot file\n")

        if os.path.isfile(snapshot_path):
            size_kb = os.path.getsize(snapshot_path) / 1024
            print(f"  Saved: {snapshot_path} ({size_kb:.1f} KB)")
        else:
            print(f"  WARNING snapshot file not found: {snapshot_path}")

        print()
