"""
llm_bridge.py — LLM Decision Bridge

Receives a scene state dict and a snapshot image path, then dispatches
them to the LLM planning layer.

Currently: prints everything to stdout for debugging.
Future:     will call the vision + text LLM API and return a plan.

Usage:
    from llm_bridge import LLMBridge

    bridge = LLMBridge()
    response = bridge.send(state_dict, snapshot_path)
"""

import json
import os

# ANSI colour helpers (gracefully disabled on terminals that don't support them)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_MAGENTA= "\033[35m"
_DIM    = "\033[2m"


def _header(text: str) -> str:
    width = 70
    pad   = (width - len(text) - 2) // 2
    return f"\n{_BOLD}{_CYAN}{'─' * pad} {text} {'─' * pad}{_RESET}\n"


class LLMBridge:
    """
    Bridge between the scene state extractor and the LLM planning layer.

    Parameters
    ----------
    verbose : bool
        If True, print the full JSON state. If False, print a summary only.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        print(f"{_GREEN}[LLMBridge]{_RESET} Initialised (stub mode — no LLM calls yet).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, state: dict, snapshot_path: str) -> dict:
        """
        Process one decision point.

        Parameters
        ----------
        state         : dict from SceneStateExtractor.capture()
        snapshot_path : str  absolute path to the saved JPEG snapshot

        Returns
        -------
        response : dict  (stub — always returns empty plan for now)
        """
        self._print_state(state, snapshot_path)
        self._print_snapshot_info(snapshot_path)

        # ── Placeholder for LLM call ─────────────────────────────────
        # In the future this will:
        #   1. Encode the snapshot as base64.
        #   2. Combine it with the JSON state into a prompt.
        #   3. Call the vision LLM API (e.g. GPT-4o, Gemini Pro Vision).
        #   4. Parse the returned action plan.
        #   5. Return structured action commands to nao_cam.py.
        # ─────────────────────────────────────────────────────────────
        response = {
            "status":  "stub",
            "message": "LLM not yet connected.",
            "actions": [],
        }
        print(f"\n{_DIM}[LLMBridge] → Response stub: {response}{_RESET}\n")
        return response

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def _print_state(self, state: dict, snapshot_path: str) -> None:
        print(_header("SCENE STATE"))

        # ── Meta ──────────────────────────────────────────────────────
        meta = state.get("meta", {})
        print(f"{_BOLD}Trigger   :{_RESET} {meta.get('trigger', '?')}")
        print(f"{_BOLD}Sim time  :{_RESET} {meta.get('sim_time_ms', 0) / 1000:.2f} s")
        print(f"{_BOLD}Wall time :{_RESET} {meta.get('wall_time', '?')}")
        print(f"{_BOLD}Frame     :{_RESET} {meta.get('frame_count', '?')}")

        # ── Robot pose ────────────────────────────────────────────────
        robot = state.get("robot", {})
        print(_header("Robot Pose & Motion"))
        ori = robot.get("orientation", {})
        if ori:
            print(
                f"  Orientation  roll={ori.get('roll_deg','?'):>7.2f}°  "
                f"pitch={ori.get('pitch_deg','?'):>7.2f}°  "
                f"yaw={ori.get('yaw_deg','?'):>7.2f}°"
            )
        acc = robot.get("acceleration", {})
        if acc:
            print(
                f"  Accel (m/s²) x={acc.get('x_ms2',0):>8.4f}  "
                f"y={acc.get('y_ms2',0):>8.4f}  "
                f"z={acc.get('z_ms2',0):>8.4f}"
            )
        gyro = robot.get("angular_velocity", {})
        if gyro:
            print(
                f"  Gyro (rad/s) x={gyro.get('x_rads',0):>8.4f}  "
                f"y={gyro.get('y_rads',0):>8.4f}  "
                f"z={gyro.get('z_rads',0):>8.4f}"
            )
        gps = robot.get("gps_position")
        if gps:
            print(
                f"  GPS (m)      x={gps.get('x_m',0):>8.4f}  "
                f"y={gps.get('y_m',0):>8.4f}  "
                f"z={gps.get('z_m',0):>8.4f}"
            )

        # ── Joints ────────────────────────────────────────────────────
        joints = robot.get("joint_positions", {})
        if joints:
            print(f"\n  {_BOLD}Joint positions (rad):{_RESET}")
            items = list(joints.items())
            for i in range(0, len(items), 3):
                row = items[i:i+3]
                print("  " + "   ".join(f"{k:<20} {v:>7.4f}" for k, v in row))

        # ── Sensors ───────────────────────────────────────────────────
        sensors = state.get("sensors", {})
        print(_header("Sensors"))
        sonar = sensors.get("sonar", {})
        if sonar:
            print(
                f"  Sonar  left={sonar.get('left_m','N/A')} m  "
                f"right={sonar.get('right_m','N/A')} m"
            )
        touch = sensors.get("touch", {})
        if touch:
            active = [k for k, v in touch.items() if v]
            print(f"  Touch  active={active if active else 'none'}")

        # ── Scene ─────────────────────────────────────────────────────
        scene = state.get("scene", {})
        objects = scene.get("objects", [])
        print(_header(f"Detected Objects  ({len(objects)} total)"))
        if not objects:
            print(f"  {_DIM}No objects detected.{_RESET}")
        else:
            for i, obj in enumerate(objects):
                dist_str = (
                    f"{obj['estimated_distance_m']:.2f} m"
                    if obj.get("estimated_distance_m") is not None
                    else obj.get("relative_distance", "unknown")
                )
                centred = "✓ centred" if obj.get("centred_in_frame") else ""
                print(
                    f"  [{i+1}] {_YELLOW}{obj['label']:<20}{_RESET}"
                    f"  conf={obj['confidence']:.2f}"
                    f"  dist≈{dist_str:<10}"
                    f"  angle={obj['horizontal_angle_deg']:>6.1f}°"
                    f"  {centred}"
                )

        # ── Full JSON (verbose mode) ───────────────────────────────────
        if self.verbose:
            print(_header("Full JSON Payload"))
            print(_DIM + json.dumps(state, indent=2, default=str) + _RESET)

    def _print_snapshot_info(self, snapshot_path: str) -> None:
        print(_header("Snapshot"))
        if os.path.isfile(snapshot_path):
            size_kb = os.path.getsize(snapshot_path) / 1024
            print(f"  {_GREEN}Saved:{_RESET} {snapshot_path}  ({size_kb:.1f} KB)")
        else:
            print(f"  {_YELLOW}WARNING: snapshot file not found at {snapshot_path}{_RESET}")
        print()
