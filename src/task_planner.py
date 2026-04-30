"""
task_planner.py — Llama 3 Task Planner Agent (via Ollama)

The TaskPlanner subscribes to the "scene_state" topic on the SceneBus.
It only calls the LLM when one of two conditions is true:

  1. prompt_user = True
       No active task exists.  The planner prompts the operator via stdin
       to provide a goal, then calls the LLM to build an initial plan.

  2. decision_needed = True
       The executor reported a failure.  The planner calls the LLM with
       the failure context and current scene state to produce a revised plan.

The LLM call runs in a background thread so the Webots simulation loop
(robot.step) is never blocked.

External API:
  planner = TaskPlanner()
  planner.attach(bus)               # subscribe to scene_state
  planner.report_failure(context)   # called by executor on step failure
  planner.report_success()          # called by executor on task completion
  planner.get_plan()                # returns latest plan dict or None
  planner.is_planning()             # True while LLM is thinking

Requirements:
  1. Install Ollama:  https://ollama.com/download  (or: brew install ollama)
  2. Pull the model:  ollama pull llama3.2-vision
  3. Start server:    ollama serve          (runs on http://localhost:11434)

Optional env vars (set in .env or shell):
  OLLAMA_MODEL  — model name, default "llama3.2-vision"
  OLLAMA_HOST   — server URL, default "http://localhost:11434"
"""

import os
import json
import threading
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Optional: load .env automatically if python-dotenv is installed
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env_path = os.path.join(_root_dir, ".env")
    load_dotenv(_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST",  "http://localhost:11434")

# Maximum characters of joint data to send (cuts down token usage)
MAX_JOINT_CHARS = 400

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a task planning agent for a NAO humanoid robot operating inside the
Webots robot simulator.

Your job is to analyse the current scene state (provided as JSON) together with
an optional camera snapshot, then produce a concrete, step-by-step action plan
for the robot to accomplish the given goal.

## Robot Overview
- Platform : SoftBank NAO (25-DOF humanoid)
- Simulator: Webots
- Sensors  : CameraTop/Bottom, Accelerometer, Gyroscope, InertialUnit,
             GPS, Sonar (left & right), foot/head touch bumpers,
             25 joint position encoders

## Available Tools / Actions
[TOOLS WILL BE REGISTERED HERE IN A FUTURE UPDATE]

For now, describe actions in plain English inside the "description" field.
Use the "action" field as a placeholder action name (snake_case).

## Scene State Fields (summary)
- meta              : trigger reason, sim time, frame count, snapshot path
- camera            : device name, resolution, field of view
- robot.orientation : roll / pitch / yaw in degrees
- robot.acceleration: x / y / z in m/s²
- robot.angular_velocity: x / y / z in rad/s
- robot.gps_position: x / y / z in metres (if GPS enabled)
- robot.joint_positions: all 25 joint angles in radians
- sensors.sonar     : left_m / right_m obstacle distances
- sensors.touch     : bumper states (bool)
- scene.objects[]   : label, confidence, bounding_box, estimated_distance_m,
                      relative_distance, horizontal_angle_deg, centred_in_frame

## Response Format
Always respond with a single valid JSON object — no markdown, no code fences:

{
  "reasoning": "<step-by-step analysis of the scene and how you reached the plan>",
  "task_summary": "<one-sentence restatement of the goal>",
  "plan": [
    {
      "step": 1,
      "action": "<snake_case_action_name>",
      "parameters": {},
      "description": "<human-readable description of what the robot should do>"
    }
  ],
  "requires_clarification": false,
  "clarification_question": null,
  "estimated_duration_s": 0
}

## Planning Guidelines
- Be conservative near humans and obstacles.
- Use sonar readings to detect nearby obstacles before planning movement.
- Use detected object angles (horizontal_angle_deg) to orient actions.
- If the scene is ambiguous or the task is unclear, set requires_clarification
  to true and ask a specific question in clarification_question.
- If replanning after a failure, explicitly address the failure in reasoning.
- Keep plans short (≤ 10 steps) and concrete.
"""


# ---------------------------------------------------------------------------
# TaskPlanner
# ---------------------------------------------------------------------------

class TaskPlanner:
    """
    Llama 3 task planning agent (via Ollama local server).

    Parameters
    ----------
    model   : Ollama model tag (default: llama3.2-vision)
    host    : Ollama server URL (default: http://localhost:11434)
    verbose : print LLM responses to stdout
    """

    def __init__(
        self,
        model:   str = DEFAULT_MODEL,
        host:    str = OLLAMA_HOST,
        verbose: bool = True,
    ):
        self._model   = model
        self._host    = host
        self._verbose = verbose

        # --- Lazy-import ollama (defer until after Webots connects) ---
        print("[TaskPlanner] Importing ollama…")
        try:
            import ollama as _ollama
            self._ollama = _ollama
            ollama_available = True
        except ImportError:
            self._ollama = None
            ollama_available = False
            print("[TaskPlanner] WARNING: 'ollama' package not installed. "
                  "Run: pip install ollama")

        # --- Verify the Ollama server is reachable ---
        if ollama_available:
            try:
                # Set a 60s timeout for the client so it doesn't hang forever
                client = _ollama.Client(host=host, timeout=60.0)
                client.list()
                self._client = client
                print(f"[TaskPlanner] Ollama server reachable at {host}")
                print(f"[TaskPlanner] Using model: {model} (60s timeout)")
            except Exception as e:
                self._client = None
                print(f"[TaskPlanner] WARNING: Cannot reach Ollama server at {host}: {e}")
                print("[TaskPlanner] Make sure 'ollama serve' is running — falling back to STUB mode.")
        else:
            self._client = None

        # --- State ---
        self._task:             Optional[str]  = None   # current goal string
        self._prompt_user:      bool           = True   # True → need a task from user
        self._decision_needed:  bool           = False  # True → executor reported failure
        self._failure_context:  Optional[str]  = None   # description of what went wrong
        self._planning:         bool           = False  # True while LLM thread is running
        self._current_plan:     Optional[dict] = None   # latest plan returned by LLM

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Bus integration
    # ------------------------------------------------------------------

    def attach(self, bus) -> None:
        """Subscribe to the scene_state topic on *bus*."""
        bus.subscribe("scene_state", self._on_scene_published)
        bus.subscribe("task_result", self._on_task_result)
        print("[TaskPlanner] Attached to SceneBus.")

    # ------------------------------------------------------------------
    # Public control API (called by executor / external modules)
    # ------------------------------------------------------------------

    def report_failure(self, context: str) -> None:
        """
        Signal that the current plan step failed.
        The next published scene state will trigger a replan.
        """
        with self._lock:
            self._decision_needed = True
            self._failure_context = context
        print(f"[TaskPlanner] Failure reported: {context}")

    def report_success(self) -> None:
        """
        Signal that the current task was completed successfully.
        Resets to waiting-for-user mode.
        """
        with self._lock:
            self._task            = None
            self._prompt_user     = True
            self._decision_needed = False
            self._failure_context = None
            self._current_plan    = None
        print("[TaskPlanner] Task completed. Waiting for next user goal.")

    def get_plan(self) -> Optional[dict]:
        """Return the most recently produced plan, or None."""
        with self._lock:
            return self._current_plan

    def is_planning(self) -> bool:
        """True while the background LLM call is in progress."""
        with self._lock:
            return self._planning

    def is_waiting_for_user(self) -> bool:
        """True when the planner needs the operator to provide a task."""
        with self._lock:
            return self._prompt_user

    # ------------------------------------------------------------------
    # Bus callbacks (called from the publisher's thread)
    # ------------------------------------------------------------------

    def _on_scene_published(self, state: dict, snapshot_path: str) -> None:
        """Called every time a scene_state is published on the bus."""
        with self._lock:
            if self._planning:
                return   # already thinking — don't stack calls

            need_decision = self._decision_needed
            need_user     = self._prompt_user
            task          = self._task
            failure_ctx   = self._failure_context

        if need_decision:
            # Executor reported a failure — replan in background
            print("\n[TaskPlanner] ⚡ Decision needed — starting replan…")
            with self._lock:
                self._decision_needed = False
                self._failure_context = None
            self._start_planning_thread(task, state, snapshot_path, failure_ctx)

        elif need_user:
            # No active task — ask the operator in a background thread
            print("\n[TaskPlanner] 🤔 No active task — prompting operator…")
            self._start_user_input_thread(state, snapshot_path)

        # else: task is running fine, no LLM call needed

    def _on_task_result(self, success: bool, context: str) -> None:
        """Called when the executor publishes a task_result."""
        if success:
            self.report_success()
        else:
            self.report_failure(context)

    # ------------------------------------------------------------------
    # Threading helpers
    # ------------------------------------------------------------------

    def _start_user_input_thread(self, state: dict, snapshot_path: str) -> None:
        with self._lock:
            self._planning    = True
            self._prompt_user = False   # prevent re-triggering while waiting

        t = threading.Thread(
            target=self._user_input_worker,
            args=(state, snapshot_path),
            daemon=True,
            name="TaskPlanner-UserInput",
        )
        t.start()

    def _user_input_worker(self, state: dict, snapshot_path: str) -> None:
        """Runs in background: reads goal from operator, then calls LLM."""
        try:
            print("\n" + "=" * 60)
            print("  NAO TASK PLANNER — Enter your goal for the robot")
            print("  (The robot is ready and awaiting instructions)")
            print("=" * 60)
            task = input("  Goal > ").strip()

            if not task:
                print("[TaskPlanner] No goal entered. Waiting for next scene state.")
                with self._lock:
                    self._planning    = False
                    self._prompt_user = True   # re-enable prompt next cycle
                return

            with self._lock:
                self._task = task

            print(f"[TaskPlanner] Goal set: '{task}'")
            self._do_plan(task, state, snapshot_path, failure_context=None)

        except Exception as exc:
            print(f"[TaskPlanner] User input error: {exc}")
            with self._lock:
                self._planning    = False
                self._prompt_user = True

    def _start_planning_thread(
        self,
        task:            Optional[str],
        state:           dict,
        snapshot_path:   str,
        failure_context: Optional[str],
    ) -> None:
        with self._lock:
            self._planning = True

        t = threading.Thread(
            target=self._do_plan,
            args=(task, state, snapshot_path, failure_context),
            daemon=True,
            name="TaskPlanner-LLM",
        )
        t.start()

    # ------------------------------------------------------------------
    # Planning logic
    # ------------------------------------------------------------------

    def _do_plan(
        self,
        task:            Optional[str],
        state:           dict,
        snapshot_path:   str,
        failure_context: Optional[str],
    ) -> None:
        """Calls the LLM and stores the result.  Always runs in a thread."""
        t0 = time.time()
        try:
            plan = self._call_llama(task, state, snapshot_path, failure_context)
            elapsed = time.time() - t0

            with self._lock:
                self._current_plan = plan
                self._planning     = False

            self._print_plan(plan, elapsed)

        except Exception as exc:
            print(f"[TaskPlanner] LLM call failed: {exc}")
            with self._lock:
                self._planning = False

    # ------------------------------------------------------------------
    # Llama / Ollama API call
    # ------------------------------------------------------------------

    def _call_llama(
        self,
        task:            Optional[str],
        state:           dict,
        snapshot_path:   str,
        failure_context: Optional[str],
    ) -> dict:
        """
        Build and send the request to the local Ollama/Llama server.
        Returns a parsed plan dict.
        """
        if self._client is None:
            return self._stub_plan(task, state)

        messages = self._build_messages(task, state, snapshot_path, failure_context)

        print(f"[TaskPlanner] Calling {self._model} via Ollama…")
        response = self._client.chat(
            model=self._model,
            messages=messages,
            format="json",          # force JSON output mode
            options={
                "temperature":  0.2,
                "num_predict":  2048,
            },
        )

        raw = response.message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[TaskPlanner] Could not parse Llama response as JSON: {e}")
            print(f"[TaskPlanner] Raw response: {raw[:300]}…")
            return {"raw_response": raw, "plan": [], "reasoning": "Parse error."}

    def _build_messages(
        self,
        task:            Optional[str],
        state:           dict,
        snapshot_path:   str,
        failure_context: Optional[str],
    ) -> list:
        """
        Build the Ollama messages list.
        System prompt is a separate message; image attached to the user message.
        """
        slim_state = self._slim_state(state)

        # --- User message text ---
        text_parts = []
        if failure_context:
            text_parts.append(
                f"REPLAN REQUEST\n"
                f"The previous plan step failed:\n{failure_context}\n\n"
                f"Please analyse the failure and produce a revised plan.\n"
            )
        text_parts.append(
            f"TASK: {task or '(no task — ask for clarification)'}\n\n"
            f"CURRENT SCENE STATE:\n"
            f"{json.dumps(slim_state, indent=2)}"
        )
        user_text = "\n".join(text_parts)

        # --- Build message list ---
        user_msg: dict = {"role": "user", "content": user_text}

        # Snapshot disabled per user request to speed up testing
        # if snapshot_path and os.path.isfile(snapshot_path):
        #     try:
        #         import base64
        #         from io import BytesIO
        #         from PIL import Image
        #         ...
        #     except Exception as e:
        #         print(f"[TaskPlanner] Could not encode snapshot: {e}")
        # else:
        print("[TaskPlanner] Snapshot disabled — sending text state only.")

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            user_msg,
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slim_state(self, state: dict) -> dict:
        """
        Return a token-efficient copy of the state.
        Keeps all high-level fields but truncates the verbose joint list.
        """
        import copy
        slim = copy.deepcopy(state)

        # Summarise joints as a short string instead of 25 individual floats
        joints = slim.get("robot", {}).get("joint_positions", {})
        if joints:
            summary = {k: round(v, 2) for k, v in joints.items() if v is not None}
            slim["robot"]["joint_positions"] = summary

        return slim


    def _stub_plan(self, task: Optional[str], state: dict) -> dict:
        """Returns a placeholder plan when Ollama is unavailable."""
        objects = state.get("scene", {}).get("objects", [])
        obj_names = [o["label"] for o in objects]
        return {
            "reasoning": (
                f"[STUB — Ollama not connected] "
                f"Task: '{task}'. "
                f"Detected objects: {obj_names}. "
                f"Make sure 'ollama serve' is running and '{self._model}' is pulled."
            ),
            "task_summary": task or "Unknown task",
            "plan": [
                {
                    "step": 1,
                    "action": "stub_action",
                    "parameters": {},
                    "description": (
                        f"STUB: Run 'ollama pull {self._model}' then 'ollama serve'."
                    ),
                }
            ],
            "requires_clarification": False,
            "clarification_question": None,
            "estimated_duration_s": 0,
        }

    def _print_plan(self, plan: dict, elapsed: float) -> None:
        """Pretty-print the plan to stdout."""
        BOLD   = "\033[1m"
        CYAN   = "\033[36m"
        YELLOW = "\033[33m"
        GREEN  = "\033[32m"
        DIM    = "\033[2m"
        RESET  = "\033[0m"

        sep = f"{CYAN}{'─' * 60}{RESET}"
        print(f"\n{sep}")
        print(f"{BOLD}  TASK PLANNER — New Plan  {DIM}({elapsed:.1f}s){RESET}")
        print(sep)

        print(f"\n{BOLD}Task:{RESET}     {plan.get('task_summary', '?')}")
        print(f"{BOLD}Duration:{RESET} ~{plan.get('estimated_duration_s', '?')}s")

        if plan.get("requires_clarification"):
            print(f"\n{YELLOW}⚠ Clarification needed:{RESET} "
                  f"{plan.get('clarification_question')}")

        print(f"\n{BOLD}Reasoning:{RESET}")
        for line in (plan.get("reasoning") or "").split(". "):
            if line.strip():
                print(f"  • {line.strip()}.")

        steps = plan.get("plan", [])
        if steps:
            print(f"\n{BOLD}Steps:{RESET}")
            for s in steps:
                print(
                    f"  {GREEN}[{s.get('step', '?')}]{RESET} "
                    f"{BOLD}{s.get('action', '?')}{RESET}  —  "
                    f"{s.get('description', '')}"
                )
                if s.get("parameters"):
                    print(f"      params: {s['parameters']}")

        if self._verbose:
            print(f"\n{DIM}Full JSON:\n{json.dumps(plan, indent=2)}{RESET}")

        print(f"\n{sep}\n")
