"""
task_planner.py — Task planner for PlanExecutor.

The TaskPlanner listens for scene updates from the SceneBus and asks either
Ollama or Google Gemini to create robot action plans.

The planner calls the LLM when:
- there is no active task and the operator needs to enter a goal
- the executor reports a failure and the planner needs to replan

The LLM runs in a background thread so the Webots robot.step loop is not blocked.
"""

import os
import json
import threading
import time
from typing import Optional


try:
    from dotenv import load_dotenv

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path, override=True)
except ImportError:
    pass


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

backend_name = (os.getenv("TASK_PLANNER_BACKEND") or "ollama").strip().lower()
if backend_name not in ("ollama", "gemini"):
    backend_name = "ollama"

TASK_PLANNER_BACKEND = backend_name

MAX_JOINT_CHARS = 400


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
You MUST use ONLY these exact action names. Do not invent new action names.

Primitive:
  turn_left        { "degrees": <number> }   ← use this, NOT "adjust_orientation"
  turn_right       { "degrees": <number> }
  move_forward     { "meters": <number> }    ← parameter is "meters", NOT "distance_m"
  move_backward    { "meters": <number> }
  stop             {}
  set_head_yaw     { "angle": <radians> }
  set_head_pitch   { "angle": <radians> }
  wave             { "cycles": <int> }

Semantic (use when objects are detected):
  look_for_object      { "label": "<coco_name>", "timeout_s": <number> }
  center_on_object     { "label": "<coco_name>", "tolerance": 0.1, "timeout_s": 5 }
  move_toward_object   { "label": "<coco_name>", "stop_distance_m": 0.45, "timeout_s": 24 }
  pick_object          { "label": "<coco_name>" }
  place_object         {}

CRITICAL RULES:
- Never use "adjust_orientation", "rotate", "navigate_to", or any name not in the list above.
- To turn, use turn_left or turn_right with a "degrees" parameter.
- To move, use move_forward with a "meters" parameter (not "distance_m").
- You cannot navigate to named rooms — you can only turn and move forward.
- If no objects are detected, use only primitive actions.
- For furniture goals, map common words to COCO labels when needed:
    "table" or "wood table" → "dining table".
- If the target object is visible in scene.objects, prefer object-guided actions
    like center_on_object and move_toward_object instead of a blind fixed-distance move.
- Do not invent a wall or room orientation if the object is already visible; use
    the detected object label and the scene state.
- For pickup or approach tasks involving a visible object, the default plan should be:
        1. center_on_object
        2. move_toward_object
        3. pick_object (only if the goal implies grasping)
    Use move_forward only when the target object is not visible or not detectable.
- If the task mentions a visible object by a common name, resolve it to the COCO
    label first, then plan against that label.
- Example pickup goals:
    "pick up the cup" → center_on_object(cup), move_toward_object(cup), pick_object(cup)
    "grab the bottle" → center_on_object(bottle), move_toward_object(bottle), pick_object(bottle)

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
- scene.objects[]   : label, confidence, bounding_box, depth_distance_m / distance_m,
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


class TaskPlanner:
    """
    Background task planner for the NAO robot.

    The backend is selected with TASK_PLANNER_BACKEND:
    - "ollama" uses a local Ollama model
    - "gemini" uses Google Gemini through gemini_llm_connector
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        verbose: bool = True,
    ):
        self._model = model
        self._host = host
        self._verbose = verbose
        self._backend = TASK_PLANNER_BACKEND
        self._client = None
        self._gemini_available = False

        if self._backend == "gemini":
            try:
                from gemini_llm_connector import gemini_available

                self._gemini_available = bool(gemini_available())
            except ImportError:
                self._gemini_available = False

            gemini_model = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()

            if self._gemini_available:
                print(
                    f"[TaskPlanner] Backend: Gemini | model={gemini_model!r} "
                    f"(GEMINI_API_KEY / GOOGLE_API_KEY)"
                )
            else:
                print(
                    "[TaskPlanner] TASK_PLANNER_BACKEND=gemini but Gemini is unavailable "
                    "(install google-genai, set GEMINI_API_KEY). Falling back to STUB."
                )

        else:
            print("[TaskPlanner] Importing ollama…")

            try:
                import ollama as ollama_module

                self._ollama = ollama_module
                ollama_available = True
            except ImportError:
                self._ollama = None
                ollama_available = False
                print(
                    "[TaskPlanner] WARNING: 'ollama' package not installed. "
                    "Run: pip install ollama"
                )

            if ollama_available:
                try:
                    ollama_client = ollama_module.Client(host=host, timeout=60.0)
                    ollama_client.list()
                    self._client = ollama_client
                    print(
                        f"[TaskPlanner] Backend: Ollama | host={host} | "
                        f"model={model} (60s timeout)"
                    )
                except Exception as exc:
                    self._client = None
                    print(
                        f"[TaskPlanner] WARNING: Cannot reach Ollama server at {host}: {exc}"
                    )
                    print(
                        "[TaskPlanner] Make sure 'ollama serve' is running — "
                        "falling back to STUB mode."
                    )
            else:
                self._client = None

        self._task: Optional[str] = None
        self._prompt_user: bool = True
        self._decision_needed: bool = False
        self._failure_context: Optional[str] = None
        self._planning: bool = False
        self._current_plan: Optional[dict] = None

        self._lock = threading.Lock()

    def attach(self, bus) -> None:
        """Subscribe to scene and task-result updates."""
        bus.subscribe("scene_state", self._on_scene_published)
        bus.subscribe("task_result", self._on_task_result)
        print("[TaskPlanner] Attached to SceneBus.")

    def report_failure(self, context: str) -> None:
        """Request a replan using the provided failure context."""
        with self._lock:
            self._decision_needed = True
            self._failure_context = context

        print(f"[TaskPlanner] Failure reported: {context}")

    def report_success(self) -> None:
        """Reset the planner after a task completes."""
        with self._lock:
            self._task = None
            self._prompt_user = True
            self._decision_needed = False
            self._failure_context = None
            self._current_plan = None

        print("[TaskPlanner] Task completed. Waiting for next user goal.")

    def get_plan(self) -> Optional[dict]:
        """Return the latest available plan."""
        with self._lock:
            return self._current_plan

    def consume_plan(self) -> None:
        """Clear the current plan after the executor accepts it."""
        with self._lock:
            self._current_plan = None

    def is_planning(self) -> bool:
        """Return True while a background planning call is running."""
        with self._lock:
            return self._planning

    def is_waiting_for_user(self) -> bool:
        """Return True when the planner needs a new operator goal."""
        with self._lock:
            return self._prompt_user

    def _on_scene_published(self, state: dict, snapshot_path: str) -> None:
        """Handle a new scene_state message from the SceneBus."""
        with self._lock:
            if self._planning:
                return

            decision_needed = self._decision_needed
            prompt_user = self._prompt_user
            current_task = self._task
            failure_context = self._failure_context

        if decision_needed:
            print("\n[TaskPlanner] ⚡ Decision needed — starting replan…")

            with self._lock:
                self._decision_needed = False
                self._failure_context = None

            self._start_planning_thread(
                current_task,
                state,
                snapshot_path,
                failure_context,
            )

        elif prompt_user:
            print("\n[TaskPlanner] 🤔 No active task — prompting operator…")
            self._start_user_input_thread(state, snapshot_path)

    def _on_task_result(self, success: bool, context: str) -> None:
        """Handle executor success or failure reports."""
        if success:
            self.report_success()
        else:
            self.report_failure(context)

    def _start_user_input_thread(self, state: dict, snapshot_path: str) -> None:
        """Start a background thread that asks the operator for a task."""
        with self._lock:
            self._planning = True
            self._prompt_user = False

        user_input_thread = threading.Thread(
            target=self._user_input_worker,
            args=(state, snapshot_path),
            daemon=True,
            name="TaskPlanner-UserInput",
        )
        user_input_thread.start()

    def _user_input_worker(self, state: dict, snapshot_path: str) -> None:
        """Read an operator goal, then build the first plan."""
        try:
            print("\n" + "=" * 60)
            print("  NAO TASK PLANNER — Enter your goal for the robot")
            print("  (The robot is ready and awaiting instructions)")
            print("=" * 60)

            task = input("  Goal > ").strip()

            if not task:
                print("[TaskPlanner] No goal entered. Waiting for next scene state.")
                with self._lock:
                    self._planning = False
                    self._prompt_user = True
                return

            with self._lock:
                self._task = task

            print(f"[TaskPlanner] Goal set: '{task}'")
            self._do_plan(task, state, snapshot_path, failure_context=None)

        except Exception as exc:
            print(f"[TaskPlanner] User input error: {exc}")
            with self._lock:
                self._planning = False
                self._prompt_user = True

    def _start_planning_thread(
        self,
        task: Optional[str],
        state: dict,
        snapshot_path: str,
        failure_context: Optional[str],
    ) -> None:
        """Start a background LLM planning thread."""
        with self._lock:
            self._planning = True

        planning_thread = threading.Thread(
            target=self._do_plan,
            args=(task, state, snapshot_path, failure_context),
            daemon=True,
            name="TaskPlanner-LLM",
        )
        planning_thread.start()

    def _do_plan(
        self,
        task: Optional[str],
        state: dict,
        snapshot_path: str,
        failure_context: Optional[str],
    ) -> None:
        """Call the selected backend and store the resulting plan."""
        start_time = time.time()

        try:
            plan = self._call_llm(task, state, snapshot_path, failure_context)
            plan = self._postprocess_plan(plan, task, state)
            elapsed = time.time() - start_time

            with self._lock:
                self._current_plan = plan
                self._planning = False

            self._print_plan(plan, elapsed)

        except Exception as exc:
            print(f"[TaskPlanner] LLM call failed: {exc}")
            with self._lock:
                self._planning = False

    def _build_user_text(
        self,
        task: Optional[str],
        state: dict,
        failure_context: Optional[str],
    ) -> str:
        """Build the user message sent to Ollama or Gemini."""
        slim_state = self._slim_state(state)
        message_parts = []

        if failure_context:
            message_parts.append(
                f"REPLAN REQUEST\n"
                f"The previous plan step failed:\n{failure_context}\n\n"
                f"Please analyse the failure and produce a revised plan.\n"
            )

        message_parts.append(
            f"TASK: {task or '(no task — ask for clarification)'}\n\n"
            f"CURRENT SCENE STATE:\n"
            f"{json.dumps(slim_state, indent=2)}"
        )

        return "\n".join(message_parts)

    def _call_llm(
        self,
        task: Optional[str],
        state: dict,
        snapshot_path: str,
        failure_context: Optional[str],
    ) -> dict:
        """Call the configured backend and return a parsed plan dictionary."""
        if self._backend == "gemini" and self._gemini_available:
            return self._call_gemini(task, state, snapshot_path, failure_context)

        if self._client is None:
            return self._stub_plan(task, state)

        messages = self._build_messages(task, state, snapshot_path, failure_context)

        print(f"[TaskPlanner] Calling {self._model} via Ollama…")

        response = self._client.chat(
            model=self._model,
            messages=messages,
            format="json",
            options={
                "temperature": 0.2,
                "num_predict": 2048,
            },
        )

        raw_response = response.message.content

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as exc:
            print(f"[TaskPlanner] Could not parse Llama response as JSON: {exc}")
            print(f"[TaskPlanner] Raw response: {raw_response[:300]}…")
            return {
                "raw_response": raw_response,
                "plan": [],
                "reasoning": "Parse error.",
            }

    def _call_gemini(
        self,
        task: Optional[str],
        state: dict,
        snapshot_path: str,
        failure_context: Optional[str],
    ) -> dict:
        """Call Gemini and return a task plan dictionary."""
        from gemini_llm_connector import extract_json_plan, gemini_generate_content

        user_text = self._build_user_text(task, state, failure_context)

        snapshot_file = (
            snapshot_path
            if snapshot_path and os.path.isfile(snapshot_path)
            else None
        )

        if snapshot_file:
            print(
                f"[TaskPlanner] Gemini: attaching snapshot "
                f"{os.path.basename(snapshot_file)}"
            )
        else:
            print("[TaskPlanner] Gemini: text-only (no snapshot file on disk)")

        try:
            raw_response = gemini_generate_content(
                user_text,
                system_instruction=_SYSTEM_PROMPT,
                snapshot_path=snapshot_file,
            )
        except Exception as exc:
            print(f"[TaskPlanner] Gemini request failed: {exc}")
            return self._stub_plan(task, state)

        plan = extract_json_plan(raw_response)

        if not isinstance(plan, dict) or "plan" not in plan:
            print("[TaskPlanner] Gemini: response was not a valid task plan JSON.")
            print(f"[TaskPlanner] Raw response: {(raw_response or '')[:400]}…")
            return self._stub_plan(task, state)

        return plan

    def _postprocess_plan(self, plan: dict, task: Optional[str], state: dict) -> dict:
        """
        Replace blind movement with object-guided movement when the target is visible.

        This prevents the robot from ignoring useful YOLO detections.
        """
        plan_steps = plan.get("plan", []) if isinstance(plan, dict) else []
        if not plan_steps:
            return plan

        target_label = self._infer_target_label(task, state)
        if target_label is None:
            return plan

        if not self._is_object_visible(state, target_label):
            return plan

        task_text = (task or "").lower()

        wants_grasp = any(
            word in task_text
            for word in ("pick", "grab", "take", "lift", "collect")
        )

        wants_approach = any(
            word in task_text
            for word in ("walk to", "go to", "approach", "move to", "reach")
        )

        if not wants_grasp and not wants_approach:
            return plan

        guided_steps = [
            {
                "step": 1,
                "action": "move_toward_object",
                "parameters": {
                    "label": target_label,
                    "stop_distance_m": 0.40,
                    "timeout_s": 24,
                },
                "description": f"Move toward the visible {target_label}.",
            },
        ]

        if wants_grasp:
            guided_steps.append(
                {
                    "step": 2,
                    "action": "pick_object",
                    "parameters": {"label": target_label},
                    "description": f"Pick up the {target_label}.",
                }
            )

        revised_plan = dict(plan)
        revised_plan["plan"] = guided_steps

        existing_reasoning = revised_plan.get("reasoning", "") or ""
        postprocess_note = (
            f"Visible target '{target_label}' detected, so the plan was "
            f"converted to object-guided motion."
        )

        revised_plan["reasoning"] = (
            f"{existing_reasoning} {postprocess_note}".strip()
        )

        return revised_plan

    def _infer_target_label(self, task: Optional[str], state: dict) -> Optional[str]:
        """Infer the COCO label that best matches the operator's task."""
        task_text = (task or "").lower()

        label_aliases = {
            "dining table": ("wood table", "wooden table", "table"),
            "couch": ("couch", "sofa"),
            "bottle": ("bottle",),
            "chair": ("chair",),
            "cup": ("cup",),
            "book": ("book",),
            "laptop": ("laptop",),
            "person": ("person",),
        }

        for canonical_label, aliases in label_aliases.items():
            if any(alias in task_text for alias in aliases):
                return canonical_label

        visible_objects = state.get("scene", {}).get("objects") or state.get("objects", [])

        for detected_object in visible_objects:
            detected_label = str(detected_object.get("label", "")).strip().lower()
            if detected_label and detected_label in task_text:
                return detected_label

        return None

    def _is_object_visible(self, state: dict, label: str) -> bool:
        """Return True if the requested object label is currently visible."""
        target_label = label.strip().lower()
        visible_objects = state.get("scene", {}).get("objects") or state.get("objects", [])

        for detected_object in visible_objects:
            detected_label = str(detected_object.get("label", "")).strip().lower()
            if detected_label == target_label:
                return True

        return False

    def _build_messages(
        self,
        task: Optional[str],
        state: dict,
        snapshot_path: str,
        failure_context: Optional[str],
    ) -> list:
        """Build the Ollama chat message list."""
        user_text = self._build_user_text(task, state, failure_context)
        user_message = {"role": "user", "content": user_text}

        print("[TaskPlanner] Ollama: text-only user message.")

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            user_message,
        ]

    def _slim_state(self, state: dict) -> dict:
        """Return a smaller copy of the scene state for the LLM prompt."""
        import copy

        slim_state = copy.deepcopy(state)

        joint_positions = slim_state.get("robot", {}).get("joint_positions", {})
        if joint_positions:
            rounded_joint_positions = {
                joint_name: round(joint_angle, 2)
                for joint_name, joint_angle in joint_positions.items()
                if joint_angle is not None
            }
            slim_state["robot"]["joint_positions"] = rounded_joint_positions

        return slim_state

    def _stub_plan(self, task: Optional[str], state: dict) -> dict:
        """Build a safe fallback plan when no LLM backend is available."""
        task_text = (task or "").lower()
        visible_objects = state.get("scene", {}).get("objects") or state.get("objects", [])

        visible_object_labels = [
            str(detected_object.get("label", "")).lower()
            for detected_object in visible_objects
            if detected_object.get("label")
        ]

        target_label = self._infer_target_label(task, state)

        visible_target_label = (
            target_label
            if target_label and self._is_object_visible(state, target_label)
            else None
        )

        if visible_target_label:
            plan_steps = [
                {
                    "step": 1,
                    "action": "move_toward_object",
                    "parameters": {
                        "label": visible_target_label,
                        "stop_distance_m": 0.40,
                        "timeout_s": 24,
                    },
                    "description": f"Move toward the visible {visible_target_label}.",
                },
            ]

            if any(
                word in task_text
                for word in ("pick", "grab", "take", "lift", "collect")
            ):
                plan_steps.append(
                    {
                        "step": 2,
                        "action": "pick_object",
                        "parameters": {"label": visible_target_label},
                        "description": f"Pick up the {visible_target_label}.",
                    }
                )

            task_summary = f"Move toward the visible {visible_target_label}."

            reasoning = (
                f"[FALLBACK — Ollama not connected] Task: '{task}'. "
                f"Detected objects: {visible_object_labels}. "
                f"Using the visible target '{visible_target_label}' with supported "
                f"executor actions."
            )

        else:
            plan_steps = [
                {
                    "step": 1,
                    "action": "turn_left",
                    "parameters": {"degrees": 20},
                    "description": "Turn left to scan for the target.",
                },
                {
                    "step": 2,
                    "action": "move_forward",
                    "parameters": {"meters": 0.5},
                    "description": "Move forward a short distance toward the goal area.",
                },
                {
                    "step": 3,
                    "action": "stop",
                    "parameters": {},
                    "description": "Stop and wait for the next scene update.",
                },
            ]

            task_summary = task or "Unknown task"

            reasoning = (
                f"[FALLBACK — Ollama not connected] Task: '{task}'. "
                f"Detected objects: {visible_object_labels}. "
                f"No visible target was found, so the robot will make a conservative "
                f"blind approach."
            )

        return {
            "reasoning": reasoning,
            "task_summary": task_summary,
            "plan": plan_steps,
            "requires_clarification": False,
            "clarification_question": None,
            "estimated_duration_s": 8,
        }

    def _print_plan(self, plan: dict, elapsed: float) -> None:
        """Print the generated plan."""
        bold = "\033[1m"
        cyan = "\033[36m"
        yellow = "\033[33m"
        green = "\033[32m"
        dim = "\033[2m"
        reset = "\033[0m"

        separator = f"{cyan}{'─' * 60}{reset}"

        print(f"\n{separator}")
        print(f"{bold}  TASK PLANNER — New Plan  {dim}({elapsed:.1f}s){reset}")
        print(separator)

        print(f"\n{bold}Task:{reset}     {plan.get('task_summary', '?')}")
        print(f"{bold}Duration:{reset} ~{plan.get('estimated_duration_s', '?')}s")

        if plan.get("requires_clarification"):
            print(
                f"\n{yellow}⚠ Clarification needed:{reset} "
                f"{plan.get('clarification_question')}"
            )

        print(f"\n{bold}Reasoning:{reset}")
        for reasoning_line in (plan.get("reasoning") or "").split(". "):
            if reasoning_line.strip():
                print(f"  • {reasoning_line.strip()}.")

        steps = plan.get("plan", [])
        if steps:
            print(f"\n{bold}Steps:{reset}")
            for step in steps:
                print(
                    f"  {green}[{step.get('step', '?')}]{reset} "
                    f"{bold}{step.get('action', '?')}{reset}  —  "
                    f"{step.get('description', '')}"
                )
                if step.get("parameters"):
                    print(f"      params: {step['parameters']}")

        if self._verbose:
            print(f"\n{dim}Full JSON:\n{json.dumps(plan, indent=2)}{reset}")

        print(f"\n{separator}\n")