"""Build Ollama/Gemini chat and parse a single JSON tool action for `simple_controller`.

The long `_SYSTEM_PROMPT` is course-specific behavior; JSON extraction follows common
chat-model habits (markdown ```json fences or a raw `{...}` block).

References (external clients used for LLM calls):
- Ollama server + models: https://github.com/ollama/ollama
- Ollama Python client (`import ollama`): https://github.com/ollama/ollama-python
- Pillow (optional JPEG for vision chat): https://github.com/python-pillow/Pillow
"""

import os
import re
import json
import base64
import threading
from io import BytesIO
from typing import List, Optional

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

_SYSTEM_PROMPT = """You are the planning brain of a NAO humanoid robot in Webots.
You MUST pass valid tool parameters in every JSON response (see PASSING PARAMETERS below).

ALLOWED ACTIONS (exact "action" string):
- "locate_object": Navigation / search goals — REQUIRED FIRST. Fields: "aliases" (required): non-empty list of
  strings (YOLO/COCO names and synonyms, e.g. ["dog","puppy"]). The robot performs a **360° spin** with **head level
  (neutral)** while checking detections; the spin sweeps the room. You may then get "FOUND: …" after a stable sighting,
  or "TIMEOUT: …" if nothing matched. While spinning, context may also include "OBJECT_IN_VIEW:" when YOLO sees the
  target in the current frame.
- "move_forward": Field "meters" (required number). Use **generous** steps: prefer **0.55–0.9** m per call
  (typical 0.6–0.75) so the robot covers ground quickly; the runtime may bump small values to a minimum stride.
  One straight segment per response. After locate_object for nav goals.
- "turn_degrees": Field "degrees" (required number). POSITIVE = turn left, NEGATIVE = turn right.
- "move_to_object": Field "aliases" (required). Vision + **RangeFinder depth** guided walk. Use ONLY after
  "OBJECT_IN_VIEW:" and/or "FOUND:" in CURRENT CONTEXT. The robot aims to get within about **0.21 m** of the target when depth distance_m is reliable and the target is roughly centered (**SUPER_CLOSE:**). **Head pitch is
  not an LLM action** — during approach the controller tilts the head **only** from the target's **vertical position in
  the image** (above/below center); head yaw stays **0**. While walking, you will receive **APPROACH_CHECKPOINT:** about
  every **10 seconds** (feet paused, head still tracking vertically, sonar + depth in context). On a checkpoint: return
  **move_to_object** with the **same aliases** to continue, or **turn_degrees** / **move_forward** (keep **≤0.5 m**
  for dodge) to avoid obstacles — **one** JSON action per response. After a dodge step finishes (**STEP_DONE** /
  **APPROACH_INTERRUPT**), return **move_to_object** with the **same aliases** again to resume the approach. The
  controller auto-completes the goal on SUPER_CLOSE (no separate "done" needed for that path).
- "done": Task succeeded when the goal does not end with an automatic SUPER_CLOSE (e.g. non-approach goals), or
  if you judge success from the image before SUPER_CLOSE fires.
- "fail": ONLY when absolutely impossible. Prefer locate_object + different moves first.
- "clarify": Field "question" (required string) if you truly need user input.

PASSING PARAMETERS (every response):
- Always include "reason": one short string explaining the choice.
- "locate_object" and "move_to_object" MUST include "aliases": ["…"] with at least one non-empty string.
- "move_forward" MUST include "meters": <positive number>.
- "turn_degrees" MUST include "degrees": <number>.
- "clarify" MUST include "question".
- Output exactly ONE JSON object per turn — no markdown fences, no extra keys beyond what the action needs.
- **Depth:** In user messages, **distance_m** is **meters** from the **RangeFinder** (depth sensor) sampled at the
  YOLO bounding box — that is the authoritative distance for approach and **SUPER_CLOSE** (~0.61 m / ~2 ft). The field
  **distance** is a coarse **visual size bucket** from the 2D box, **not** meters — do not confuse it with depth.

SEARCH + SIGNALS:
- After locate_object, expect spin then FOUND or TIMEOUT; OBJECT_IN_VIEW may appear anytime YOLO sees the target.
- While searching **after the spin** (between moves, still before FOUND), issue ONE primitive per step: **move_forward**,
  **turn_degrees**, or **locate_object** again. Head stays **neutral**; you cannot command look_up/look_down (not supported).
- STEP_DONE may include FEEDBACK: (target vs image center). During move_to_object expect **APPROACH_CHECKPOINT** (~10 s)
  for safety replanning (path clear vs dodge). **STUCK:** means no progress for a long time — suggest turn then
  re-approach. SUPER_CLOSE ends the run automatically.

RECOVERY:
- On TIMEOUT or LOST, prefer ``locate_object`` again, ``move_forward``, or ``turn_degrees`` while the head stays neutral.
- Use "fail" only as a last resort.

STRICT RULES:
1. First step for go-to / find / approach goals: locate_object with good aliases (for laptops include synonyms like "laptop","computer","keyboard").
2. **Never** output ``look_up`` or ``look_down`` — they are **disabled**. Head pitch follows the target bbox vertically only; you control motion with locate_object / move_forward / turn_degrees / move_to_object / done / fail / clarify.
3. Do not use move_to_object until OBJECT_IN_VIEW and/or FOUND has appeared in context.
4. For approach goals, **move_to_object** ends automatically on **SUPER_CLOSE** (~2 ft / ~0.61 m depth when centered, or large bbox).
  On **APPROACH_CHECKPOINT**, prefer **move_to_object** with unchanged aliases if the path is clear; otherwise dodge
  (short **move_forward** ≤0.5 m, **turn_degrees**). Use **done** for non-approach goals.
5. For **go to / approach** goals, plan until the robot is within about **0.21 m** of the target; NEVER return "done" during move_to_object unless abandoning the task, let SUPER_CLOSE end it automatically.
6. Follow PASSING PARAMETERS exactly.

JSON EXAMPLES:
{"action":"locate_object","aliases":["dog","puppy"],"reason":"Spin search for dog."}
{"action":"locate_object","aliases":["laptop","computer","keyboard"],"reason":"Search for laptop on floor or desk."}
{"action":"move_forward","meters":0.65,"reason":"Timeout after spin; advance into room."}
{"action":"move_to_object","aliases":["dog"],"reason":"APPROACH_CHECKPOINT clear; continue same aliases."}
{"action":"done","reason":"Non-approach goal satisfied (or success before SUPER_CLOSE)."}
{"action":"fail","reason":"No safe path after exhaustive retries."}
"""


_GOAL_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "go",
        "get",
        "and",
        "for",
        "nao",
        "robot",
        "walk",
        "move",
        "head",
        "toward",
        "towards",
        "near",
        "find",
        "locate",
        "approach",
        "please",
        "me",
        "from",
        "with",
        "that",
        "this",
        "your",
        "its",
        "our",
        "use",
        "then",
        "when",
        "how",
        "can",
        "you",
        "let",
        "make",
        "just",
    }
)  # ignore these when matching goal words to labels


def _label_matches_user_goal(goal: str, label: str) -> bool:
    # Compare YOLO class name to words in the user's goal text.
    label_normalized = (label or "").strip().lower()
    goal_lower = (goal or "").lower()

    if not label_normalized or not goal_lower:
        return False

    # Whole label appears inside the goal sentence.
    if label_normalized in goal_lower:
        return True

    # Otherwise try each word from the goal (skip tiny words and stopwords).
    goal_with_spaces = goal_lower.replace(",", " ").replace(".", " ")
    for word_token in goal_with_spaces.split():
        word_token = word_token.strip("?!'\"")

        if len(word_token) < 3 or word_token in _GOAL_STOPWORDS:
            continue

        if word_token == label_normalized or word_token in label_normalized or label_normalized in word_token:
            return True

    return False


def _target_distance_summary(goal: str, objs: List[dict]) -> str:
    summary_lines: List[str] = []

    for scene_object in objs:
        object_label = scene_object.get("label")

        if not object_label or not _label_matches_user_goal(goal, str(object_label)):
            continue  # not the object the user asked about

        raw_depth = scene_object.get("distance_m")
        try:
            depth_meters = float(raw_depth) if raw_depth is not None else None
        except (TypeError, ValueError):
            depth_meters = None

        screen_position = scene_object.get("position", "?")

        if depth_meters is not None:
            summary_lines.append(
                f"  - {object_label}: distance_m={depth_meters:.3f} m (RangeFinder @ YOLO bbox), position={screen_position}"
            )
        else:
            summary_lines.append(
                f"  - {object_label}: distance_m=n/a (no RangeFinder sample for this box), position={screen_position}"
            )

    if not summary_lines:
        return ""

    return (
        "TARGET DEPTH (goal-relevant rows — use **distance_m** in meters for range, not the `distance` bucket):\n"
        + "\n".join(summary_lines)
        + "\n"
    )


def build_planner_user_text(goal: str, scene_state: dict, context: str) -> str:
    visible_objects = scene_state.get("objects", [])

    object_lines: List[str] = []
    for scene_object in visible_objects:
        height_frac = scene_object.get("height_frac")
        height_suffix = f", height_frac={height_frac}" if height_frac is not None else ""

        cx_norm = scene_object.get("cx_norm")
        cy_norm = scene_object.get("cy_norm")
        bbox_center_suffix = ""
        if cx_norm is not None and cy_norm is not None:
            bbox_center_suffix = f", cx_norm={cx_norm}, cy_norm={cy_norm}"

        raw_depth = scene_object.get("distance_m")
        try:
            depth_meters = float(raw_depth) if raw_depth is not None else None
        except (TypeError, ValueError):
            depth_meters = None

        depth_suffix = f", distance_m={depth_meters:.3f}" if depth_meters is not None else ", distance_m=n/a"

        object_lines.append(
            f"- {scene_object['label']}: position={scene_object['position']}, distance={scene_object['distance']}"
            f"{depth_suffix}{height_suffix}{bbox_center_suffix}"
        )

    objects_block = "\n".join(object_lines) if object_lines else "(None)"
    target_depth_block = _target_distance_summary(goal, visible_objects)

    goal_lower = goal.lower()
    floor_scan_reminder = ""

    low_object_keywords = (
        "laptop",
        "computer",
        "keyboard",
        "floor",
        "ground",
        "low",
        "desk",
        "table",
        "bag",
        "backpack",
        "phone",
        "book",
    )
    if any(keyword in goal_lower for keyword in low_object_keywords):
        floor_scan_reminder = (
            "\nLOW-OBJECT NOTE: The goal may involve something low. The head stays **level** during search; "
            "after the target is visible, **move_to_object** handles approach while head pitch tracks the bbox vertically — "
            "you cannot command look_down.\n"
        )

    context_upper = (context or "").upper()
    if not object_lines and ("TIMEOUT" in context_upper or "LOST" in context_upper):
        floor_scan_reminder += (
            "\nNO TARGET IN VISIBLE OBJECTS after search trouble — prefer **locate_object**, **move_forward**, or "
            "**turn_degrees** (head stays neutral).\n"
        )

    range_finder_legend = (
        "RANGE_FINDER: **distance_m** is true depth in **meters** from the Webots RangeFinder sampled at the "
        "YOLO bbox (same geometry as the RGB frame for alignment). It is **not** inferred from RGB appearance. "
        "The **distance** field on each line is a coarse **2D box size bucket**, not meters.\n\n"
    )

    return f"""USER GOAL: {goal}
CURRENT CONTEXT: {context or "Initial planning — no executor status yet"}

{range_finder_legend}{target_depth_block}VISIBLE OBJECTS:
{objects_block}

SONAR: Left={scene_state.get('sonar',{}).get('left_m')}m, Right={scene_state.get('sonar',{}).get('right_m')}m
{floor_scan_reminder}
If the goal is to go to / find / approach an object, your FIRST response MUST be locate_object with a non-empty "aliases" list.
That starts a **360° spin** (head **neutral / level**) plus detection; watch for FOUND:, TIMEOUT:, and OBJECT_IN_VIEW: in CURRENT CONTEXT.
After the spin, explore with **move_forward** and **turn_degrees** (one tool per step). **look_up** and **look_down**
are **not supported** — head stays neutral until approach; then pitch follows the target vertically only.
Then move_to_object after the target is confirmed visible. While approaching, respond to **APPROACH_CHECKPOINT:**
(continue with same **move_to_object** aliases, or dodge with move/turn only). Goal: reach within about **2 ft (~0.61 m)** RangeFinder depth when centered.
Always pass the required JSON fields for each tool (see system prompt). Prefer **0.55–0.9 m** for move_forward when exploring.
Use "done" when the goal is met **without** relying on SUPER_CLOSE (e.g. inspection-only goals). For go-to-object, use move_to_object and let depth / bbox + SUPER_CLOSE finish when within ~**2 ft** (~0.61 m).

What is your next action? Respond in JSON only."""


class SimplePlanner:
    def __init__(self):
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2-vision")

        self._backend = os.getenv("SIMPLE_PLANNER_BACKEND", "ollama").strip().lower()
        self._model = ollama_model
        self._client: Optional[object] = None
        self._lock = threading.Lock()

        self._goal: Optional[str] = None
        self._plan: Optional[dict] = None
        self._planning = False
        self._done = False

        if self._backend == "gemini":
            gemini_model = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()
            self._backend_label = "Gemini"
            self._active_model = gemini_model
        else:
            self._backend_label = "Ollama"
            self._active_model = ollama_model

        if self._backend == "gemini":
            print(
                f"[SimplePlanner] LLM backend: Gemini (Google API) | model={self._active_model}\n"
                f"[SimplePlanner]   (set SIMPLE_PLANNER_BACKEND=gemini; key: GEMINI_API_KEY or GOOGLE_API_KEY)"
            )
        elif _OLLAMA_AVAILABLE:
            try:
                ollama_client = _ollama.Client(host=ollama_host, timeout=300.0)
                ollama_client.list()
                self._client = ollama_client
                print(
                    f"[SimplePlanner] LLM backend: Ollama (local) | model={self._active_model} | host={ollama_host}\n"
                    f"[SimplePlanner]   (use SIMPLE_PLANNER_BACKEND=gemini to use Google Gemini instead)"
                )
            except Exception as connect_error:
                print(f"[SimplePlanner] WARNING: Ollama unavailable: {connect_error}")
        else:
            print(
                "[SimplePlanner] WARNING: Ollama not installed or backend misconfigured.\n"
                f"[SimplePlanner]   SIMPLE_PLANNER_BACKEND={self._backend!r} | expected pip package: ollama"
            )

    def set_goal(self, goal: str):
        with self._lock:
            self._goal = goal
            self._done = False
            self._plan = None
            self._planning = False

    def get_goal(self) -> Optional[str]:
        with self._lock:
            return self._goal

    def is_planning(self) -> bool:
        with self._lock:
            return self._planning

    def has_plan(self) -> bool:
        with self._lock:
            return self._plan is not None

    def consume_plan(self) -> Optional[dict]:
        with self._lock:
            consumed_plan = self._plan
            self._plan = None
            return consumed_plan

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def request_plan(self, scene_state: dict, snapshot_path: Optional[str] = None, context: str = ""):
        with self._lock:
            if self._planning or self._goal is None:
                return
            self._planning = True

        active_goal = self._goal
        planning_thread = threading.Thread(
            target=self._call_llm,
            args=(active_goal, scene_state, snapshot_path, context),
            daemon=True,
        )
        planning_thread.start()

    def _call_llm(self, goal: str, scene_state: dict,
                  snapshot_path: Optional[str], context: str) -> None:
        extracted_json_string = None
        try:
            if self._backend == "gemini":
                from gemini_llm_connector import gemini_plan_from_scene

                print(
                    f"[SimplePlanner] Calling LLM: Gemini | model={self._active_model}"
                )
                plan = gemini_plan_from_scene(
                    goal,
                    scene_state,
                    context=context,
                    snapshot_path=snapshot_path,
                )
                if plan is None:
                    return
                with self._lock:
                    if plan.get("action") == "done":
                        self._done = True
                    self._plan = plan
                return

            user_text = build_planner_user_text(goal, scene_state, context)
            chat_messages = self._build_messages(user_text, snapshot_path)

            if self._client is None:
                print("[SimplePlanner] ERROR: Ollama client not available.")
                return

            print(
                f"[SimplePlanner] Calling LLM: Ollama (local) | model={self._active_model}"
            )
            chat_response = self._client.chat(model=self._model, messages=chat_messages)
            response_text = chat_response["message"]["content"].strip()

            print(f"[SimplePlanner] Raw response ({len(response_text)} chars):\n{response_text[:400]}")

            if not response_text:
                print("[SimplePlanner] ERROR: LLM returned an empty response.")
                return

            markdown_fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
            if markdown_fence_match:
                extracted_json_string = markdown_fence_match.group(1).strip()
            else:
                brace_match = re.search(r"\{[\s\S]*\}", response_text)
                if brace_match:
                    extracted_json_string = brace_match.group(0)

            if not extracted_json_string:
                print("[SimplePlanner] ERROR: No JSON object found in response.")
                return

            plan = json.loads(extracted_json_string)
            print(f"[SimplePlanner] Parsed action: '{plan.get('action')}'")

            with self._lock:
                if plan.get("action") == "done":
                    self._done = True
                self._plan = plan

        except json.JSONDecodeError as parse_error:
            print(f"[SimplePlanner] JSON parse error: {parse_error}")
            print(f"[SimplePlanner] Attempted to parse: {extracted_json_string!r}")
        except Exception as llm_error:
            print(f"[SimplePlanner] LLM error: {llm_error}")
        finally:
            with self._lock:
                self._planning = False

    def _build_messages(self, user_text: str, snapshot_path: Optional[str]) -> list:
        user_message: dict = {"role": "user", "content": user_text}

        if snapshot_path and os.path.isfile(snapshot_path) and _PIL_AVAILABLE:
            try:
                with Image.open(snapshot_path) as pil_image:
                    jpeg_buffer = BytesIO()
                    pil_image.save(jpeg_buffer, format="JPEG")
                    image_base64 = base64.b64encode(jpeg_buffer.getvalue()).decode("utf-8")
                user_message["images"] = [image_base64]
                print(f"[SimplePlanner] Attached image: {os.path.basename(snapshot_path)}")
            except Exception as encode_error:
                print(f"[SimplePlanner] Image encode failed: {encode_error}")

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            user_message,
        ]
