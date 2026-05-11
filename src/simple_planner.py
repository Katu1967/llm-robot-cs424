"""
simple_planner.py — LLM planner for the simple controller stack.

The planner chooses high-level robot actions. Head pitch is not exposed as an LLM
tool: during visual approach, the controller adjusts pitch from the target's
vertical image position while keeping yaw neutral.

Search behavior:
- locate_object performs a 360° spin with the head neutral.
- SEARCH_MODE before FOUND keeps the head neutral so YOLO scans the horizon.
- move_to_object uses vision plus RangeFinder depth to approach the target.
- APPROACH_CHECKPOINT pauses the feet briefly so the planner can continue or dodge.
"""

import base64
import json
import os
import re
import threading
from io import BytesIO
from typing import List, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import ollama as ollama_api
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


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
- "crouch": Use ONLY when you have received "SUPER_CLOSE:" in the context and need to lower your center of gravity or inspect something low.
- "pick_object": Use ONLY when you have received "SUPER_CLOSE:" in the context and the user asked you to grab, lift, or pick something up.
- "move_to_object": Field "aliases" (required). Vision + **RangeFinder depth** guided walk. Use ONLY after
  "OBJECT_IN_VIEW:" and/or "FOUND:" in CURRENT CONTEXT. The robot aims to get within about **0.21 m** of the target when depth distance_m is reliable and the target is roughly centered (**SUPER_CLOSE:**). **Head pitch is
  not an LLM action** — during approach the controller tilts the head **only** from the target's **vertical position in
  the image** (above/below center); head yaw stays **0**. While walking, you will receive **APPROACH_CHECKPOINT:** about
  every **10 seconds** (feet paused, head still tracking vertically, sonar + depth in context). On a checkpoint: return
  **move_to_object** with the **same aliases** to continue, or **turn_degrees** / **move_forward** (keep **≤0.5 m**
  for dodge) to avoid obstacles — **one** JSON action per response. After a dodge step finishes (**STEP_DONE** /
  **APPROACH_INTERRUPT**), return **move_to_object** with the **same aliases** again to resume the approach. The
  controller auto-completes the goal on SUPER_CLOSE (no separate "done" needed for that path).
- "done": Task succeeded. Use this ONLY when the ENTIRE user goal is complete (e.g., after navigating AND performing the requested physical action like crouching/picking). Do NOT use this just because you arrived at the object.
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
        "the", "a", "an", "to", "go", "get", "and", "for", "nao", "robot",
        "walk", "move", "head", "toward", "towards", "near", "find", "locate",
        "approach", "please", "me", "from", "with", "that", "this", "your",
        "its", "our", "use", "then", "when", "how", "can", "you", "let",
        "make", "just",
    }
)


_LOW_OBJECT_KEYWORDS = (
    "laptop", "computer", "keyboard", "floor", "ground", "low", "desk",
    "table", "bag", "backpack", "phone", "book",
)


def _label_matches_user_goal(goal: str, label: str) -> bool:
    """Return True when a detected label plausibly refers to the user's target."""
    normalized_label = (label or "").strip().lower()
    normalized_goal = (goal or "").lower()

    if not normalized_label or not normalized_goal:
        return False

    if normalized_label in normalized_goal:
        return True

    goal_words = normalized_goal.replace(",", " ").replace(".", " ").split()
    for word in goal_words:
        cleaned_word = word.strip("?!'\"")
        if len(cleaned_word) < 3 or cleaned_word in _GOAL_STOPWORDS:
            continue
        if (
            cleaned_word == normalized_label
            or cleaned_word in normalized_label
            or normalized_label in cleaned_word
        ):
            return True

    return False


def _safe_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _target_distance_summary(goal: str, detected_objects: List[dict]) -> str:
    """Summarize RangeFinder depth for detections related to the current goal."""
    summary_lines: List[str] = []

    for detected_object in detected_objects:
        label = detected_object.get("label")
        if not label or not _label_matches_user_goal(goal, str(label)):
            continue

        depth_meters = _safe_float(detected_object.get("distance_m"))
        position = detected_object.get("position", "?")

        if depth_meters is None:
            summary_lines.append(
                f"  - {label}: distance_m=n/a (no RangeFinder sample for this box), position={position}"
            )
        else:
            summary_lines.append(
                f"  - {label}: distance_m={depth_meters:.3f} m "
                f"(RangeFinder @ YOLO bbox), position={position}"
            )

    if not summary_lines:
        return ""

    return (
        "TARGET DEPTH (goal-relevant rows — use **distance_m** in meters for range, not the `distance` bucket):\n"
        + "\n".join(summary_lines)
        + "\n"
    )


def _format_visible_objects(detected_objects: List[dict]) -> str:
    object_lines: List[str] = []

    for detected_object in detected_objects:
        height_fraction = detected_object.get("height_frac")
        height_text = f", height_frac={height_fraction}" if height_fraction is not None else ""

        center_x = detected_object.get("cx_norm")
        center_y = detected_object.get("cy_norm")
        center_text = ""
        if center_x is not None and center_y is not None:
            center_text = f", cx_norm={center_x}, cy_norm={center_y}"

        depth_meters = _safe_float(detected_object.get("distance_m"))
        depth_text = f", distance_m={depth_meters:.3f}" if depth_meters is not None else ", distance_m=n/a"

        object_lines.append(
            f"- {detected_object['label']}: position={detected_object['position']}, "
            f"distance={detected_object['distance']}{depth_text}{height_text}{center_text}"
        )

    return "\n".join(object_lines) if object_lines else "(None)"


def _low_object_note(goal: str, detected_objects: List[dict], context: str) -> str:
    note = ""
    normalized_goal = goal.lower()

    if any(keyword in normalized_goal for keyword in _LOW_OBJECT_KEYWORDS):
        note = (
            "\nLOW-OBJECT NOTE: The goal may involve something low. The head stays **level** during search; "
            "after the target is visible, **move_to_object** handles approach while head pitch tracks the bbox vertically — "
            "you cannot command look_down.\n"
        )

    normalized_context = (context or "").upper()
    search_failed = "TIMEOUT" in normalized_context or "LOST" in normalized_context

    if not detected_objects and search_failed:
        note += (
            "\nNO TARGET IN VISIBLE OBJECTS after search trouble — prefer **locate_object**, **move_forward**, or "
            "**turn_degrees** (head stays neutral).\n"
        )

    return note


def build_planner_user_text(goal: str, scene_state: dict, context: str) -> str:
    detected_objects = scene_state.get("objects", [])
    visible_objects_text = _format_visible_objects(detected_objects)
    target_depth_text = _target_distance_summary(goal, detected_objects)
    low_object_text = _low_object_note(goal, detected_objects, context)

    sonar_state = scene_state.get("sonar", {})
    left_sonar_m = sonar_state.get("left_m")
    right_sonar_m = sonar_state.get("right_m")

    range_finder_legend = (
        "RANGE_FINDER: **distance_m** is true depth in **meters** from the Webots RangeFinder sampled at the "
        "YOLO bbox (same geometry as the RGB frame for alignment). It is **not** inferred from RGB appearance. "
        "The **distance** field on each line is a coarse **2D box size bucket**, not meters.\n\n"
    )

    return f"""USER GOAL: {goal}
CURRENT CONTEXT: {context or "Initial planning — no executor status yet"}

{range_finder_legend}{target_depth_text}VISIBLE OBJECTS:
{visible_objects_text}

SONAR: Left={left_sonar_m}m, Right={right_sonar_m}m
{low_object_text}
If the goal is to go to / find / approach an object, your FIRST response MUST be locate_object with a non-empty "aliases" list.
That starts a **360° spin** (head **neutral / level**) plus detection; watch for FOUND:, TIMEOUT:, and OBJECT_IN_VIEW: in CURRENT CONTEXT.
After the spin, explore with **move_forward** and **turn_degrees** (one tool per step). **look_up** and **look_down**
are **not supported** — head stays neutral until approach; then pitch follows the target vertically only.
Then move_to_object after the target is confirmed visible. While approaching, respond to **APPROACH_CHECKPOINT:**
(continue with same **move_to_object** aliases, or dodge with move/turn only). Goal: reach within about **0.21 m** RangeFinder depth when centered.
Always pass the required JSON fields for each tool (see system prompt). Prefer **0.55–0.9 m** for move_forward when exploring.
Use "done" when the goal is met **without** relying on SUPER_CLOSE (e.g. inspection-only goals). For go-to-object, always return move_to_object to continue the approach and let the automatic SUPER_CLOSE finish the job when distance_m <= 0.21 m.

What is your next action? Respond in JSON only."""


class SimplePlanner:
    """Non-blocking wrapper around the selected LLM planner backend."""

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
                "[SimplePlanner]   (set SIMPLE_PLANNER_BACKEND=gemini; key: GEMINI_API_KEY or GOOGLE_API_KEY)"
            )
        elif OLLAMA_AVAILABLE:
            self._init_ollama_client(ollama_host)
        else:
            print(
                "[SimplePlanner] WARNING: Ollama not installed or backend misconfigured.\n"
                f"[SimplePlanner]   SIMPLE_PLANNER_BACKEND={self._backend!r} | expected pip package: ollama"
            )

    def _init_ollama_client(self, ollama_host: str) -> None:
        try:
            client = ollama_api.Client(host=ollama_host, timeout=300.0)
            client.list()
            self._client = client
            print(
                f"[SimplePlanner] LLM backend: Ollama (local) | model={self._active_model} | host={ollama_host}\n"
                "[SimplePlanner]   (use SIMPLE_PLANNER_BACKEND=gemini to use Google Gemini instead)"
            )
        except Exception as error:
            print(f"[SimplePlanner] WARNING: Ollama unavailable: {error}")

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
            current_plan = self._plan
            self._plan = None
            return current_plan

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def request_plan(self, scene_state: dict, snapshot_path: Optional[str] = None, context: str = ""):
        with self._lock:
            if self._planning or self._goal is None:
                return
            self._planning = True
            goal = self._goal

        planning_thread = threading.Thread(
            target=self._call_llm,
            args=(goal, scene_state, snapshot_path, context),
            daemon=True,
        )
        planning_thread.start()

    def _call_llm(
        self,
        goal: str,
        scene_state: dict,
        snapshot_path: Optional[str],
        context: str,
    ) -> None:
        extracted_json = None

        try:
            if self._backend == "gemini":
                self._call_gemini(goal, scene_state, snapshot_path, context)
                return

            user_text = build_planner_user_text(goal, scene_state, context)
            messages = self._build_messages(user_text, snapshot_path)

            if self._client is None:
                print("[SimplePlanner] ERROR: Ollama client not available.")
                return

            print(f"[SimplePlanner] Calling LLM: Ollama (local) | model={self._active_model}")
            response = self._client.chat(model=self._model, messages=messages)
            raw_response = response["message"]["content"].strip()

            print(f"[SimplePlanner] Raw response ({len(raw_response)} chars):\n{raw_response[:400]}")

            if not raw_response:
                print("[SimplePlanner] ERROR: LLM returned an empty response.")
                return

            extracted_json = self._extract_json_object(raw_response)
            if not extracted_json:
                print("[SimplePlanner] ERROR: No JSON object found in response.")
                return

            plan = json.loads(extracted_json)
            print(f"[SimplePlanner] Parsed action: '{plan.get('action')}'")
            self._store_plan(plan)

        except json.JSONDecodeError as error:
            print(f"[SimplePlanner] JSON parse error: {error}")
            print(f"[SimplePlanner] Attempted to parse: {extracted_json!r}")
        except Exception as error:
            print(f"[SimplePlanner] LLM error: {error}")
        finally:
            with self._lock:
                self._planning = False

    def _call_gemini(
        self,
        goal: str,
        scene_state: dict,
        snapshot_path: Optional[str],
        context: str,
    ) -> None:
        from gemini_llm_connector import gemini_plan_from_scene

        print(f"[SimplePlanner] Calling LLM: Gemini | model={self._active_model}")
        plan = gemini_plan_from_scene(
            goal,
            scene_state,
            context=context,
            snapshot_path=snapshot_path,
        )

        if plan is not None:
            self._store_plan(plan)

    def _store_plan(self, plan: dict) -> None:
        with self._lock:
            if plan.get("action") == "done":
                self._done = True
            self._plan = plan

    @staticmethod
    def _extract_json_object(raw_response: str) -> Optional[str]:
        fenced_json = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_response)
        if fenced_json:
            return fenced_json.group(1).strip()

        inline_json = re.search(r"\{[\s\S]*\}", raw_response)
        if inline_json:
            return inline_json.group(0)

        return None

    def _build_messages(self, user_text: str, snapshot_path: Optional[str]) -> list:
        user_message: dict = {"role": "user", "content": user_text}

        if snapshot_path and os.path.isfile(snapshot_path) and PIL_AVAILABLE:
            try:
                with Image.open(snapshot_path) as image:
                    image_buffer = BytesIO()
                    image.save(image_buffer, format="JPEG")
                    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

                user_message["images"] = [encoded_image]
                print(f"[SimplePlanner] Attached image: {os.path.basename(snapshot_path)}")
            except Exception as error:
                print(f"[SimplePlanner] Image encode failed: {error}")

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            user_message,
        ]
