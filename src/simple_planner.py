"""
simple_planner.py — LLM planner for the simple controller stack.

The LLM chooses actions: locate_object (spin search + SEARCH_MODE), move_forward,
turn_degrees, look_up, look_down (emphasized for floor-level targets), move_to_object, done, fail, clarify.

After ``locate_object`` the executor spins ~360° while the controller still polls
YOLO and may inject ``OBJECT_IN_VIEW`` into context before the spin finishes.
"""

import os
import re
import json
import base64
import threading
import subprocess
import sys
from io import BytesIO
from typing import Optional

# Optional: load .env automatically if python-dotenv is installed
try:
    from dotenv import load_dotenv

    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env_path = os.path.join(_root_dir, ".env")
    load_dotenv(_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars


def _add_virtualenv_site_packages() -> None:
    """Make the controller process able to import packages installed in the venv.

    Webots may launch this module with a Python interpreter that does not inherit
    the venv's site-packages automatically, even if PYTHON_COMMAND points at the
    venv Python. When that happens, google-genai is installed but still not visible.
    """
    candidates: list[str] = []

    py_cmd = (os.getenv("PYTHON_COMMAND") or "").strip()
    if py_cmd and os.path.exists(py_cmd):
        try:
            out = subprocess.check_output(
                [py_cmd, "-c", "import site; print(site.getsitepackages()[0])"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            candidates.append(out.strip())
        except Exception:
            pass

    venv_root = (os.getenv("VIRTUAL_ENV") or "").strip()
    if venv_root:
        candidates.extend(
            [
                os.path.join(
                    venv_root,
                    "lib",
                    f"python{sys.version_info.major}.{sys.version_info.minor}",
                    "site-packages",
                ),
                os.path.join(venv_root, "lib", "site-packages"),
            ]
        )

    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_add_virtualenv_site_packages()

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

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the planning brain of a NAO humanoid robot in Webots.
You MUST pass valid tool parameters in every JSON response (see PASSING PARAMETERS below).

ALLOWED ACTIONS (exact "action" string):
- "locate_object": Navigation / search goals — REQUIRED FIRST. Fields: "aliases" (required): non-empty list of
  strings (YOLO/COCO names and synonyms, e.g. ["dog","puppy"]). The robot performs a **360° spin** while checking
  detections; you may then get "FOUND: …" after a stable sighting, or "TIMEOUT: …" if nothing matched. While
  spinning, context may also include "OBJECT_IN_VIEW:" when YOLO sees the target in the current frame.
- "move_forward": Field "meters" (required number). Use **generous** steps: prefer **0.55–0.9** m per call
  (typical 0.6–0.75) so the robot covers ground quickly; the runtime may bump small values to a minimum stride.
  One straight segment per response. After locate_object for nav goals.
- "turn_degrees": Field "degrees" (required number). POSITIVE = turn left, NEGATIVE = turn right.
- "look_up": Optional "degrees" (number, default ~12, range ~3–25). Tilts head up — use when searching high shelves / walls.
- "look_down": Optional "degrees" (number, default ~**18** at runtime if omitted, range **8–30**). Tilts head **toward the floor** — **critical** for
  objects that may sit **on the ground** (laptop, keyboard, backpack, book, phone, shoes, etc.). You **must** use
  ``look_down`` often during search (not only after failures): e.g. **every 1–2 exploration steps** alternate with
  ``move_forward`` / ``turn_degrees``, and **always** after ``TIMEOUT:`` / ``LOST:`` before only moving again.
  Prefer **18–26°** when actively scanning for floor-level targets so YOLO sees the laptop / low object; include ``reason``.
- "move_to_object": Field "aliases" (required). Vision-guided walk toward target. Use ONLY after "OBJECT_IN_VIEW:"
  and/or "FOUND:" appears in CURRENT CONTEXT (target confirmed visible). The executor **creeps forward** when the
  bbox grows, then ends the approach with **SUPER_CLOSE:** when vision thresholds are met; the **controller auto-completes the goal**
  on SUPER_CLOSE (you do not need a follow-up "done" for that path).
- "done": Task succeeded when the goal does not end with an automatic SUPER_CLOSE (e.g. non-approach goals), or
  if you judge success from the image before SUPER_CLOSE fires.
- "fail": ONLY when absolutely impossible. Prefer locate_object + different moves first.

PASSING PARAMETERS (every response):
- Always include "reason": one short string explaining the choice.
- "locate_object" and "move_to_object" MUST include "aliases": ["…"] with at least one non-empty string.
- "move_forward" MUST include "meters": <positive number>.
- "turn_degrees" MUST include "degrees": <number>.
- "look_up" MUST include "reason". "look_down" MUST include "reason" and usually explicit "degrees" (see look_down action above).
- Output exactly ONE JSON object per turn — no markdown fences, no extra keys beyond what the action needs.

SEARCH + SIGNALS:
- After locate_object, expect spin then FOUND or TIMEOUT; OBJECT_IN_VIEW may appear anytime YOLO sees the target.
- While searching (after spin or between moves), issue ONE primitive per step: move_forward, turn_degrees, look_up, or look_down — or locate_object again to re-spin / re-search.
- **Floor / laptop discipline:** If the user goal mentions a **laptop**, **keyboard**, **bag**, **floor**, **ground**, or any object that could be **below knee height**, treat ``look_down`` as a **first-class** tool: use it **often** (roughly half of non-move primitives) with **degrees** often **18–26** so the camera covers the floor in front of the robot. Do **not** chain many ``move_forward`` / ``turn_degrees`` steps without inserting ``look_down`` in between.
- STEP_DONE may include FEEDBACK: (target vs image center). PROGRESS / STUCK during move_to_object; SUPER_CLOSE ends the run automatically.

RECOVERY:
- On TIMEOUT or LOST, your **next** step should often be ``look_down`` (strong floor scan, e.g. 20–26°) or ``locate_object`` again — not only ``move_forward``.
- Use "fail" only as a last resort.

STRICT RULES:
1. First step for go-to / find / approach goals: locate_object with good aliases (for laptops include synonyms like "laptop","computer","keyboard").
2. Do not use move_to_object until OBJECT_IN_VIEW and/or FOUND has appeared in context.
3. For approach goals, **move_to_object** often ends the run automatically when the controller receives a **SUPER_CLOSE** status (final creep); use **done** for other goals or if you judge success earlier from the image.
4. When the goal may involve **floor-level** objects, use **look_down** regularly with explicit **degrees** (see action list); do not omit it for many steps in a row.
5. Follow PASSING PARAMETERS exactly.

JSON EXAMPLES:
{"action":"locate_object","aliases":["dog","puppy"],"reason":"Spin search for dog."}
{"action":"locate_object","aliases":["laptop","computer","keyboard"],"reason":"Search for laptop on floor or desk."}
{"action":"look_down","degrees":22,"reason":"Scan floor for laptop after last move."}
{"action":"move_forward","meters":0.65,"reason":"Timeout after spin; advance into room."}
{"action":"move_to_object","aliases":["dog"],"reason":"FOUND in context; approach."}
{"action":"done","reason":"Non-approach goal satisfied (or success before SUPER_CLOSE)."}
{"action":"fail","reason":"No safe path after exhaustive retries."}
"""


def build_planner_user_text(goal: str, scene_state: dict, context: str) -> str:
    """Shared user message for planner LLMs (Ollama, Gemini, etc.)."""
    objs = scene_state.get("objects", [])

    obj_list = []
    for o in objs:
        hf = o.get("height_frac")
        hf_s = f", height_frac={hf}" if hf is not None else ""
        cx = o.get("cx_norm")
        cy = o.get("cy_norm")
        xy_s = ""
        if cx is not None and cy is not None:
            xy_s = f", cx_norm={cx}, cy_norm={cy}"
        obj_list.append(
            f"- {o['label']}: position={o['position']}, distance={o['distance']}{hf_s}{xy_s}"
        )

    objs_str = "\n".join(obj_list) if obj_list else "(None)"

    gl = goal.lower()
    floor_scan_reminder = ""
    if any(
        k in gl
        for k in (
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
    ):
        floor_scan_reminder = (
            "\nFLOOR-VISION REMINDER: This goal may involve something **on the floor or a low surface**. "
            "Use **look_down** often (e.g. every 1–2 steps with other primitives) with **degrees** typically **18–26** "
            "so the camera sees low objects (e.g. laptop). Do not rely only on move_forward / turn_degrees.\n"
        )

    cu = (context or "").upper()
    if not obj_list and ("TIMEOUT" in cu or "LOST" in cu):
        floor_scan_reminder += (
            "\nNO TARGET IN VISIBLE OBJECTS after search trouble — use **look_down** (strong pitch, e.g. **20–28°**) "
            "to sweep the **floor** in front of the robot before only moving or turning again.\n"
        )

    return f"""USER GOAL: {goal}
CURRENT CONTEXT: {context or "Initial planning — no executor status yet"}

VISIBLE OBJECTS:
{objs_str}

SONAR: Left={scene_state.get('sonar',{}).get('left_m')}m, Right={scene_state.get('sonar',{}).get('right_m')}m
{floor_scan_reminder}
If the goal is to go to / find / approach an object, your FIRST response MUST be locate_object with a non-empty "aliases" list.
That starts a **360° spin** plus detection; watch for FOUND:, TIMEOUT:, and OBJECT_IN_VIEW: in CURRENT CONTEXT.
Then use move_forward / turn_degrees / look_up / **look_down** (one tool per step) as needed. **Bias toward look_down**
when the target could be on the floor (laptop, electronics, bags) — insert look_down regularly, not only after failures.
Then move_to_object after the target is confirmed visible.
Always pass the required JSON fields for each tool (see system prompt). Prefer **0.55–0.9 m** for move_forward when exploring.
Use "done" when the goal is met **without** relying on SUPER_CLOSE (e.g. inspection-only goals). For go-to-object, prefer move_to_object and let final creep + SUPER_CLOSE finish the task.

What is your next action? Respond in JSON only."""


class SimplePlanner:
    """
    Thin wrapper around Ollama.  Maintains the goal and fires off
    planning requests in a background thread so it never blocks the
    Webots step loop.
    """

    def __init__(self):
        host  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2-vision")

        self._backend = os.getenv("SIMPLE_PLANNER_BACKEND", "ollama").strip().lower()
        self._model  = model
        self._client: Optional[object] = None
        self._lock   = threading.Lock()

        self._goal:    Optional[str]  = None
        self._plan:    Optional[dict] = None   # latest parsed LLM response
        self._planning = False
        self._done     = False

        # Human-readable backend + model id for logs (SIMPLE_PLANNER_BACKEND=ollama|gemini)
        if self._backend == "gemini":
            gemini_model = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()
            self._backend_label = "Gemini"
            self._active_model = gemini_model
        else:
            self._backend_label = "Ollama"
            self._active_model = model

        if self._backend == "gemini":
            print(
                f"[SimplePlanner] LLM backend: Gemini (Google API) | model={self._active_model}\n"
                f"[SimplePlanner]   (set SIMPLE_PLANNER_BACKEND=gemini; key: GEMINI_API_KEY or GOOGLE_API_KEY)"
            )
        elif _OLLAMA_AVAILABLE:
            try:
                client = _ollama.Client(host=host, timeout=300.0)
                client.list()
                self._client = client
                print(
                    f"[SimplePlanner] LLM backend: Ollama (local) | model={self._active_model} | host={host}\n"
                    f"[SimplePlanner]   (use SIMPLE_PLANNER_BACKEND=gemini to use Google Gemini instead)"
                )
            except Exception as e:
                print(f"[SimplePlanner] WARNING: Ollama unavailable: {e}")
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
        with self._lock: return self._planning

    def has_plan(self) -> bool:
        with self._lock: return self._plan is not None

    def consume_plan(self) -> Optional[dict]:
        with self._lock:
            p = self._plan
            self._plan = None
            return p

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def request_plan(self, scene_state: dict, snapshot_path: Optional[str] = None, context: str = ""):
        with self._lock:
            if self._planning or self._goal is None:
                return
            self._planning = True

        goal  = self._goal
        t = threading.Thread(
            target=self._call_llm,
            args=(goal, scene_state, snapshot_path, context),
            daemon=True,
        )
        t.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm(self, goal: str, scene_state: dict,
                  snapshot_path: Optional[str], context: str) -> None:
        json_str = None
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
            messages  = self._build_messages(user_text, snapshot_path)

            if self._client is None:
                print("[SimplePlanner] ERROR: Ollama client not available.")
                return

            print(
                f"[SimplePlanner] Calling LLM: Ollama (local) | model={self._active_model}"
            )
            resp = self._client.chat(model=self._model, messages=messages)
            raw  = resp["message"]["content"].strip()

            # Always log what we got so problems are visible
            print(f"[SimplePlanner] Raw response ({len(raw)} chars):\n{raw[:400]}")

            if not raw:
                print("[SimplePlanner] ERROR: LLM returned an empty response.")
                return

            # --- Robust JSON extraction ---
            import re

            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if fence_match:
                json_str = fence_match.group(1).strip()
            else:
                brace_match = re.search(r"\{[\s\S]*\}", raw)
                if brace_match:
                    json_str = brace_match.group(0)

            if not json_str:
                print("[SimplePlanner] ERROR: No JSON object found in response.")
                return

            plan = json.loads(json_str)
            print(f"[SimplePlanner] Parsed action: '{plan.get('action')}'")

            with self._lock:
                if plan.get("action") == "done":
                    self._done = True
                self._plan = plan

        except json.JSONDecodeError as e:
            print(f"[SimplePlanner] JSON parse error: {e}")
            print(f"[SimplePlanner] Attempted to parse: {json_str!r}")
        except Exception as e:
            print(f"[SimplePlanner] LLM error: {e}")
        finally:
            with self._lock:
                self._planning = False

    def _build_messages(self, user_text: str, snapshot_path: Optional[str]) -> list:
        user_msg: dict = {"role": "user", "content": user_text}

        if snapshot_path and os.path.isfile(snapshot_path) and _PIL_AVAILABLE:
            try:
                with Image.open(snapshot_path) as img:
                    buf = BytesIO()
                    img.save(buf, format="JPEG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                user_msg["images"] = [img_b64]
                print(f"[SimplePlanner] Attached image: {os.path.basename(snapshot_path)}")
            except Exception as e:
                print(f"[SimplePlanner] Image encode failed: {e}")

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            user_msg,
        ]
