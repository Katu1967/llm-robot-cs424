# Simple stack (CS424)

This folder documents the **simple** Webots path: camera → vision → scene JSON → LLM → executor → NAO motions.

## Data flow

1. **`simple_controller`** — Webots `Robot` loop: grab `CameraTop` BGRA, convert to BGR, run YOLO every few frames, optionally show RangeFinder depth. Publishes **`scene_state`** on `SceneBus` when you capture (SPACE, status hooks, or live during search).
2. **`SceneStateExtractor`** — Builds the JSON: YOLO boxes, sonar, GPS/heading if present, **RangeFinder** median depth per box when the device exists.
3. **`SimplePlanner`** — Ollama or Gemini returns **one JSON object** per turn (`locate_object`, `move_forward`, …). Same system prompt for both backends.
4. **`SimpleExecutor`** — Runs locate spin, approach walk, short dodge steps. Calls **`on_status`** so the controller can re-call the LLM (e.g. `APPROACH_CHECKPOINT`).
5. **`NaoInterface`** — Webots `Motion` clips for walk/turn and `Motor.setPosition` for head; motion paths come from the **SoftBank NAO** tree under `WEBOTS_HOME` (or common install locations).

## Where to change behavior

| Want to change… | Look at… |
|------------------|-----------|
| LLM instructions / tools | `simple_planner.py` (`_SYSTEM_PROMPT`, `build_planner_user_text`) |
| “Must locate first” / plan vs target aliases | `simple_controller.py` (`_NAV_PHRASES`, `goal_requires_locate_first`, `plan_aliases_compatible_with_target`) |
| Walk speeds, stop distance, stuck timers | `simple_executor.py` (env vars at top of class) |
| YOLO rate / confidence | `simple_controller.py` constants |
| Label smoothing over frames | `detection_stabilizer.py` + `YOLO_STAB_*` env |
| Depth read quirks | `range_finder_util.py` |

## Backends

- **Ollama** — local; optional JPEG from last snapshot in chat.
- **Gemini** — `google-genai`; same user text builder; `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

Set `SIMPLE_PLANNER_BACKEND=gemini` or `ollama` (default).
