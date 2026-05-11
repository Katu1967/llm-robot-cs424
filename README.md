# llm-robot-cs424 (Webots NAO, simple stack)

Python controller: camera + YOLOv8 + scene JSON + Ollama or Gemini + walk/turn primitives.

## Setup

- Webots, Python 3.10+, `pip install -r requirements.txt`
- Create `.env` with at least `WEBOTS_HOME` (path to Webots.app or install root)
- Ollama: `ollama serve` then `make pull-model` (default model `llama3.2-vision`)
- Optional Gemini: `SIMPLE_PLANNER_BACKEND=gemini` and `GEMINI_API_KEY` in `.env`

## Run

```bash
export WEBOTS_HOME=/Applications/Webots.app   # example
make simple
```

First YOLO run may download `yolov8n.pt` into `src/models/`.

## Layout

- `docs/SIMPLE_STACK.md` — data flow and where to edit behavior
- `src/simple_controller.py` — main loop (nav-goal phrases + alias checks live here)
- `src/simple_planner.py` — LLM + system prompt
- `src/simple_executor.py` — spin search, approach, steps
- `src/nao_interface.py` — Webots motions + head
- `src/scene_state.py` — YOLO + RangeFinder + sonar JSON
