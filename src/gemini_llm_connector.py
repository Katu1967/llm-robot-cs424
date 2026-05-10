"""
gemini_llm_connector.py — Call Google Gemini with the same planner prompt/shape as SimplePlanner.

Uses the official Google Gen AI SDK (``google-genai``), not the legacy
``google-generativeai`` package.

Environment
-----------
GEMINI_API_KEY (required)
    Create a key in Google AI Studio: https://aistudio.google.com/apikey

GEMINI_MODEL (optional)
    Model id, e.g. ``gemini-2.0-flash``, ``gemini-2.5-flash``. Defaults to
    ``gemini-2.0-flash`` if unset.

Install
-------
    pip install google-genai

Typical use
-----------
Set ``SIMPLE_PLANNER_BACKEND=gemini`` and run the simple controller as usual, or call
``gemini_plan_from_scene(...)`` from your own code.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

try:
    from google import genai as _genai
    from google.genai import types as _genai_types

    _GEMINI_SDK_AVAILABLE = True
except ImportError:
    _genai = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]
    _GEMINI_SDK_AVAILABLE = False

from simple_planner import _SYSTEM_PROMPT, build_planner_user_text

_DEFAULT_MODEL = "gemini-2.0-flash"


def _resolve_api_key(explicit: Optional[str]) -> str:
    key = (explicit or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Missing API key: set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
        )
    return key


def _resolve_model(explicit: Optional[str]) -> str:
    return (explicit or os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL).strip()


def extract_json_plan(raw: str) -> Optional[dict[str, Any]]:
    """
    Parse a planner JSON object from model output (fences or first {...} block).
    Same behavior as SimplePlanner._call_llm post-processing.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    json_str: Optional[str] = None
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        json_str = fence_match.group(1).strip()
    else:
        brace_match = re.search(r"\{[\s\S]*\}", raw)
        if brace_match:
            json_str = brace_match.group(0)

    if not json_str:
        return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _response_text(response: Any) -> str:
    """Best-effort text extraction for google-genai responses."""
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    try:
        parts = response.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts).strip()
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _load_pil_image(path: str):
    from PIL import Image

    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def gemini_generate_content(
    user_text: str,
    *,
    system_instruction: Optional[str] = None,
    snapshot_path: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Send one user turn (optional camera image + text) to Gemini; return raw text.

    Parameters mirror the planner: system instruction + user blob + optional JPEG/PNG path.
    """
    if not _GEMINI_SDK_AVAILABLE:
        raise ImportError("Gemini SDK not installed. Run: pip install google-genai")

    key = _resolve_api_key(api_key)
    mdl = _resolve_model(model)
    sys_instr = system_instruction if system_instruction is not None else _SYSTEM_PROMPT

    client = _genai.Client(api_key=key)

    contents: list = []
    if snapshot_path and os.path.isfile(snapshot_path):
        try:
            contents.append(_load_pil_image(snapshot_path))
            print(f"[gemini_llm_connector] Attached image: {os.path.basename(snapshot_path)}")
        except Exception as e:
            print(f"[gemini_llm_connector] Image load failed: {e}")

    contents.append(user_text)

    response = client.models.generate_content(
        model=mdl,
        contents=contents,
        config=_genai_types.GenerateContentConfig(system_instruction=sys_instr),
    )
    return _response_text(response)


def gemini_plan_from_scene(
    goal: str,
    scene_state: dict,
    *,
    context: str = "",
    snapshot_path: Optional[str] = None,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    One planning step: same inputs as ``SimplePlanner._call_llm`` → parsed plan dict
    (``action``, ``aliases``, …) or ``None`` on failure.
    """
    try:
        user_text = build_planner_user_text(goal, scene_state, context)
        mdl = _resolve_model(model)
        print(f"[gemini_llm_connector] Calling Gemini model={mdl!r}…")
        raw = gemini_generate_content(
            user_text,
            system_instruction=system_instruction,
            snapshot_path=snapshot_path,
            model=model,
            api_key=api_key,
        )
        print(f"[gemini_llm_connector] Raw response ({len(raw)} chars):\n{raw[:400]}")

        plan = extract_json_plan(raw)
        if plan is None:
            print("[gemini_llm_connector] ERROR: No valid JSON plan in response.")
        else:
            print(f"[gemini_llm_connector] Parsed action: {plan.get('action')!r}")
        return plan
    except ValueError as e:
        print(f"[gemini_llm_connector] Config error: {e}")
        return None
    except ImportError as e:
        print(f"[gemini_llm_connector] {e}")
        return None
    except Exception as e:
        print(f"[gemini_llm_connector] Gemini error: {e}")
        return None


def gemini_available() -> bool:
    """True if ``google-genai`` is importable and ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) is set."""
    if not _GEMINI_SDK_AVAILABLE:
        return False
    try:
        _resolve_api_key(None)
        return True
    except ValueError:
        return False
