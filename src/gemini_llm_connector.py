"""
Call Google Gemini with the same planner text shape as SimplePlanner.

Uses the official google-genai Python package. Not the older google-generativeai package.

Environment variables:
GEMINI_API_KEY or GOOGLE_API_KEY is required.
GEMINI_MODEL is optional. If unset the code uses gemini 2.0 flash.

Install with pip install google-genai.

Typical use is to set the planner backend to gemini in env and run the simple controller or call gemini_plan_from_scene from your code.
"""

from __future__ import annotations

import json
import os
import re
import time
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
            "Missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY in the environment."
        )

    return key


def _resolve_model(explicit: Optional[str]) -> str:
    return (explicit or os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL).strip()


def _is_http_503(exc: BaseException) -> bool:
    """True when the error looks like HTTP 503 service unavailable."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    if code == 503:
        return True

    response = getattr(exc, "response", None)
    if response is not None:
        sc = getattr(response, "status_code", None)
        if sc == 503:
            return True

    msg = str(exc).lower()
    if "503" in msg or "service unavailable" in msg:
        return True

    return False


def extract_json_plan(raw: str) -> Optional[dict[str, Any]]:
    """
    Pull one JSON object from model text.

    Accepts markdown code fences or the first brace block. Same idea as SimplePlanner post processing.
    """
    raw = (raw or "").strip()

    if not raw:
        return None

    json_str: Optional[str] = None

    # Try fenced json first since many models wrap output that way.
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
    # Prefer the convenience text field if the SDK filled it.
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    # Otherwise stitch candidate parts manually.
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
    Send one user turn with optional snapshot image plus text. Returns raw model text.
    """
    if not _GEMINI_SDK_AVAILABLE:
        raise ImportError("Gemini SDK not installed. Run pip install google-genai")

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

    # Retry a few times on 503 because the service can be briefly overloaded.
    max_attempts = max(1, int(os.getenv("GEMINI_503_MAX_ATTEMPTS", "5")))
    retry_delay_s = max(0.0, float(os.getenv("GEMINI_503_RETRY_DELAY_S", "2.0")))

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=mdl,
                contents=contents,
                config=_genai_types.GenerateContentConfig(system_instruction=sys_instr),
            )
            return _response_text(response)
        except Exception as err:
            if not _is_http_503(err) or attempt >= max_attempts - 1:
                raise

            print(
                f"[gemini_llm_connector] Gemini unavailable 503 "
                f"attempt {attempt + 1} of {max_attempts} retry in {retry_delay_s:.1f} s"
            )
            time.sleep(retry_delay_s)


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
    One planning step like SimplePlanner call LLM but through Gemini.

    Returns the parsed plan dict or None on failure.
    """
    try:
        user_text = build_planner_user_text(goal, scene_state, context)
        mdl = _resolve_model(model)

        print(f"[gemini_llm_connector] Calling Gemini model={mdl!r}")

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
            print("[gemini_llm_connector] ERROR no valid JSON plan in response.")
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
    """True if the SDK imports and an API key is set."""
    if not _GEMINI_SDK_AVAILABLE:
        return False

    try:
        _resolve_api_key(None)
        return True
    except ValueError:
        return False
