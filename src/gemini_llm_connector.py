
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
    api_key = (explicit or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

    if not api_key:
        raise ValueError(
            "Missing API key: set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
        )

    return api_key


def _resolve_model(explicit: Optional[str]) -> str:
    return (explicit or os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL).strip()


def _is_http_503(exc: BaseException) -> bool:
    """True if the error looks like HTTP 503 Service Unavailable from Gemini / transport."""
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


def extract_json_plan(raw_llm_text: str) -> Optional[dict[str, Any]]:
    trimmed = (raw_llm_text or "").strip()

    if not trimmed:
        return None

    extracted_json_string: Optional[str] = None
    markdown_fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed)

    if markdown_fence_match:
        extracted_json_string = markdown_fence_match.group(1).strip()
    else:
        brace_match = re.search(r"\{[\s\S]*\}", trimmed)
        if brace_match:
            extracted_json_string = brace_match.group(0)

    if not extracted_json_string:
        return None

    try:
        return json.loads(extracted_json_string)
    except json.JSONDecodeError:
        return None


def _response_text(response: Any) -> str:
    direct_text = getattr(response, "text", None)
    if direct_text:
        return str(direct_text).strip()

    try:
        content_parts = response.candidates[0].content.parts
        return "".join(getattr(part, "text", "") or "" for part in content_parts).strip()
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _load_pil_image(path: str):
    from PIL import Image

    pil_image = Image.open(path)
    if pil_image.mode not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGB")

    return pil_image


def gemini_generate_content(
    user_text: str,
    *,
    system_instruction: Optional[str] = None,
    snapshot_path: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    if not _GEMINI_SDK_AVAILABLE:
        raise ImportError("Gemini SDK not installed. Run: pip install google-genai")

    resolved_key = _resolve_api_key(api_key)
    model_name = _resolve_model(model)
    system_text = system_instruction if system_instruction is not None else _SYSTEM_PROMPT

    client = _genai.Client(api_key=resolved_key)

    contents: list = []

    if snapshot_path and os.path.isfile(snapshot_path):
        try:
            contents.append(_load_pil_image(snapshot_path))
            print(f"[gemini_llm_connector] Attached image: {os.path.basename(snapshot_path)}")
        except Exception as load_error:
            print(f"[gemini_llm_connector] Image load failed: {load_error}")

    contents.append(user_text)

    max_attempts = max(1, int(os.getenv("GEMINI_503_MAX_ATTEMPTS", "5")))
    retry_delay_s = max(0.0, float(os.getenv("GEMINI_503_RETRY_DELAY_S", "2.0")))

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=_genai_types.GenerateContentConfig(system_instruction=system_text),
            )
            return _response_text(response)
        except Exception as err:
            if not _is_http_503(err) or attempt >= max_attempts - 1:
                raise
            print(
                f"[gemini_llm_connector] Gemini returned 503 / unavailable "
                f"(attempt {attempt + 1}/{max_attempts}), retrying in {retry_delay_s:.1f}s…"
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
    try:
        user_text = build_planner_user_text(goal, scene_state, context)
        model_name = _resolve_model(model)
        print(f"[gemini_llm_connector] Calling Gemini model={model_name!r}...")

        raw_response_text = gemini_generate_content(
            user_text,
            system_instruction=system_instruction,
            snapshot_path=snapshot_path,
            model=model,
            api_key=api_key,
        )
        print(f"[gemini_llm_connector] Raw response ({len(raw_response_text)} chars):\n{raw_response_text[:400]}")

        plan = extract_json_plan(raw_response_text)

        if plan is None:
            print("[gemini_llm_connector] ERROR: No valid JSON plan in response.")
        else:
            print(f"[gemini_llm_connector] Parsed action: {plan.get('action')!r}")

        return plan

    except ValueError as config_error:
        print(f"[gemini_llm_connector] Config error: {config_error}")
        return None

    except ImportError as import_error:
        print(f"[gemini_llm_connector] {import_error}")
        return None

    except Exception as api_error:
        print(f"[gemini_llm_connector] Gemini error: {api_error}")
        return None


def gemini_available() -> bool:
    if not _GEMINI_SDK_AVAILABLE:
        return False

    try:
        _resolve_api_key(None)
        return True
    except ValueError:
        return False
