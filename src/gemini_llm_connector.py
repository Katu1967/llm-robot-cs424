"""
gemini_llm_connector.py — Call Google Gemini with the same planner prompt/shape as SimplePlanner.

Uses the official Google Gen AI SDK (`google-genai`).

This version is Webots-safe on Apple Silicon because Webots may run as x86_64.
It explicitly loads packages from .venv_x86 / Python 3.11 site-packages.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Force Webots to see the x86 virtual environment packages
# ---------------------------------------------------------------------------

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CANDIDATE_SITE_PACKAGES = [
    os.path.join(_ROOT_DIR, ".venv_x86", "lib", "python3.11", "site-packages"),
    os.path.join(_ROOT_DIR, ".venv_x86", "lib", "python3.10", "site-packages"),
    os.path.join(_ROOT_DIR, ".venv_x86", "lib", "python3.9", "site-packages"),
]

for path in _CANDIDATE_SITE_PACKAGES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"[gemini_llm_connector] Added site-packages: {path}")
        break


try:
    from google import genai as _genai
    from google.genai import types as _genai_types

    _GEMINI_SDK_AVAILABLE = True
    print("[gemini_llm_connector] Gemini SDK imported successfully")

except Exception as e:
    print(f"[gemini_llm_connector] IMPORT ERROR: {e}")
    _genai = None
    _genai_types = None
    _GEMINI_SDK_AVAILABLE = False


from simple_planner import _SYSTEM_PROMPT, build_planner_user_text


_DEFAULT_MODEL = "gemini-2.0-flash"


def _load_dotenv_if_available() -> None:
    """
    Load .env manually if python-dotenv is installed.
    This helps when Webots does not inherit shell env variables.
    """
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(_ROOT_DIR, ".env")
        if os.path.isfile(env_path):
            load_dotenv(env_path, override=False)
            print(f"[gemini_llm_connector] Loaded .env from {env_path}")
    except Exception:
        pass


_load_dotenv_if_available()


def _resolve_api_key(explicit: Optional[str]) -> str:
    key = (
        explicit
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()

    if not key:
        raise ValueError(
            "Missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY in your shell or .env file."
        )

    return key


def _resolve_model(explicit: Optional[str]) -> str:
    return (
        explicit
        or os.getenv("GEMINI_MODEL")
        or _DEFAULT_MODEL
    ).strip()


def extract_json_plan(raw: str) -> Optional[dict[str, Any]]:
    """
    Parse a planner JSON object from model output.
    Handles markdown fences or the first {...} block.
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
    Send one user turn to Gemini.
    Optionally includes a camera snapshot image.
    """

    if not _GEMINI_SDK_AVAILABLE:
        raise ImportError(
            "Gemini SDK not available in Webots Python. "
            "Check .venv_x86/lib/python3.11/site-packages and architecture."
        )

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
        config=_genai_types.GenerateContentConfig(
            system_instruction=sys_instr,
            temperature=0.2,
        ),
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
    One planning step:
    same inputs as SimplePlanner._call_llm -> parsed action dict or None.
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
    """
    True if google-genai is importable and API key is configured.
    """
    if not _GEMINI_SDK_AVAILABLE:
        return False

    try:
        _resolve_api_key(None)
        return True
    except ValueError:
        return False