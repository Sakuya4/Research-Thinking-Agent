from __future__ import annotations

import json
import re
from typing import Any, Dict


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJ = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor for LLM outputs.

    Supports:
    - pure JSON
    - ```json ...``` fenced blocks
    - text that contains a JSON object somewhere

    Raises:
    - ValueError with helpful preview when not parseable.
    """
    if not text or not text.strip():
        raise ValueError("LLM returned empty response (expected JSON).")

    t = text.strip()

    # 1) fenced JSON block
    m = _JSON_FENCE.search(t)
    if m:
        t = m.group(1).strip()

    # 2) direct parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # 3) first {...} block
    m2 = _FIRST_OBJ.search(t)
    if m2:
        return json.loads(m2.group(1).strip())

    preview = t[:400].replace("\n", "\\n")
    raise ValueError(f"LLM response is not JSON. preview={preview}")
