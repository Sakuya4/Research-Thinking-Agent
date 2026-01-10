from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm.gemini_client import GeminiClient
from ..logger import EventLogger
from ..schemas import InputPayload, QueryPlan


_SYSTEM = """You are a research-scoping agent.
Given a short topic query, you must produce a compact JSON query plan for literature search.

Hard requirements:
- Output MUST be valid JSON only (no markdown fences, no extra text).
- ALL strings MUST be in English.
- The user topic may be non-English; translate internally but output English.
- Keep outputs short and compact.
"""

_SCHEMA_HINT = """{
  "expanded_queries": ["string", "..."],
  "must_include": ["string", "..."],
  "exclude": ["string", "..."],
  "target_subtasks": ["string", "..."],
  "notes": "string"
}"""


def _extract_json_block(text: str) -> str:
    """
    Try to extract a JSON object from model output.
    Handles:
      - pure JSON
      - ```json ... ```
      - extra accidental text around JSON
    """
    t = text.strip()

    # Strip common code fences
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)

    # If it's already a JSON object, return it
    if t.startswith("{") and t.endswith("}"):
        return t

    # Try to find the first {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]

    return t


def _sanitize_json_text(s: str) -> str:
    """
    Make JSON parsing more robust:
    - remove BOM / weird invisible chars
    - normalize newlines
    - fix unescaped newlines inside JSON string values (common cause of Unterminated string)
    """
    # remove BOM
    s = s.lstrip("\ufeff").strip()

    # normalize CRLF to LF
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Fix raw newlines that appear inside JSON string literals:
    # We scan char by char and when "in_string" and see a literal newline,
    # we replace it with '\\n'.
    out = []
    in_string = False
    escape = False
    for ch in s:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_string = True
            else:
                out.append(ch)
    return "".join(out).strip()


def _repair_json_via_llm(client: GeminiClient, logger: EventLogger, broken_json: str) -> str:
    system = (
        "You fix broken JSON.\n"
        "Return ONLY valid JSON. No markdown fences. No explanations.\n"
        "All strings must be in English."
    )
    user = {
        "task": "repair_json",
        "broken_json": broken_json,
    }
    raw, meta = client.generate_json(
        logger=logger,
        stage="stage1_repair",
        system=system,
        user=json.dumps(user, ensure_ascii=False),
        schema_hint=_SCHEMA_HINT,
        temperature=0.0,
        max_output_tokens=2000,
    )
    logger.log(
        "stage1",
        "repair_meta",
        {"model": meta.get("model"), "latency_ms": meta.get("latency_ms")},
    )
    return raw


def stage_query_plan_gemini(user_input: InputPayload, logger: EventLogger) -> QueryPlan:
    client = GeminiClient.from_env()

    user = f"""Topic: {user_input.query}
Context (optional): {user_input.context or ""}

Requirements:
- expanded_queries: exactly 12 (short strings)
- must_include: 3-6
- exclude: 0-6
- target_subtasks: 5-8
- notes: 1-2 short sentences
Return ONLY JSON.
"""

    raw_text, meta = client.generate_json(
        logger=logger,
        stage="stage1",
        system=_SYSTEM,
        user=user,
        schema_hint=_SCHEMA_HINT,
        temperature=0.2,
        max_output_tokens=2000,
    )

    logger.log(
        "stage1",
        "raw_preview",
        {"preview": raw_text[:240], "model": meta.get("model"), "latency_ms": meta.get("latency_ms")},
    )

    # ✅ Save raw output for debugging (super useful)
    # Find run_dir from logger path: runs/<run_id>/logs.jsonl
    try:
        run_dir = Path(logger.log_path).parent
        (run_dir / "stage1_raw.txt").write_text(raw_text, encoding="utf-8", errors="replace")
    except Exception as _:
        pass

    # ---- Attempt 1: parse directly ----
    candidate = _sanitize_json_text(_extract_json_block(raw_text))
    try:
        obj = json.loads(candidate)
    except Exception as e:
        logger.log("stage1", "json_parse_failed", {"attempt": 1, "reason": str(e)})

        # ---- Attempt 2-3: repair up to 2 times ----
        last_err = None
        repaired_candidate = candidate

        for attempt in (2, 3):
            repaired = _repair_json_via_llm(client, logger, repaired_candidate)
            repaired_candidate = _sanitize_json_text(_extract_json_block(repaired))

            logger.log("stage1", "json_repair_preview", {"attempt": attempt, "preview": repaired_candidate[:220]})

            try:
                obj = json.loads(repaired_candidate)
                last_err = None
                break
            except Exception as e2:
                last_err = e2
                logger.log("stage1", "json_parse_failed", {"attempt": attempt, "reason": str(e2)})

        if last_err is not None:
            logger.log("stage1", "schema_validated", {"ok": False, "reason": f"json_parse_error_after_repairs: {last_err}"})
            raise RuntimeError(f"Stage1 JSON parse failed after repairs: {last_err}")

    # ---- Validate schema with Pydantic ----
    try:
        qp = QueryPlan.model_validate(obj)
        logger.log("stage1", "schema_validated", {"ok": True})
        return qp
    except Exception as e:
        logger.log("stage1", "schema_validated", {"ok": False, "reason": f"pydantic_validation_error: {e}"})
        raise
