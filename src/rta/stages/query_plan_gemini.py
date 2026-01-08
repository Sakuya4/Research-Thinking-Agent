from __future__ import annotations

import json
from typing import Tuple

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
    # If you already have this in your file, keep your existing one.
    # This is a minimal fallback.
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
    return t


def _repair_json_via_llm(client: GeminiClient, logger: EventLogger, broken_json: str) -> str:
    system = """You fix broken JSON.
Return ONLY valid JSON. No markdown fences. No explanations.
All strings must be in English."""
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
    logger.log("stage1", "repair_meta", {"model": meta.get("model"), "latency_ms": meta.get("latency_ms")})
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

    # Make stage1 less likely to truncate
    raw_text, meta = client.generate_json(
        logger=logger,
        stage="stage1",
        system=_SYSTEM,
        user=user,
        schema_hint=_SCHEMA_HINT,
        temperature=0.2,
        max_output_tokens=2000,
    )

    logger.log("stage1", "raw_preview", {"preview": raw_text[:220], "model": meta.get("model"), "latency_ms": meta.get("latency_ms")})

    # ---- Attempt 1: parse directly ----
    candidate = _extract_json_block(raw_text)
    try:
        obj = json.loads(candidate)
    except Exception as e:
        logger.log("stage1", "json_parse_failed", {"attempt": 1, "reason": str(e)})

        # ---- Attempt 2-3: repair up to 2 times ----
        last_err = None
        repaired_candidate = candidate

        for attempt in (2, 3):
            repaired = _repair_json_via_llm(client, logger, repaired_candidate)
            repaired_candidate = _extract_json_block(repaired)

            logger.log("stage1", "json_repair_preview", {"attempt": attempt, "preview": repaired_candidate[:180]})

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
