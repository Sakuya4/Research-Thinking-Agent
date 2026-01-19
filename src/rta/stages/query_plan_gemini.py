"""
Stage 1: Query Planning Module (Production Grade).
Generates a structured research plan and search queries from a user topic.
Includes robust JSON parsing, sanitization, and LLM-based self-repair.

File: src/rta/stages/query_plan_gemini.py
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List

# --- Imports from Project Structure ---
try:
    from rta.schemas.query_plan import QueryPlan
    # We use the unified client factory we created earlier
    from rta.utils.llm_client import get_default_client
except ImportError:
    # Fallback definitions to prevent ImportErrors during CLI startup
    from pydantic import BaseModel
    class QueryPlan(BaseModel):
        original_topic: str
        expanded_queries: List[str]
        must_include: List[str] = []
        exclude: List[str] = []
        target_subtasks: List[str] = []
        notes: str = ""
    
    def get_default_client():
        raise ImportError("rta.utils.llm_client not found")

logger = logging.getLogger(__name__)

# --- Constants ---

_SYSTEM_PROMPT = """You are a research-scoping agent.
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

# --- Robust JSON Helper Functions ---

def _extract_json_block(text: str) -> str:
    """
    Try to extract a JSON object from model output.
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
    Make JSON parsing more robust.
    """
    s = s.lstrip("\ufeff").strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Simple escape logic
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


def _repair_json_via_llm(client: Any, broken_json: str) -> str:
    """
    Self-Correction Mechanism.
    """
    logger.info("[QueryPlan] Attempting to repair broken JSON via LLM...")
    prompt = (
        f"You fix broken JSON. Return ONLY valid JSON.\n\n"
        f"Broken JSON:\n{broken_json}\n\n"
        f"Schema Hint:\n{_SCHEMA_HINT}"
    )
    try:
        raw_repaired = client.generate_text(prompt)
        return raw_repaired
    except Exception as e:
        logger.error(f"[QueryPlan] JSON repair failed: {e}")
        return broken_json


# --- Main Execution Logic ---

def run_query_planning(topic: str) -> QueryPlan:
    """
    Executes the query planning stage using the LLM.
    """
    client = get_default_client()
    
    # Build User Prompt
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Topic: {topic}\n\n"
        f"Requirements:\n"
        f"- expanded_queries: exactly 12 (short strings)\n"
        f"- must_include: 3-6 items\n"
        f"- exclude: 0-6 items\n"
        f"- target_subtasks: 5-8 items\n"
        f"- notes: 1-2 short sentences\n"
        f"Return ONLY JSON.\n\n"
        f"Expected Format:\n{_SCHEMA_HINT}"
    )

    logger.info(f"[QueryPlan] Generating plan for topic: {topic}")

    # 1. Generate Raw Text
    try:
        raw_text = client.generate_text(full_prompt)
    except Exception as e:
        logger.error(f"[QueryPlan] LLM generation failed: {e}")
        # If generation fails completely, raise to trigger fallback
        raw_text = ""

    # 2. Parse and Validate Loop
    candidate = _sanitize_json_text(_extract_json_block(raw_text))
    obj = None
    last_err = None

    if candidate:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.warning(f"[QueryPlan] JSON Parse Attempt 1 failed: {e}")
            # Attempt Repair
            repaired_text = _repair_json_via_llm(client, candidate)
            repaired_candidate = _sanitize_json_text(_extract_json_block(repaired_text))
            try:
                obj = json.loads(repaired_candidate)
                logger.info("[QueryPlan] JSON repaired successfully.")
            except json.JSONDecodeError as e2:
                last_err = e2
                logger.warning(f"[QueryPlan] Repair failed: {e2}")

    # 3. Validation or Fallback
    if obj:
        try:
            if "original_topic" not in obj:
                obj["original_topic"] = topic
            plan = QueryPlan.model_validate(obj)
            logger.info("[QueryPlan] Schema validation successful.")
            return plan
        except Exception as e:
            logger.error(f"[QueryPlan] Pydantic Validation Error: {e}")
    
    # --- FIXED FALLBACK BLOCK ---
    # This block now includes ALL required fields to prevent "Field required" errors
    logger.warning("[QueryPlan] Triggering Fallback Plan.")
    return QueryPlan(
        original_topic=topic,
        expanded_queries=[topic, f"{topic} survey", f"{topic} methodology", f"{topic} challenges", f"{topic} state of the art"],
        must_include=["Key concepts", "Recent advances", "Benchmarks"],  # Fixed: Added missing field
        exclude=["Irrelevant domains", "Outdated methods"],              # Fixed: Added missing field
        target_subtasks=["Define terminology", "Categorize methods", "Compare performance"], # Fixed: Added missing field
        notes="Generated via fallback due to JSON parsing failure."
    )