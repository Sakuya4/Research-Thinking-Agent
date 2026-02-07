"""
Stage 1: Query Planning Module (Production Grade).
Generates a structured research plan and search queries from a user topic.
File: src/rta/stages/query_plan_gemini.py
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Internal Schema Definition to avoid circular imports
class QueryPlan(BaseModel):
    original_topic: str
    expanded_queries: List[str]
    must_include: List[str] = []
    exclude: List[str] = []
    target_subtasks: List[str] = []
    notes: str = ""

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a research-scoping agent.
Given a short topic query, you must produce a compact JSON query plan for literature search.
Hard requirements:
- Output MUST be valid JSON only (no markdown fences, no extra text).
- ALL strings MUST be in English.
- The user topic may be non-English; translate internally but output English.
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
    Extracts the JSON object from raw LLM text output.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1:
        return t[start : end + 1]
    return t

def _repair_json_via_llm(client: Any, broken_json: str) -> str:
    """
    Attempts to fix malformed JSON using a secondary LLM request.
    """
    logger.info("[QueryPlan] Attempting to repair broken JSON via LLM...")
    prompt = f"Fix this broken JSON. Return ONLY valid JSON:\n{broken_json}\nSchema Hint:\n{_SCHEMA_HINT}"
    try:
        return client.generate_text(prompt)
    except Exception:
        return broken_json

def run_query_planning(topic: str) -> QueryPlan:
    """
    Main entry point for Stage 1. Executes full planning logic as per research paper.
    """
    # Import locally to avoid potential top-level circular dependencies
    from rta.utils.llm_client import get_default_client
    client = get_default_client()
    
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\nTopic: {topic}\n\n"
        f"Requirements:\n"
        f"- expanded_queries: 12 specific search terms\n"
        f"Return ONLY JSON.\n\n"
        f"Format:\n{_SCHEMA_HINT}"
    )

    logger.info(f"[QueryPlan] Generating full research plan for: {topic}")

    try:
        raw_text = client.generate_text(full_prompt)
        candidate = _extract_json_block(raw_text)
        obj = json.loads(candidate)
    except Exception as e:
        logger.warning(f"[QueryPlan] Parsing failed: {e}. Initiating repair.")
        repaired = _repair_json_via_llm(client, raw_text)
        obj = json.loads(_extract_json_block(repaired))

    if "original_topic" not in obj:
        obj["original_topic"] = topic
        
    return QueryPlan.model_validate(obj)