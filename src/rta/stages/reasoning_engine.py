"""
Reasoning Engine with Self-Refinement and Beginner-Friendly Synthesis.
File: src/rta/stages/reasoning_engine.py
"""

import logging
from typing import Any, Optional
import json

# Schemas imports (trusted from pipeline injection)
try:
    from rta.schemas.reasoning import ReasoningResult
    from rta.schemas.query_plan import QueryPlan
    from rta.schemas.topic_structuring import TopicStructuringResult
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Orchestrates research insights with an iterative refinement loop 
    and persona-driven explanations (Beginner-Friendly).
    """
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self.MAX_RETRIES = 2 

    def run(self, query_plan: Any, clustering_result: Any, papers: list) -> Any:
        topic_str = self._extract_topic_str(query_plan)
        logger.info(f"[Reasoning] Started for topic: {topic_str}")

        # Phase 1: Drafting with Persona
        logger.info("[Reasoning] Phase 1: Generating initial beginner-friendly draft...")
        draft_prompt = self._build_draft_prompt(topic_str, clustering_result, papers)
        
        try:
            from rta.schemas.reasoning import ReasoningResult
            schema_cls = ReasoningResult
        except ImportError:
            logger.error("[Reasoning] Could not import ReasoningResult schema.")
            return None

        current_result = self.llm_client.generate_structured(
            prompt=draft_prompt, 
            schema=schema_cls
        )

        # Topic Injection
        if current_result and hasattr(current_result, 'topic') and not current_result.topic:
            try:
                current_result.topic = topic_str
            except:
                pass

        # Phase 2: Refinement (The Critique Loop)
        for attempt in range(self.MAX_RETRIES):
            logger.info(f"[Reasoning] Phase 2: Refinement attempt {attempt + 1}/{self.MAX_RETRIES}")
            
            feedback = self._critique_result(current_result)
            
            if self._is_feedback_positive(feedback):
                logger.info("[Reasoning] Critique passed.")
                break
            
            logger.info(f"[Reasoning] Refining based on feedback...")
            current_result = self._refine_result(current_result, feedback, schema_cls)

        return current_result

    def _extract_topic_str(self, plan: Any) -> str:
        val = getattr(plan, 'original_topic', getattr(plan, 'topic', getattr(plan, 'query', "Unknown Topic")))
        return val

    def _build_draft_prompt(self, topic: str, clusters: Any, papers: list) -> str:
        """
        [TODO COMPLIANCE] Injects the 'Explain to a Beginner' requirement.
        """
        cluster_names = []
        if hasattr(clusters, 'clusters'):
            for c in clusters.clusters:
                name = getattr(c, 'name', getattr(c, 'topic_name', 'Unnamed Cluster'))
                cluster_names.append(name)
        
        # Combined Persona: Visionary Lead Researcher + Patient Mentor
        return (
            f"You are a Lead Researcher and Mentor. The target audience is a 'master student with no foundation'.\n"
            f"User Topic: '{topic}'\n"
            f"Retrieved {len(papers)} real papers, grouped into: {', '.join(cluster_names)}.\n\n"
            f"REQUIRED OUTPUT MINDSET:\n"
            f"1. **Step-by-Step Guidance**: Explain the technology progression from basics to advanced.\n"
            f"2. **Simplified Analogies**: For every complex method, provide a daily-life analogy (e.g., explaining ECG sampling like a camera shutter).\n"
            f"3. **Innovative Extensions**: Propose 3 directions like 'AI-guided handheld ultrasound' or 'Automated LVEF' based on the papers.\n"
            f"4. **Persona**: Be extremely patient and clear. Do not skip fundamental steps.\n\n"
            f"Strictly output valid JSON matching the ReasoningResult schema."
        )

    def _critique_result(self, result: Any) -> str:
        """Acts as the Senior Critic."""
        try:
            result_json = result.model_dump_json(indent=2) if hasattr(result, 'model_dump_json') else str(result)
        except:
            result_json = str(result)

        critic_prompt = (
            f"You are a Senior Reviewer. Check if this research output is suitable for a FOUNDATIONLESS student:\n"
            f"```json\n{result_json}\n```\n\n"
            f"Criteria:\n"
            f"1. Is the explanation simple enough for a beginner?\n"
            f"2. Are the analogies accurate and helpful?\n"
            f"3. Are the Innovative Extensions technically grounded in the papers provided?\n\n"
            f"Respond 'PASS' if perfect, otherwise list required improvements."
        )
        return self.llm_client.generate_text(critic_prompt).strip()

    def _is_feedback_positive(self, feedback: str) -> bool:
        return "PASS" in feedback.upper()

    def _refine_result(self, previous_result: Any, feedback: str, schema_cls: Any) -> Any:
        try:
            prev_json = previous_result.model_dump_json() if hasattr(previous_result, 'model_dump_json') else str(previous_result)
        except:
            prev_json = str(previous_result)

        refine_prompt = (
            f"Original Draft:\n{prev_json}\n\n"
            f"Reviewer Feedback:\n{feedback}\n\n"
            f"Task: Fix the JSON. Ensure it is simpler, more guiding, and corrects all listed issues."
        )
        return self.llm_client.generate_structured(prompt=refine_prompt, schema=schema_cls)