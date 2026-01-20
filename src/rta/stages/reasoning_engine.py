"""
Reasoning Engine with Self-Refinement.
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
    # Fallback for compilation checking
    pass

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Orchestrates the generation of research insights with an iterative refinement loop.
    """
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self.MAX_RETRIES = 2 

    def run(self, query_plan: Any, clustering_result: Any, papers: list) -> Any:
        # [FIX] Robustly extract topic string handling different Schema versions
        topic_str = self._extract_topic_str(query_plan)
        
        logger.info(f"[Reasoning] Started for topic: {topic_str}")

        # Phase 1: Drafting
        logger.info("[Reasoning] Phase 1: Generating initial draft...")
        draft_prompt = self._build_draft_prompt(topic_str, clustering_result, papers)
        
        # Determine schema class dynamically if needed
        try:
            from rta.schemas.reasoning import ReasoningResult
            schema_cls = ReasoningResult
        except ImportError:
            # If strictly mocking, we might not have the class, but usually pipeline handles this.
            logger.error("[Reasoning] Could not import ReasoningResult schema.")
            return None

        current_result = self.llm_client.generate_structured(
            prompt=draft_prompt, 
            schema=schema_cls
        )

        # Inject topic if missing (Common issue with LLM generation)
        if current_result and hasattr(current_result, 'topic') and not current_result.topic:
            try:
                current_result.topic = topic_str
            except:
                pass

        # Phase 2: Refinement
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
        """Helper to safely get the topic string from various QueryPlan schemas."""
        # Try 'original_topic' (Our preferred)
        val = getattr(plan, 'original_topic', None)
        if val: return val
        
        # Try 'topic' (Common alternative)
        val = getattr(plan, 'topic', None)
        if val: return val
        
        # Try 'query' (Another common alternative)
        val = getattr(plan, 'query', None)
        if val: return val
        
        return "Unknown Research Topic"

    def _build_draft_prompt(self, topic: str, clusters: Any, papers: list) -> str:
        """Constructs the prompt for the initial draft generation with EXTENSION capabilities."""
        # Extract cluster names for context
        cluster_names = []
        if hasattr(clusters, 'clusters'):
            for c in clusters.clusters:
                name = getattr(c, 'name', getattr(c, 'topic_name', 'Unnamed Cluster'))
                cluster_names.append(name)
        
        # [UPDATED] Added explicit instructions for "Innovative Extensions" and "Applications"
        return (
            f"You are a visionary Lead Researcher. Your task is to analyze the retrieved literature and conduct a deep synthesis.\n"
            f"User Topic: '{topic}'\n"
            f"The system has retrieved {len(papers)} papers, categorized into: {', '.join(cluster_names)}.\n\n"
            
            f"Please generate a 'ReasoningResult' JSON with the following mindset:\n"
            f"1. **Synthesis**: Summarize the core findings and consensus from the papers.\n"
            f"2. **Critical Gaps**: Identify what is missing in the current literature.\n"
            f"3. **INNOVATIVE EXTENSIONS (Crucial)**: \n"
            f"   - Based on the retrieved methods, how can we extend this technology further?\n"
            f"   - Think explicitly about downstream applications and practical use cases. \n"
            f"   - Example: If the topic is 'Handheld Ultrasound', do not just stop at 'image quality'. \n"
            f"     Extend it to 'AI-guided needle insertion', 'Automated LVEF calculation', or 'Real-time pathology detection'.\n"
            f"   - Propose 3 specific, technically grounded future directions that combine the user's topic with the retrieved evidence.\n\n"
            
            f"Ensure the output is strictly valid JSON matching the schema."
        )

    def _critique_result(self, result: Any) -> str:
        """Acts as the 'Quality Evaluator' agent."""
        # Robust dump to JSON
        try:
            if hasattr(result, 'model_dump_json'):
                result_json = result.model_dump_json(indent=2)
            elif hasattr(result, 'json'):
                result_json = result.json()
            else:
                result_json = str(result)
        except:
            result_json = str(result)

        critic_prompt = (
            f"You are a strict Senior Research Fellow. Review the following structured research output:\n"
            f"```json\n{result_json}\n```\n\n"
            f"Evaluation Criteria:\n"
            f"1. Are all claims supported by evidence?\n"
            f"2. Are the identified research gaps logical?\n"
            f"3. Is the classification of papers consistent?\n\n"
            f"Instruction:\n"
            f"- If the output is high quality, respond exactly with 'PASS'.\n"
            f"- If there are issues, provide a numbered list of specific corrections required."
        )
        
        return self.llm_client.generate_text(critic_prompt).strip()

    def _is_feedback_positive(self, feedback: str) -> bool:
        """Determines if the critic is satisfied."""
        if not feedback: return True # Fail open if empty
        return "PASS" in feedback.upper()

    def _refine_result(self, previous_result: Any, feedback: str, schema_cls: Any) -> Any:
        """Regenerates the result based on the critic's feedback."""
        # Robust dump
        try:
            prev_json = previous_result.model_dump_json() if hasattr(previous_result, 'model_dump_json') else str(previous_result)
        except:
            prev_json = str(previous_result)

        refine_prompt = (
            f"Original Draft:\n{prev_json}\n\n"
            f"Reviewer Feedback:\n{feedback}\n\n"
            f"Task: Regenerate the ReasoningResult JSON to address ALL feedback points.\n"
            f"Maintain the same structure."
        )
        
        return self.llm_client.generate_structured(
            prompt=refine_prompt,
            schema=schema_cls
        )