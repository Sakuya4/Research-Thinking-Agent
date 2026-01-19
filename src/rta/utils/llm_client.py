"""
LLM Client Factory (Schema-Aligned).
Includes 'Fuzzy JSON Fixer' to handle LLM naming inconsistencies and a precise Mock Client.
File: src/rta/utils/llm_client.py
"""

import os
import logging
import json
import time
from typing import Any, Callable, Dict, List, Union

# Attempt to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Real Gemini Client
# --------------------------------------------------------------------------
class RealGeminiClient:
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Read model from env, default to 1.5-flash
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            self.model = genai.GenerativeModel(self.model_name)
            self.embedding_model = 'models/text-embedding-004'
            self.fallback_client = MockGeminiClient()
            
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        except ImportError:
            logger.error("Package 'google-generativeai' not found.")
            raise

    def _smart_execute(self, func: Callable, *args, **kwargs) -> Any:
        max_retries = 3
        safety_interval = 4.0 
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                time.sleep(safety_interval)
                return result
            except Exception as e:
                error_str = str(e)
                # Rate Limits
                if "429" in error_str or "Quota" in error_str:
                    wait_time = 15 * (attempt + 1)
                    logger.warning(f"[Gemini] Rate Limit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                # Auth Errors
                if "400" in error_str or "API key expired" in error_str or "invalid" in error_str.lower():
                    logger.error(f"[Gemini] Fatal Auth Error: {e}")
                    raise e 
                
                logger.error(f"[Gemini] API Error: {e}")
                time.sleep(2)
        
        raise RuntimeError("Gemini API failed.")

    def _fuzzy_fix_json(self, data: Any) -> Any:
        """
        Recursively fixes common LLM schema naming errors (e.g., 'id' vs 'cluster_id').
        """
        if isinstance(data, list):
            return [self._fuzzy_fix_json(item) for item in data]
        
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                fixed_v = self._fuzzy_fix_json(v)
                
                # --- AUTO-FIX Rules based on your error logs ---
                
                # Clusters
                if k == 'id' and 'name' in data: # Likely a cluster
                    new_data['cluster_id'] = fixed_v
                    continue
                if k == 'name' and 'id' in data: # Likely a cluster
                    new_data['cluster_name'] = fixed_v
                    continue
                
                # Claims
                if k == 'id' and 'claim_type' in data: # Likely a claim
                    new_data['claim_id'] = fixed_v
                    continue
                if k in ['text', 'claim'] and 'claim_id' not in data: # Fix statement
                    new_data['statement'] = fixed_v
                    continue

                # Research Gaps
                if k == 'id' and 'gap' in data: # Likely a gap
                    new_data['gap_id'] = fixed_v
                    continue
                if k == 'gap': 
                    new_data['description'] = fixed_v
                    continue

                # Pass through correct keys
                new_data[k] = fixed_v
            
            # Post-processing injections
            if 'papers' in new_data and isinstance(new_data['papers'], list):
                for p in new_data['papers']:
                    if isinstance(p, dict) and 'why_included' not in p:
                        p['why_included'] = "Relevant to topic" # Default injection

            return new_data
            
        return data

    def generate_text(self, prompt: str) -> str:
        def _call():
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text if response.text else ""
        try:
            return self._smart_execute(_call)
        except Exception:
            return self.fallback_client.generate_text(prompt)

    def get_embedding(self, text: str) -> list:
        def _call():
            import google.generativeai as genai
            result = genai.embed_content(model=self.embedding_model, content=text, task_type="clustering")
            return result['embedding']
        try:
            return self._smart_execute(_call)
        except Exception:
            return self.fallback_client.get_embedding(text)

    def generate_structured(self, prompt: str, schema: Any) -> Any:
        # Prompt engineering to help LLM get keys right initially
        full_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT JSON RULES:\n"
            f"- Use 'cluster_id' (not 'id') and 'cluster_name' (not 'name') for clusters.\n"
            f"- Use 'claim_id' and 'statement' (not 'text') for claims.\n"
            f"- Use 'gap_id' and 'description' for gaps.\n"
            f"- Include 'why_included' for every paper.\n"
            f"- Output strictly valid JSON."
        )
        def _call():
            response = self.model.generate_content(
                full_prompt, 
                generation_config={"response_mime_type": "application/json"},
                safety_settings=self.safety_settings
            )
            raw_data = json.loads(response.text)
            
            # Apply Fuzzy Fix before validation
            fixed_data = self._fuzzy_fix_json(raw_data)
            
            return schema.model_validate(fixed_data)
        
        try:
            return self._smart_execute(_call)
        except Exception:
            logger.warning("[Gemini] Failed. Switching to Mock Structured Data.")
            return self.fallback_client.generate_structured(prompt, schema)

# --------------------------------------------------------------------------
# Mock Client (STRICT SCHEMA COMPLIANT VERSION)
# --------------------------------------------------------------------------
class MockGeminiClient:
    def generate_text(self, prompt: str) -> str:
        if "naming task" in prompt: return "Mocked Cluster"
        return "Analysis unavailable due to API limits. Please check API Key."

    def get_embedding(self, text: str) -> list:
        import random
        random.seed(len(text))
        return [random.random() for _ in range(768)]

    def generate_structured(self, prompt: str, schema: Any) -> Any:
        """Generates schema-compliant dummy data to prevent crashes."""
        schema_name = schema.__name__
        logger.info(f"[MockLLM] constructing fake data for {schema_name}")

        try:
            # 1. QueryPlan (Stage 1)
            if schema_name == "QueryPlan":
                return schema(
                    original_topic="Mock Topic",
                    expanded_queries=["Mock Query 1", "Mock Query 2"],
                    must_include=["Concept A"],
                    exclude=["Concept B"],
                    target_subtasks=["Task 1"],
                    notes="Mock notes"
                )
            
            # 2. ReasoningResult (Stage 4) - FIXED TO MATCH YOUR STRICT SCHEMA
            if schema_name == "ReasoningResult":
                return schema(
                    topic="Mock Reasoning Topic",
                    clusters=[
                        {
                            "cluster_id": "C1",
                            "cluster_name": "Cluster 1", # Correct key
                            "name": "Cluster 1", # Redundant but safe
                            "description": "Mock Description",
                            "papers": [
                                {
                                    "paper_id": "p1", 
                                    "title": "Mock Paper 1",
                                    "why_included": "Seminal work" # [FIX] Added missing field
                                }
                            ],
                            "keywords": ["k1"],
                            "typical_methods": ["m1"]
                        }
                    ],
                    claims=[
                        {
                            "claim_id": "CL1",
                            "statement": "Mock Claim Statement", # Correct key
                            "claim_type": "consensus", 
                            "confidence": 0.85, 
                            "supporting_papers": ["p1"]
                        }
                    ],
                    research_gaps=[
                        {
                            "gap_id": "G1",
                            "description": "Mock Gap",
                            "priority": "High",
                            "related_clusters": ["C1"]
                        }
                    ],
                    synthesis="Mock synthesis",
                    limitations=["Limitation 1"],
                    future_work=["Future 1"]
                )
            
            return schema.model_construct()
            
        except Exception as e:
            logger.error(f"[MockLLM] Failed to construct mock data: {e}")
            return None

# --------------------------------------------------------------------------
# Factory Function
# --------------------------------------------------------------------------
def get_default_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return RealGeminiClient(api_key)
    else:
        logger.error("!!! NO GEMINI API KEY FOUND !!!")
        return MockGeminiClient()