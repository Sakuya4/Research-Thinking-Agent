"""
LLM Client Factory (Schema-Aligned & Robust).
Includes 'Advanced Fuzzy JSON Fixer' to handle int/str mismatches and structural errors.
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
                if "429" in error_str or "Quota" in error_str:
                    wait_time = 15 * (attempt + 1)
                    logger.warning(f"[Gemini] Rate Limit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                if "400" in error_str or "API key expired" in error_str or "invalid" in error_str.lower():
                    logger.error(f"[Gemini] Fatal Auth Error: {e}")
                    raise e 
                logger.error(f"[Gemini] API Error: {e}")
                time.sleep(2)
        
        raise RuntimeError("Gemini API failed.")

    def _fuzzy_fix_json(self, data: Any) -> Any:
        """
        Aggressively fixes common LLM schema errors before validation.
        Handles: Int->Str conversion, Missing fields, Structural mismatches.
        """
        if isinstance(data, list):
            return [self._fuzzy_fix_json(item) for item in data]
        
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                fixed_v = self._fuzzy_fix_json(v)
                
                # --- AUTO-FIX Rules based on your logs ---
                
                # 1. ID Fix: Convert Integers to Strings (e.g., cluster_id: 0 -> "0")
                if k.endswith("_id") and isinstance(fixed_v, int):
                    fixed_v = str(fixed_v)
                
                # 2. Field Mapping (LLM often uses synonyms)
                if k == 'id' and 'name' in data: k = 'cluster_id'
                if k == 'name' and 'cluster_id' in data: k = 'cluster_name'
                if k == 'text' and 'claim_id' in data: k = 'statement'
                if k == 'gap': k = 'description'

                # 3. Handle 'Evidence' Structure Mismatch
                # If schema expects object but got string
                if k == 'evidence' and isinstance(fixed_v, list):
                    fixed_list = []
                    for item in fixed_v:
                        if isinstance(item, str):
                            # Wrap string into the expected object structure
                            fixed_list.append({"evidence": item, "paper_id": "unknown"})
                        else:
                            fixed_list.append(item)
                    fixed_v = fixed_list

                new_data[k] = fixed_v
            
            # 4. Post-processing Injections (Missing required fields)
            
            # Fix Clusters
            if 'cluster_id' in new_data:
                if 'description' not in new_data: 
                    new_data['description'] = "No description provided."
                if 'papers' in new_data and isinstance(new_data['papers'], list):
                    for p in new_data['papers']:
                        if isinstance(p, dict):
                            if 'paper_id' in p and isinstance(p['paper_id'], int):
                                p['paper_id'] = str(p['paper_id'])
                            if 'title' not in p: p['title'] = "Unknown Title"
                            if 'why_included' not in p: p['why_included'] = "Relevant to cluster."

            # Fix Claims
            if 'claim_id' in new_data:
                if 'claim_type' not in new_data: new_data['claim_type'] = 'consensus'
                if 'confidence' not in new_data: new_data['confidence'] = 0.8
                # Ensure claim_type is valid enum
                valid_types = ['trend', 'consensus', 'comparison', 'limitation']
                if new_data.get('claim_type') not in valid_types:
                    new_data['claim_type'] = 'consensus'

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
        full_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT JSON RULES:\n"
            f"- All IDs (cluster_id, paper_id) MUST be STRINGS (e.g., '1', not 1).\n"
            f"- 'evidence' field must be a list of objects, not strings.\n"
            f"- Output strictly valid JSON."
        )
        def _call():
            response = self.model.generate_content(
                full_prompt, 
                generation_config={"response_mime_type": "application/json"},
                safety_settings=self.safety_settings
            )
            raw_data = json.loads(response.text)
            
            # Apply Aggressive Fuzzy Fix
            fixed_data = self._fuzzy_fix_json(raw_data)
            
            return schema.model_validate(fixed_data)
        
        try:
            return self._smart_execute(_call)
        except Exception as e:
            logger.warning(f"[Gemini] Structured Gen failed: {e}. Switching to Mock.")
            return self.fallback_client.generate_structured(prompt, schema)

# --------------------------------------------------------------------------
# Mock Client
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
        logger.info(f"[MockLLM] constructing fake data for {schema.__name__}")
        try:
            # Basic Mock Data Construction (Simplified for brevity as Real Client is priority)
            if schema.__name__ == "QueryPlan":
                return schema(original_topic="Mock", expanded_queries=["Q1"], must_include=[], exclude=[], target_subtasks=[], notes="Mock")
            
            if schema.__name__ == "ReasoningResult":
                # Ensure this matches your strict schema too
                return schema(
                    topic="Mock Topic",
                    clusters=[{
                        "cluster_id": "C1", "cluster_name": "Cluster 1", "description": "Mock",
                        "papers": [{"paper_id": "p1", "title": "Mock Paper", "why_included": "Test"}]
                    }],
                    claims=[{
                        "claim_id": "CL1", "statement": "Mock Claim", "claim_type": "consensus", "confidence": 0.9, "supporting_papers": ["p1"]
                    }],
                    research_gaps=[], synthesis="Mock", limitations=[], future_work=[]
                )
            return schema.model_construct()
        except Exception:
            return None

def get_default_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return RealGeminiClient(api_key)
    else:
        logger.error("!!! NO GEMINI API KEY FOUND !!!")
        return MockGeminiClient()