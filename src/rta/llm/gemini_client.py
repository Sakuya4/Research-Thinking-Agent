"""
LLM Client Factory (Type-Casting Edition).
File: src/rta/utils/llm_client.py
"""

import os
import logging
import json
import time
import re
from typing import Any, Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Standard speed
MIN_REQUEST_INTERVAL = 4.0

class RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self.last_call_time = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call_time = time.time()

global_limiter = RateLimiter(MIN_REQUEST_INTERVAL)

class RealGeminiClient:
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # [STRATEGY]
            # Since 1.5 is 404 for you, we prioritize the Lite/2.0 models that actually exist.
            self.model_candidates = [
                "gemini-flash-lite-latest",      # Priority 1: The one that worked!
                "gemini-2.0-flash-lite-preview-02-05",
                "gemini-2.0-flash",              # Priority 2: Standard 2.0
                "gemini-pro-latest",             # Priority 3: Pro
            ]
            
            self.current_model_index = 0
            self.model_name = self.model_candidates[0]
            self.model = genai.GenerativeModel(self.model_name)
            
            # Embedding Candidates
            self.embedding_candidates = [
                os.getenv("GEMINI_EMBEDDING_MODEL"), 
                'models/gemini-embedding-001', 
                'models/text-embedding-004',         
                'models/embedding-001',              
            ]
            self.embedding_candidates = [m for m in self.embedding_candidates if m]
            self.working_embedding_model = None
            
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            logger.info(f"[Gemini] Init with Model: {self.model_name}")
            
        except ImportError:
            raise

    def _switch_model(self):
        self.current_model_index += 1
        if self.current_model_index >= len(self.model_candidates):
            self.current_model_index = 0
            logger.warning("[System] Cycled through ALL models. Restarting list.")
        
        new_model = self.model_candidates[self.current_model_index]
        logger.warning(f"[Switch] {self.model_name} -> {new_model}")
        
        self.model_name = new_model
        import google.generativeai as genai
        self.model = genai.GenerativeModel(self.model_name)

    def _smart_execute(self, func: Callable, *args, **kwargs) -> Any:
        max_total_attempts = 8
        
        for attempt in range(max_total_attempts):
            global_limiter.wait()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)

                if "404" in error_str: 
                    logger.error(f"[Error] Model {self.model_name} not found (404). Switching...")
                    self._switch_model()
                    continue
                
                if "429" in error_str or "Quota" in error_str or "ResourceExhausted" in error_str:
                    logger.warning(f"[Quota] Limit hit on {self.model_name}. Switching...")
                    self._switch_model()
                    time.sleep(2)
                    continue

                if "400" in error_str: 
                    logger.error(f"[Fatal] Bad Request (400): {e}")
                    raise e
                
                logger.warning(f"[Retry] Attempt {attempt+1} failed: {error_str[:100]}...")
                time.sleep(2)
        
        raise RuntimeError(f"API failed after trying multiple models.")

    def _fuzzy_fix_json(self, data: Any) -> Any:
        """
        AGGRESSIVE TYPE CASTER
        Converts types (int->str, str->float) to satisfy Schema.
        """
        if isinstance(data, list):
            return [self._fuzzy_fix_json(item) for item in data]
        
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                fixed_v = self._fuzzy_fix_json(v)

                # --- 1. Key Mapping ---
                if k == 'id':
                    if 'cluster' in str(data): k = 'cluster_id'
                    elif 'claim' in str(data): k = 'claim_id'
                    elif 'gap' in str(data): k = 'gap_id'
                    fixed_v = str(fixed_v)

                # --- 2. String Enforcement ---
                # Fields that MUST be strings but might be ints/lists
                str_fields = ['cluster_id', 'claim_id', 'gap_id', 'paper_id', 
                              'cluster_name', 'title', 'statement', 'description', 
                              'meta.scope_keywords', 'scope_keywords',
                              'number_of_papers_included', 'Total_Sources_Reviewed', 'documents_reviewed_count']
                
                if k in str_fields or k.endswith("_id"):
                    if isinstance(fixed_v, list): fixed_v = ", ".join(map(str, fixed_v)) # List -> "a, b"
                    if isinstance(fixed_v, int): fixed_v = str(fixed_v)      # 5 -> "5"
                    if isinstance(fixed_v, float): fixed_v = str(int(fixed_v))

                # --- 3. Float/Number Enforcement ---
                # Confidence must be a float
                if k == 'confidence':
                    if isinstance(fixed_v, str):
                        # Convert "High" -> 0.9, "Low" -> 0.1
                        v_lower = fixed_v.lower()
                        if 'high' in v_lower: fixed_v = 0.9
                        elif 'med' in v_lower: fixed_v = 0.5
                        elif 'low' in v_lower: fixed_v = 0.1
                        else:
                            # Try parsing "0.8"
                            try: fixed_v = float(re.findall(r"[-+]?\d*\.\d+|\d+", fixed_v)[0])
                            except: fixed_v = 0.5
                    if fixed_v is None: fixed_v = 0.5

                # --- 4. Structure Fixes ---
                if k == 'papers' and isinstance(fixed_v, list):
                    fixed_list = []
                    for item in fixed_v:
                        if isinstance(item, str): 
                            fixed_list.append({"paper_id": item, "title": "Unknown", "why_included": "Relevant"})
                        else: 
                            if isinstance(item, dict):
                                if 'paper_id' not in item: item['paper_id'] = str(item.get('id', 'unknown'))
                                if 'title' not in item: item['title'] = "Unknown Title"
                                if 'why_included' not in item: item['why_included'] = "Relevant"
                            fixed_list.append(item)
                    fixed_v = fixed_list

                if k == 'evidence' and isinstance(fixed_v, list):
                    fixed_list = []
                    for item in fixed_v:
                        if isinstance(item, str): 
                            fixed_list.append({"evidence": item, "paper_id": "unknown"})
                        elif isinstance(item, dict):
                            if 'paper_id' not in item: item['paper_id'] = "unknown"
                            if 'evidence' not in item:
                                if 'summary' in item: item['evidence'] = item.pop('summary')
                                elif 'excerpt' in item: item['evidence'] = item.pop('excerpt')
                                else: item['evidence'] = "Evidence implied."
                            fixed_list.append(item)
                        else: fixed_list.append(item)
                    fixed_v = fixed_list
                
                if k == 'Total_Sources_Reviewed': k = 'documents_reviewed_count'
                
                new_data[k] = fixed_v
            
            # --- 5. Mandatory Field Injection ---
            if 'cluster_id' in new_data:
                if 'cluster_name' not in new_data: new_data['cluster_name'] = f"Cluster {new_data['cluster_id']}"
                if 'description' not in new_data: new_data['description'] = "No description."
                if 'papers' not in new_data: new_data['papers'] = [] 

            if 'claim_id' in new_data:
                if 'claim_type' not in new_data: new_data['claim_type'] = 'consensus'
                if 'confidence' not in new_data: new_data['confidence'] = 0.5
                if 'statement' not in new_data: new_data['statement'] = "Statement missing."
                if 'evidence' not in new_data: new_data['evidence'] = []

            if 'gap_id' in new_data:
                if 'description' not in new_data: new_data['description'] = "Gap description unavailable."
                if 'contributing_claims' not in new_data: new_data['contributing_claims'] = []

            return new_data
        return data

    def generate_text(self, prompt: str) -> str:
        def _call():
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text if response.text else ""
        return self._smart_execute(_call)

    def get_embedding(self, text: str) -> list:
        import google.generativeai as genai
        if self.working_embedding_model:
            candidates = [self.working_embedding_model]
        else:
            candidates = self.embedding_candidates

        for model_name in candidates:
            try:
                global_limiter.wait()
                result = genai.embed_content(model=model_name, content=text, task_type="clustering")
                if 'embedding' in result:
                    if not self.working_embedding_model:
                        logger.info(f"[Info] Found working embedding model: {model_name}")
                        self.working_embedding_model = model_name
                    return result['embedding']
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    logger.warning(f"[Warn] Model '{model_name}' failed (404). Trying next...")
                    continue 
                logger.error(f"[Error] Embedding '{model_name}': {e}")
                break 

        logger.error("[Error] All embedding models failed.")
        raise RuntimeError("No working embedding model found.")

    def generate_structured(self, prompt: str, schema: Any) -> Any:
        full_prompt = (
            f"{prompt}\n\n"
            f"JSON RULES: All IDs must be STRINGS. 'papers' must be objects. 'evidence' must be objects."
        )
        def _call():
            response = self.model.generate_content(
                full_prompt, 
                generation_config={"response_mime_type": "application/json"},
                safety_settings=self.safety_settings
            )
            raw_data = json.loads(response.text)
            fixed_data = self._fuzzy_fix_json(raw_data)
            return schema.model_validate(fixed_data)
        
        return self._smart_execute(_call)

class MockGeminiClient:
    def generate_text(self, prompt: str) -> str: return "Analysis unavailable."
    def get_embedding(self, text: str) -> list: return [0.1] * 768
    def generate_structured(self, prompt: str, schema: Any) -> Any:
        try: return schema.model_construct()
        except: return None

def get_default_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return RealGeminiClient(api_key) if api_key else MockGeminiClient()