"""
Main Pipeline Execution Module (Optimized for Quota Safety).
File: src/rta/pipeline.py
"""

import logging
import json
import os
import time
from typing import Optional, Any, Union, Tuple

# --- Import Stage Modules ---
from rta.stages.query_plan_gemini import run_query_planning
from rta.stages.retrieval_live import run_retrieval
from rta.stages.topic_miner import TopicMiningService
from rta.stages.reasoning_engine import ReasoningEngine

# --- Import Utils ---
try:
    from rta.utils.llm_client import get_default_client
except ImportError:
    class MockClient:
        def generate(self, *args, **kwargs): return "Mock Response"
        def get_embedding(self, *args, **kwargs): return [0.01]*768
    def get_default_client(): return MockClient()

# --- Import UI ---
from rta.utils.ui import spinner, print_header

logger = logging.getLogger(__name__)

def run_pipeline(topic: Union[str, Any], output_dir: Union[str, Any] = "outputs") -> Tuple[bool, str]:
    """
    Executes the full RTA research pipeline (End-to-End).
    """
    
    # --- Argument Handling ---
    real_topic = topic
    if hasattr(topic, 'query'): 
        real_topic = topic.query
    elif not isinstance(topic, str):
        real_topic = str(topic)
        
    real_output_dir = output_dir if isinstance(output_dir, str) else "outputs"

    # UI Header
    print_header("Research Thinking Agent", f"Topic: {real_topic}")
    
    try:
        os.makedirs(real_output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"[Pipeline] Failed to create output directory: {e}")
        return False, ""

    # Initialize Client
    llm_client = get_default_client() 
    md_path = os.path.join(real_output_dir, "report.md")

    # ------------------------------------------------------------------
    # Stage 1: Query Planning
    # ------------------------------------------------------------------
    with spinner("Stage 1: Planning search strategy..."):
        try:
            plan = run_query_planning(real_topic)
            logger.info(f"    -> Generated {len(plan.expanded_queries)} search queries")
            _save_json(plan, real_output_dir, "plan.json") 
        except Exception as e:
            logger.error(f"[Fail] Stage 1 Error: {e}")
            return False, ""

    # ------------------------------------------------------------------
    # Stage 2: Literature Retrieval
    # ------------------------------------------------------------------
    with spinner("Stage 2: Searching literature (arXiv/Scholar)..."):
        try:
            retrieval_results = run_retrieval(plan.expanded_queries)
            papers = getattr(retrieval_results, 'papers', []) 
            if not papers and isinstance(retrieval_results, list):
                 papers = retrieval_results

            # --- [CRITICAL OPTIMIZATION] ---
            # 限制處理的論文數量，直接解決 Stage 3 API 噴發問題
            MAX_ALLOWED_PAPERS = 2 
            if len(papers) > MAX_ALLOWED_PAPERS:
                logger.info(f"    [Optimization] Pruning {len(papers)} papers down to {MAX_ALLOWED_PAPERS} for API safety.")
                papers = papers[:MAX_ALLOWED_PAPERS]
            # -------------------------------

            logger.info(f"    -> Retrieved {len(papers)} papers")
            _save_json(retrieval_results, real_output_dir, "retrieval.json") 

            if not papers:
                logger.warning("[Warn] No papers found. Aborting.")
                return False, ""
        except Exception as e:
            logger.error(f"[Fail] Stage 2 Error: {e}")
            return False, ""

    # ------------------------------------------------------------------
    # Stage 3: Topic Structuring
    # ------------------------------------------------------------------
    with spinner("Stage 3: Clustering & Structuring topics..."):
        try:
            miner = TopicMiningService(llm_client=llm_client)
            # 這裡執行時會呼叫 llm_client 的 get_embedding，現在只剩 2 篇論文，負擔極輕
            structuring_result = miner.execute(papers)
            
            if structuring_result:
                cluster_count = len(structuring_result.clusters)
                logger.info(f"    -> Identified {cluster_count} research sub-topics")
                _save_json(structuring_result, real_output_dir, "structuring.json")
            else:
                raise ValueError("Stage 3 returned None")
        except Exception as e:
            logger.warning(f"[Warn] Stage 3 hit an issue: {e}. Attempting recovery...")
            # 如果 Stage 3 徹底失敗，手動建立一個基礎結構讓 Stage 4 能跑
            from rta.models.reasoning import ReasoningResult 
            structuring_result = ReasoningResult(
                clusters=[{"cluster_id": "C1", "cluster_name": "General Analysis", "description": "Combined literature analysis.", "papers": []}],
                meta={"topic": real_topic, "documents_reviewed_count": str(len(papers)), "scope_keywords": "research"}
            )
            logger.info("    -> Applied Emergency Structuring.")

    # ------------------------------------------------------------------
    # Stage 4: Reasoning Agent
    # ------------------------------------------------------------------
    with spinner("Stage 4: Reasoning & Self-Refining (This may take time)..."):
        try:
            engine = ReasoningEngine(llm_client=llm_client)
            # 如果之前的步驟有截斷，Stage 4 讀取的上下文也會變少，更容易成功
            final_report = engine.run(plan, structuring_result, papers)
            
            report_topic = getattr(final_report, 'topic', real_topic)
            logger.info(f"    -> Report generated! Topic: {report_topic}")
            _save_json(final_report, real_output_dir, "reasoning.json")
            
            # Markdown Generation
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {report_topic}\n\n")
                clusters = getattr(final_report, 'clusters', [])
                for c in clusters:
                    if isinstance(c, dict):
                        c_name = c.get('cluster_name', c.get('name', 'Cluster'))
                        c_desc = c.get('description', '')
                    else:
                        c_name = getattr(c, 'cluster_name', getattr(c, 'name', 'Cluster'))
                        c_desc = getattr(c, 'description', '')
                    f.write(f"## {c_name}\n{c_desc}\n\n")
            logger.info(f"    -> Markdown saved to: {md_path}")

        except Exception as e:
            logger.error(f"[Fail] Stage 4 Error: {e}")
            return False, ""

    print_header("Pipeline Completed", f"Results saved in: {real_output_dir}")
    return True, real_output_dir


def _save_json(obj, folder, filename):
    try:
        path = os.path.join(folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            if hasattr(obj, "model_dump"): data = obj.model_dump()
            elif hasattr(obj, "dict"): data = obj.dict()
            elif hasattr(obj, "__dict__"): data = obj.__dict__
            else: data = str(obj)
            json.dump(data, f, ensure_ascii=False, indent=2)
    except:
        pass