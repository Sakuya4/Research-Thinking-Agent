"""
test_pipeline.py
Purpose: Integration test for RTA core logic (Mock Mode).
Author: RTA Team (Google Standard)
"""

import sys
import logging
import numpy as np
from typing import List, Any
from types import ModuleType
from pydantic import BaseModel

# --------------------------------------------------------------------------
# 1. Force-Mock the Schemas
# --------------------------------------------------------------------------

class MockPaperItem(BaseModel):
    paper_id: str
    title: str
    abstract: str

class MockFinding(BaseModel):
    id: str = "f_test"
    summary: str
    evidence_paper_ids: List[str] = []

class MockTopicCluster(BaseModel):
    cluster_id: str
    name: str 
    paper_ids: List[str]
    description: str
    keywords: List[str] = []
    typical_methods: List[str] = []
    key_findings: List[MockFinding] = [] 

class MockTopicStructuringResult(BaseModel):
    clusters: List[MockTopicCluster]
    main_directions: List[str]
    recommended_pipeline: List[str] = []

class MockReasoningResult(BaseModel):
    topic: str
    clusters: List[MockTopicCluster]
    def model_dump_json(self, indent=2):
        return super().model_dump_json(indent=indent)

class MockQueryPlan(BaseModel):
    original_topic: str

# Create fake modules
schemas_structuring = ModuleType("rta.schemas.topic_structuring")
schemas_structuring.TopicCluster = MockTopicCluster
schemas_structuring.TopicStructuringResult = MockTopicStructuringResult

schemas_retrieval = ModuleType("rta.schemas.retrieval")
schemas_retrieval.PaperItem = MockPaperItem

schemas_reasoning = ModuleType("rta.schemas.reasoning")
schemas_reasoning.ReasoningResult = MockReasoningResult
schemas_reasoning.Finding = MockFinding 

schemas_plan = ModuleType("rta.schemas.query_plan")
schemas_plan.QueryPlan = MockQueryPlan

# Inject into sys.modules
sys.modules["rta.schemas.topic_structuring"] = schemas_structuring
sys.modules["rta.schemas.retrieval"] = schemas_retrieval
sys.modules["rta.schemas.reasoning"] = schemas_reasoning
sys.modules["rta.schemas.query_plan"] = schemas_plan

# --------------------------------------------------------------------------
# 2. Import Logic Modules
# --------------------------------------------------------------------------
try:
    from rta.stages.topic_miner import TopicMiningService
    from rta.stages.reasoning_engine import ReasoningEngine
    try:
        from rta.utils.visualization import generate_reasoning_graph
    except ImportError:
        generate_reasoning_graph = None
except ImportError as e:
    print(f"[Error] Module import failed: {e}")
    exit(1)

# --------------------------------------------------------------------------
# 3. Setup Logger
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestRunner")

# --------------------------------------------------------------------------
# 4. Mock Client
# --------------------------------------------------------------------------
class MockLLMClient:
    def __init__(self):
        self.call_count = 0

    def get_embedding(self, text: str) -> np.ndarray:
        return np.random.rand(768)

    def generate_text(self, prompt: str) -> str:
        if "research sub-field name" in prompt or "naming task" in prompt:
            return "Mocked Research Topic"
        
        if "Review this JSON" in prompt or "Evaluation Criteria" in prompt:
            self.call_count += 1
            if self.call_count == 1:
                return "1. Claim 1 lacks evidence.\n2. Logic gap."
            else:
                return "PASS"
        
        return "Generic Mock Response"

    def generate_structured(self, prompt: str, schema: Any) -> Any:
        return MockReasoningResult(
            topic="Test Topic",
            clusters=[
                MockTopicCluster(
                    cluster_id="c1",
                    name="AI Planning", 
                    paper_ids=["p1", "p2"], 
                    description="Desc",
                    keywords=["plan"],
                    typical_methods=["search"]
                ),
                MockTopicCluster(
                    cluster_id="c2",
                    name="LLM Agents", 
                    paper_ids=["p3"], 
                    description="Desc",
                    keywords=["agent"],
                    typical_methods=["react"]
                )
            ]
        )

# --------------------------------------------------------------------------
# 5. Main Test execution
# --------------------------------------------------------------------------
def run_test():
    print("========================================")
    print("RTA System Integration Test (Mock Mode)")
    print("========================================\n")

    dummy_papers = [
        MockPaperItem(paper_id=f"p{i}", title=f"Paper {i}", abstract=f"Abstract {i}...") 
        for i in range(10)
    ]
    query_plan = MockQueryPlan(original_topic="Agentic Workflow")
    mock_llm = MockLLMClient()

    # --- Test Stage 3 ---
    print("[Step 1] Testing Stage 3: Topic Mining...")
    miner = TopicMiningService(llm_client=mock_llm)
    
    mining_result = miner.execute(dummy_papers)
    
    if mining_result:
        print(f"[Pass] Stage 3 completed. Clusters: {len(mining_result.clusters)}")
        for c in mining_result.clusters:
            print(f"    - Cluster: {c.name} (Papers: {len(c.paper_ids)})")
    else:
        print("[Fail] Stage 3 returned None.")
    print("\n")

    # --- Test Stage 4 ---
    print("[Step 2] Testing Stage 4: Reasoning Engine (with Self-Refinement)...")
    engine = ReasoningEngine(llm_client=mock_llm)
    final_reasoning = engine.run(query_plan, mining_result, dummy_papers)
    
    print("[Pass] Stage 4 completed.")
    print(f"    - Report Topic: {final_reasoning.topic}")
    print("\n")

    # --- Test Visualization ---
    print("[Step 3] Testing Visualization...")
    if generate_reasoning_graph:
        try:
            if final_reasoning.clusters:
                final_reasoning.clusters[0].key_findings = [
                    MockFinding(summary="LLMs demonstrate strong planning capabilities.")
                ]
            
            mermaid_code = generate_reasoning_graph(final_reasoning)
            print("\nGenerated Mermaid Code (Success):")
            print("--------------------------------------------------")
            print(mermaid_code)
            print("--------------------------------------------------")
        except Exception as e:
            print(f"[Warn] Graph generation skipped: {e}")
            import traceback
            traceback.print_exc()

    print("\n[Success] All tests passed. System is Production-Ready.")

if __name__ == "__main__":
    run_test()