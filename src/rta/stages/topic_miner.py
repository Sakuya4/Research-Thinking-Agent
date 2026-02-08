"""
Topic Mining Module with Resilient K-Means Clustering.
File: src/rta/stages/topic_miner.py
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from sklearn.cluster import KMeans

try:
    from rta.schemas.topic_structuring import TopicStructuringResult, TopicCluster
    HAS_REAL_SCHEMA = True
except ImportError:
    HAS_REAL_SCHEMA = False
    class TopicStructuringResult: pass
    class TopicCluster: pass

logger = logging.getLogger(__name__)

class TopicMiningService:
    def __init__(self, llm_client: Any, embedding_model: str = "models/text-embedding-004"):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.MAX_CLUSTERS = 5
        self.MIN_CLUSTERS = 2

    def execute(self, papers: List[Any]) -> Any:
        if not papers:
            logger.warning("[TopicMiner] No papers provided.")
            return None

        logger.info(f"[TopicMiner] Analyzing {len(papers)} papers via embeddings...")

        try:
            # Step 1: Generate Embeddings
            valid_papers, embeddings = self._generate_embeddings(papers)
            if not valid_papers or len(embeddings) < self.MIN_CLUSTERS:
                raise ValueError("Insufficient embeddings for clustering.")

            # Step 2: K-Means
            n_clusters = self._determine_optimal_clusters(len(valid_papers))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Step 3: Synthesis
            clusters = self._synthesize_cluster_labels(valid_papers, labels, n_clusters)
            
        except Exception as e:
            logger.warning(f"[TopicMiner] Clustering failed: {e}. Executing fallback strategy...")
            # Emergency Fallback: Merge all papers into one cluster
            clusters = self._create_fallback_clusters(papers)

        if HAS_REAL_SCHEMA:
            return TopicStructuringResult(
                clusters=clusters,
                main_directions=[c.name for c in clusters],
                recommended_pipeline=["Analysis"]
            )
        return clusters

    def _generate_embeddings(self, papers: List[Any]) -> tuple:
        embeddings = []
        valid_papers = []
        for paper in papers:
            content = getattr(paper, 'abstract', getattr(paper, 'title', ''))
            try:
                # Calls the new get_embedding method in RealGeminiClient
                vector = self.llm_client.get_embedding(content)
                embeddings.append(vector)
                valid_papers.append(paper)
            except Exception:
                continue
        return valid_papers, np.array(embeddings)

    def _determine_optimal_clusters(self, num_papers: int) -> int:
        return min(self.MAX_CLUSTERS, int(num_papers / 3) + 1)

    def _synthesize_cluster_labels(self, papers: List[Any], labels: np.ndarray, n_clusters: int) -> List[Any]:
        final_clusters = []
        cluster_map = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            cluster_map[label].append(papers[idx])

        for label_id, cluster_papers in cluster_map.items():
            titles = "\n".join([f"- {p.title}" for p in cluster_papers])
            prompt = f"Name this research theme based on these titles:\n{titles}\nReturn ONLY JSON: {{\"name\": \"...\", \"description\": \"...\"}}"
            
            try:
                res = self.llm_client.generate_text(prompt)
                import json
                data = json.loads(res.strip().strip('`').replace('json', ''))
            except:
                data = {"name": f"Theme {label_id+1}", "description": "Grouped academic results."}

            cluster_obj = TopicCluster(
                cluster_id=f"cluster_{label_id}",
                name=data.get("name", "Research Cluster"),
                paper_ids=[getattr(p, 'paper_id', str(i)) for i, p in enumerate(cluster_papers)],
                description=data.get("description", ""),
                keywords=["Research"],
                typical_methods=["Analysis"]
            )
            final_clusters.append(cluster_obj)
        return final_clusters

    def _create_fallback_clusters(self, papers: List[Any]) -> List[Any]:
        """Creates a single safe cluster if AI/Embedding fails."""
        return [TopicCluster(
            cluster_id="cluster_0",
            name="General Research",
            paper_ids=[getattr(p, 'paper_id', str(i)) for i, p in enumerate(papers)],
            description="Synthesis of retrieved documents.",
            keywords=["General"],
            typical_methods=["Review"]
        )]