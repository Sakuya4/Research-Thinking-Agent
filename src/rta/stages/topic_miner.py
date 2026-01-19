"""
Topic Mining Module based on Embedding Clustering.
File: src/rta/stages/topic_miner.py
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from sklearn.cluster import KMeans

# Attempt to import real schemas, but allow for fallback
try:
    from rta.schemas.topic_structuring import TopicStructuringResult, TopicCluster
    from rta.schemas.retrieval import PaperItem
    HAS_REAL_SCHEMA = True
except ImportError:
    HAS_REAL_SCHEMA = False
    class TopicStructuringResult: pass
    class TopicCluster: pass
    PaperItem = Any

logger = logging.getLogger(__name__)

class TopicMiningService:
    """
    Service for clustering research papers into structured topics using embeddings.
    """

    def __init__(self, llm_client: Any, embedding_model: str = "models/text-embedding-004"):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.MAX_CLUSTERS = 8 
        self.MIN_CLUSTERS = 2

    def execute(self, papers: List[Any]) -> Any:
        if not papers:
            logger.warning("[TopicMiner] No papers provided.")
            return None

        logger.info(f"[TopicMiner] Starting mining for {len(papers)} papers.")

        # Step 1: Vectorization
        valid_papers, embeddings = self._generate_embeddings(papers)
        if not valid_papers:
            raise ValueError("No valid embeddings generated.")

        # Step 2: Clustering
        n_clusters = self._determine_optimal_clusters(len(valid_papers))
        logger.info(f"[TopicMiner] Optimal cluster count: {n_clusters}")
        labels = self._perform_clustering(embeddings, n_clusters)

        # Step 3: Synthesis
        logger.info("[TopicMiner] Generating labels...")
        clusters = self._synthesize_cluster_labels(valid_papers, labels, n_clusters)

        logger.info(f"[TopicMiner] Completed. Generated {len(clusters)} clusters.")
        
        # Determine return type based on available schema
        if HAS_REAL_SCHEMA:
            return TopicStructuringResult(
                clusters=clusters,
                main_directions=[c.name for c in clusters],
                recommended_pipeline=["Standard Analysis"]
            )
        else:
            # Fallback Mock Result
            class MockResult:
                def __init__(self, c, m): 
                    self.clusters = c
                    self.main_directions = m
                    self.recommended_pipeline = ["Mock Pipeline"]
            return MockResult(clusters, [c.name for c in clusters])

    def _generate_embeddings(self, papers: List[Any]) -> tuple[List[Any], np.ndarray]:
        embeddings = []
        valid_papers = []

        for paper in papers:
            content = getattr(paper, 'abstract', getattr(paper, 'description', ''))
            if not content: content = getattr(paper, 'title', 'No content')

            try:
                vector = self.llm_client.get_embedding(content) 
                embeddings.append(vector)
                valid_papers.append(paper)
            except Exception as e:
                logger.warning(f"[TopicMiner] Embed failed for paper: {e}")

        return valid_papers, np.array(embeddings)

    def _determine_optimal_clusters(self, num_papers: int) -> int:
        if num_papers < self.MIN_CLUSTERS: return 1
        heuristic = int((num_papers / 2) ** 0.5)
        return max(self.MIN_CLUSTERS, min(heuristic, self.MAX_CLUSTERS))

    def _perform_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        return kmeans.labels_

    def _synthesize_cluster_labels(self, papers: List[Any], labels: np.ndarray, n_clusters: int) -> List[Any]:
        final_clusters = []
        cluster_map = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            cluster_map[label].append(papers[idx])

        # Dynamic Schema Resolution
        if HAS_REAL_SCHEMA:
            from rta.schemas.topic_structuring import TopicCluster
        else:
            # Fallback definition
            class TopicCluster:
                def __init__(self, cluster_id, name, paper_ids, description, keywords, typical_methods):
                    self.cluster_id = cluster_id
                    self.name = name
                    self.paper_ids = paper_ids
                    self.description = description
                    self.keywords = keywords
                    self.typical_methods = typical_methods

        for label_id, cluster_papers in cluster_map.items():
            if not cluster_papers: continue
            
            prompt = f"Cluster {label_id} naming task"
            topic_name_str = self.llm_client.generate_text(prompt).strip()
            
            paper_ids = [getattr(p, 'paper_id', str(i)) for i, p in enumerate(cluster_papers)]
            
            cluster_data = {
                "cluster_id": f"cluster_{label_id}",
                "name": topic_name_str,
                "paper_ids": paper_ids,
                "description": f"Group focused on {topic_name_str}",
                "keywords": ["AI", "Research"],        
                "typical_methods": ["Method Analysis"] 
            }

            try:
                # Try to instantiate the object
                cluster_obj = TopicCluster(**cluster_data)
                final_clusters.append(cluster_obj)
            except TypeError as e:
                # Safety catch: If 'paper_ids' causes error in Real Schema, try without it
                if "paper_ids" in str(e):
                    logger.warning(f"[TopicMiner] Schema mismatch on 'paper_ids'. Retrying without it.")
                    cluster_data.pop("paper_ids")
                    final_clusters.append(TopicCluster(**cluster_data))
                else:
                    logger.error(f"[TopicMiner] Schema Validation Error: {e}")
                    # If strictly using fallback, just append raw data wrapper
                    if not HAS_REAL_SCHEMA:
                        final_clusters.append(TopicCluster(**cluster_data))

        return final_clusters