"""
Topic Mining Module based on Embedding Clustering.

This module implements the 'Topic Mining & Clustering' agent described in 
the "Agentic AutoSurvey" paper. It utilizes semantic embeddings and 
K-Means clustering to organize research papers into coherent topics.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans

# Assuming these schemas exist in your project
from rta.schemas.topic_structuring import TopicStructuringResult, TopicCluster
from rta.schemas.retrieval import PaperItem

# Abstract interface for LLM interactions (Dependency Injection pattern)
# You should replace 'Any' with your actual LLM client class type
from typing import Any 

logger = logging.getLogger(__name__)

class TopicMiningService:
    """
    Service for clustering research papers into structured topics using embeddings.
    """

    def __init__(self, llm_client: Any, embedding_model: str = "models/text-embedding-004"):
        """
        Initialize the TopicMiningService.

        Args:
            llm_client: An instance of the wrapper for Gemini API interactions.
            embedding_model: The model identifier for generating text embeddings.
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        # Configuration constant: Maximum number of clusters to attempt
        self.MAX_CLUSTERS = 8 
        self.MIN_CLUSTERS = 3

    def execute(self, papers: List[PaperItem]) -> TopicStructuringResult:
        """
        Executes the full topic mining pipeline: Embedding -> Clustering -> Labeling.

        Args:
            papers: A list of PaperItem objects retrieved from the search stage.

        Returns:
            TopicStructuringResult: Structured clusters with generated topic names.
        """
        if not papers:
            logger.warning("No papers provided for topic mining. Returning empty result.")
            return TopicStructuringResult(clusters=[], main_directions=[])

        logger.info(f"Starting topic mining for {len(papers)} papers.")

        # Step 1: Vectorization
        valid_papers, embeddings = self._generate_embeddings(papers)
        if not valid_papers:
            logger.error("Failed to generate embeddings for any papers.")
            raise ValueError("No valid embeddings generated.")

        # Step 2: Clustering
        # Dynamically determine cluster count (e.g., sqrt(N) or capped logic)
        n_clusters = self._determine_optimal_clusters(len(valid_papers))
        labels = self._perform_clustering(embeddings, n_clusters)

        # Step 3: Synthesis (Labeling)
        clusters = self._synthesize_cluster_labels(valid_papers, labels, n_clusters)

        logger.info(f"Topic mining completed. Generated {len(clusters)} clusters.")
        
        return TopicStructuringResult(
            clusters=clusters,
            main_directions=[c.topic_name for c in clusters]
        )

    def _generate_embeddings(self, papers: List[PaperItem]) -> tuple[List[PaperItem], np.ndarray]:
        """Fetches embeddings for paper abstracts, filtering out failures."""
        embeddings = []
        valid_papers = []

        for paper in papers:
            content = paper.abstract if paper.abstract else paper.title
            try:
                # Replace this line with your actual API call method
                # vector = self.llm_client.get_embedding(text=content, model=self.embedding_model)
                
                # MOCKING the API call for demonstration purposes
                # In production, this must be the real API call
                vector = self.llm_client.get_embedding(content) 
                
                embeddings.append(vector)
                valid_papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to embed paper {paper.paper_id}: {str(e)}")

        return valid_papers, np.array(embeddings)

    def _determine_optimal_clusters(self, num_papers: int) -> int:
        """Heuristic to determine optimal K for K-Means."""
        if num_papers < self.MIN_CLUSTERS:
            return 1
        # A simple heuristic: sqrt(N/2), capped between MIN and MAX
        heuristic = int((num_papers / 2) ** 0.5)
        return max(self.MIN_CLUSTERS, min(heuristic, self.MAX_CLUSTERS))

    def _perform_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Runs K-Means clustering on the embedding matrix."""
        logger.debug(f"Running K-Means with n_clusters={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        return kmeans.labels_

    def _synthesize_cluster_labels(
        self, papers: List[PaperItem], labels: np.ndarray, n_clusters: int
    ) -> List[TopicCluster]:
        """Generates semantic labels for each cluster using the LLM."""
        final_clusters = []

        # Group papers by label
        cluster_map = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            cluster_map[label].append(papers[idx])

        for label_id, cluster_papers in cluster_map.items():
            if not cluster_papers:
                continue

            # Context window optimization: Only use top 3 papers to define the topic
            sample_docs = [
                f"Title: {p.title}\nAbstract: {p.abstract[:300]}..." 
                for p in cluster_papers[:3]
            ]
            context_text = "\n\n".join(sample_docs)

            prompt = (
                f"Analyze the following research paper abstracts from a cluster:\n"
                f"{context_text}\n\n"
                f"Task: Provide a concise, professional research sub-field name (e.g., 'Graph Neural Networks', 'Few-Shot Learning') "
                f"that best represents this cluster. Output ONLY the name."
            )

            try:
                topic_name = self.llm_client.generate_text(prompt).strip()
            except Exception as e:
                logger.error(f"LLM failed to label cluster {label_id}: {e}")
                topic_name = f"Cluster {label_id}"

            final_clusters.append(TopicCluster(
                topic_name=topic_name,
                paper_ids=[p.paper_id for p in cluster_papers],
                description=f"A cluster containing {len(cluster_papers)} papers focused on {topic_name}."
            ))

        return final_clusters