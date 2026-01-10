from pydantic import BaseModel
from typing import List
from .retrieval import PaperItem
from .topic_structuring import TopicCluster


class FinalOutput(BaseModel):
    query: str
    main_directions: List[str]
    recommended_pipeline: List[str]
    clusters: List[TopicCluster]
    top_papers: List[PaperItem]
