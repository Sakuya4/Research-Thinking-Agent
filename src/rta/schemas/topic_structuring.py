from pydantic import BaseModel
from typing import List


class TopicCluster(BaseModel):
    cluster_id: str
    name: str
    description: str
    keywords: List[str]
    typical_methods: List[str]


class TopicStructuringResult(BaseModel):
    main_directions: List[str]
    recommended_pipeline: List[str]
    clusters: List[TopicCluster]
