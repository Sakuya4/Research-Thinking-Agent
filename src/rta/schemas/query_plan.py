from pydantic import BaseModel
from typing import List


class QueryPlan(BaseModel):
    expanded_queries: List[str]
    must_include: List[str]
    exclude: List[str]
    target_subtasks: List[str]
    notes: str
