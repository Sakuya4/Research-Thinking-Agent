from pydantic import BaseModel
from typing import List, Optional


class PaperItem(BaseModel):
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: str
    url: str
    source: str


class RetrievalResult(BaseModel):
    queries_used: List[str]
    papers: List[PaperItem]
    dedup_before: int
    dedup_after: int
    warnings: List[str]
