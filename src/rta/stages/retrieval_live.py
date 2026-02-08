"""
Stage 2: Literature Retrieval Module (Production Grade - Live Data).
Fetches real papers from the arXiv API with strict count control and text cleaning.
File: src/rta/stages/retrieval_live.py
"""

import logging
import time
import re
import arxiv  # Ensure you ran: pip install arxiv
from typing import List, Any
from pydantic import BaseModel

class PaperItem(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: List[str] = []
    year: int = 2024
    url: str = ""
    source: str = "arxiv"

class RetrievalResult(BaseModel):
    papers: List[PaperItem]
    total_found: int = 0
    queries_used: List[str] = []

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Sanitizes and compresses text to optimize token usage.
    Removes line breaks, LaTeX fragments, and redundant whitespace.
    """
    if not text:
        return ""
    # Replace newlines and tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ")
    # Remove redundant multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove common LaTeX junk often found in arXiv abstracts (e.g., $, \, {})
    text = re.sub(r'\$[^$]*\$', '', text) 
    return text.strip()

def run_retrieval(queries: List[str], max_per_query: int = 1) -> RetrievalResult:
    """
    Retrieves authentic papers from arXiv. 
    Applies text cleaning to reduce payload for the Reasoning Stage.
    """
    logger.info(f"[Retrieval] Searching arXiv for {len(queries)} queries...")
    all_papers = []
    seen_ids = set()
    
    # Process queries (Limited to top 8 to control total token volume)
    for query in queries[:8]: 
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_per_query,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in search.results():
                p_id = result.get_short_id()
                if p_id not in seen_ids:
                    # Apply cleaning to both Title and Abstract
                    cleaned_title = clean_text(result.title)
                    cleaned_abstract = clean_text(result.summary)
                    
                    all_papers.append(PaperItem(
                        paper_id=p_id,
                        title=cleaned_title,
                        abstract=cleaned_abstract,
                        authors=[a.name for a in result.authors],
                        year=result.published.year,
                        url=result.pdf_url,
                        source="arxiv"
                    ))
                    seen_ids.add(p_id)
            
            # Rate limiting compliance
            time.sleep(0.5) 
        except Exception as e:
            logger.error(f"[Retrieval] Query '{query}' failed: {e}")

    logger.info(f"[Retrieval] Total authentic papers collected: {len(all_papers)}")
    return RetrievalResult(
        papers=all_papers, 
        total_found=len(all_papers), 
        queries_used=queries
    )