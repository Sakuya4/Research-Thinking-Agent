from __future__ import annotations

import random
import json
import time
import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET

from ..config import RTAConfig
from ..logger import EventLogger
from ..schemas import QueryPlan, RetrievalResult, PaperItem

import os
sources = os.getenv("RTA_SOURCES", "both").lower()
use_s2 = sources in ("both", "s2")
use_arxiv = sources in ("both", "arxiv")

# -------------------------
# Helpers
# -------------------------

def _norm_title(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s\-:]", "", t)
    return t


def _title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20) -> Dict[str, Any]:
    req = Request(url, headers=headers or {"User-Agent": "RTA/0.1 (paper retrieval)"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _http_get_text(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20) -> str:
    req = Request(url, headers=headers or {"User-Agent": "RTA/0.1 (paper retrieval)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _cache_dir(cfg: RTAConfig) -> Path:
    return Path(cfg.runs_dir) / "_cache"


def _cache_key(prefix: str, payload: str) -> str:
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{h}.json"


def _cache_get(cfg: RTAConfig, key: str) -> Optional[Dict[str, Any]]:
    p = _cache_dir(cfg) / key
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("_ts", 0)
        ttl = cfg.cache_ttl_hours * 3600
        if (time.time() - ts) > ttl:
            return None
        return data.get("payload")
    except Exception:
        return None


def _cache_put(cfg: RTAConfig, key: str, payload: Dict[str, Any]) -> None:
    d = _cache_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    obj = {"_ts": time.time(), "payload": payload}
    (d / key).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def _http_get_json_with_retry(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20, retries: int = 3) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return _http_get_json(url, headers=headers, timeout=timeout)
        except HTTPError as e:
            last_err = e
            # 429: rate limit -> exponential backoff
            if e.code == 429:
                sleep_s = (2 ** (attempt - 1)) + random.random()
                time.sleep(sleep_s)
                continue
            raise
        except URLError as e:
            last_err = e
            sleep_s = (2 ** (attempt - 1)) + random.random()
            time.sleep(sleep_s)
            continue
    raise last_err  # type: ignore

# -------------------------
# Semantic Scholar
# -------------------------

def _search_semantic_scholar(cfg: RTAConfig, query: str, limit: int = 20) -> List[PaperItem]:
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": str(limit),
        "fields": "title,authors,year,abstract,url,venue,citationCount"
    }
    url = f"{base}?{urlencode(params, quote_via=quote_plus)}"
    key = _cache_key("s2", url)

    cached = _cache_get(cfg, key)
    if cached is None:
        data = _http_get_json_with_retry(url)
        _cache_put(cfg, key, data)
    else:
        data = cached

    out: List[PaperItem] = []
    for it in data.get("data", []) or []:
        title = (it.get("title") or "").strip()
        if not title:
            continue
        authors = []
        for a in it.get("authors", []) or []:
            name = (a.get("name") or "").strip()
            if name:
                authors.append(name)

        out.append(
            PaperItem(
                title=title,
                authors=authors,
                year=_safe_int(it.get("year")),
                abstract=(it.get("abstract") or "").strip(),
                url=(it.get("url") or "").strip(),
                source="SemanticScholar",
            )
        )
    return out


# -------------------------
# arXiv
# -------------------------

def _search_arxiv(cfg: RTAConfig, query: str, max_results: int = 20) -> List[PaperItem]:
    # arXiv Atom API
    # search_query=all:<terms>
    q = query.replace('"', "")
    api = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{q}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{api}?{urlencode(params, quote_via=quote_plus)}"
    key = _cache_key("arxiv", url)

    cached = _cache_get(cfg, key)
    if cached is None:
        xml_text = _http_get_text(url)
        _cache_put(cfg, key, {"xml": xml_text})
    else:
        xml_text = (cached.get("xml") or "")

    if not xml_text.strip():
        return []

    # Parse Atom XML
    root = ET.fromstring(xml_text)
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    out: List[PaperItem] = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        if not title:
            continue

        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        year = None
        if len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])

        authors = []
        for a in entry.findall("a:author", ns):
            name = (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)

        url_link = ""
        for link in entry.findall("a:link", ns):
            href = link.attrib.get("href", "")
            rel = link.attrib.get("rel", "")
            if rel == "alternate" and href:
                url_link = href
                break

        out.append(
            PaperItem(
                title=title,
                authors=authors,
                year=year,
                abstract=summary,
                url=url_link,
                source="arXiv",
            )
        )
    return out


# -------------------------
# Public API for pipeline
# -------------------------

def retrieval_live(cfg: RTAConfig, qp: QueryPlan, logger: EventLogger) -> RetrievalResult:
    """
    Retrieve papers from Semantic Scholar + arXiv.
    - Deduplicate by title similarity >= 0.90
    - Filter by year range
    - Prefer papers with abstracts
    - Truncate to cfg.max_papers
    """
    queries = [q.strip() for q in qp.expanded_queries if q.strip()]
    warnings: List[str] = []

    s2_disabled = False
    s2_429_count = 0

    all_papers: List[PaperItem] = []

    per_query_limit = max(10, min(25, cfg.max_papers // max(1, len(queries))))
    per_query_limit = min(per_query_limit, 20)

    for i, q in enumerate(queries):
        logger.log("stage2", "query", {"i": i, "q": q, "per_query_limit": per_query_limit})

        # Semantic Scholar
        if not s2_disabled:
            try:
                s2 = _search_semantic_scholar(cfg, q, limit=per_query_limit)
                logger.log("stage2", "source", {"q": q, "source": "SemanticScholar", "count": len(s2)})
                all_papers.extend(s2)
            except HTTPError as e:
                if getattr(e, "code", None) == 429:
                    s2_429_count += 1
                    warnings.append(f"SemanticScholar rate-limited (429) for query='{q}'")
                    logger.log(
                        "stage2",
                        "warn",
                        {"source": "SemanticScholar", "q": q, "error": f"429 rate limit (count={s2_429_count})"},
                    )
                    if s2_429_count >= 3:
                        s2_disabled = True
                        warnings.append("SemanticScholar disabled for this run due to repeated 429.")
                        logger.log("stage2", "warn", {"source": "SemanticScholar", "msg": "disabled due to repeated 429"})
                else:
                    warnings.append(f"SemanticScholar failed for query='{q}': {e}")
                    logger.log("stage2", "warn", {"source": "SemanticScholar", "q": q, "error": str(e)})
            except URLError as e:
                warnings.append(f"SemanticScholar failed for query='{q}': {e}")
                logger.log("stage2", "warn", {"source": "SemanticScholar", "q": q, "error": str(e)})

        # arXiv
        try:
            ax = _search_arxiv(cfg, q, max_results=per_query_limit)
            logger.log("stage2", "source", {"q": q, "source": "arXiv", "count": len(ax)})
            all_papers.extend(ax)
        except (HTTPError, URLError, ET.ParseError) as e:
            msg = f"arXiv failed for query='{q}': {e}"
            warnings.append(msg)
            logger.log("stage2", "warn", {"source": "arXiv", "q": q, "error": str(e)})

        # stop early if we already have plenty (before dedup)
        if len(all_papers) >= cfg.max_papers * 4:
            break

    dedup_before = len(all_papers)

    # Dedup by title similarity
    uniq: List[PaperItem] = []
    for p in all_papers:
        title = p.title.strip()
        if not title:
            continue

        if any(_title_sim(title, u.title) >= 0.90 for u in uniq):
            continue
        uniq.append(p)

        if len(uniq) >= cfg.max_papers * 2:
            break

    # Filter by year range
    filtered: List[PaperItem] = []
    for p in uniq:
        y = p.year
        if y is not None and (y < cfg.min_year or y > cfg.max_year):
            continue
        filtered.append(p)

    # prefer those with abstracts first
    filtered.sort(key=lambda x: 0 if (x.abstract and x.abstract.strip()) else 1)

    # truncate to max_papers
    filtered = filtered[: cfg.max_papers]

    missing_abs = sum(1 for p in filtered if not (p.abstract and p.abstract.strip()))
    if filtered and missing_abs / len(filtered) > 0.4:
        warnings.append(f"High missing-abstract ratio in final set: {missing_abs}/{len(filtered)}")

    return RetrievalResult(
        queries_used=queries,
        papers=filtered,
        dedup_before=dedup_before,
        dedup_after=len(filtered),
        warnings=warnings,
    )
