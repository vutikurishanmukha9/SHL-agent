"""
Retriever — BM25-based retrieval over the SHL catalog with metadata re-ranking.

Design choice: BM25 over FAISS/sentence-transformers because:
  - No GPU / large model download needed at cold start
  - 30s timeout constraint → BM25 is instant (< 5ms at 389 docs)
  - 389 documents is tiny; BM25 works extremely well at this scale
  - Avoids cold-start penalty on free-tier hosting (Render, Railway, Fly)

Re-ranking applies metadata boosts on top of BM25:
  - Job level match  → +2.0
  - Test type match  → +1.0 per type
  - Missing job_levels penalty → -0.5
"""

import json
import re
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Load catalog ─────────────────────────────────────────────────────────────

_CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"


def _load_catalog() -> List[Dict]:
    path = _CATALOG_PATH
    if not path.exists():
        # fallback: look next to this file
        path = Path(__file__).parent / "shl_catalog.json"
    with open(path) as f:
        raw = json.load(f)

    cleaned = []
    for item in raw:
        desc = item.get("description", "") or ""
        # strip boilerplate sales noise so it doesn't pollute BM25
        desc = re.sub(
            r"(Speak to our team today.*?|There are multiple configurations.*?|Multiple configurations.*?|Contact SHL to find out more.*)",
            "",
            desc,
            flags=re.IGNORECASE,
        ).strip()

        # build a rich search text blob
        search_text = " ".join(
            filter(
                None,
                [
                    item.get("name", ""),
                    desc,
                    item.get("measures", "") or "",
                    " ".join(item.get("test_type_labels", [])),
                    " ".join(item.get("job_levels", [])),
                    " ".join(item.get("languages", [])),
                ],
            )
        ).lower()

        cleaned.append(
            {
                **item,
                "description": desc,
                "_search_text": search_text,
            }
        )

    return cleaned


CATALOG: List[Dict] = _load_catalog()
logger.info(f"Catalog loaded: {len(CATALOG)} assessments")

# ── Tokenizer ─────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "is", "are",
    "that", "this", "with", "as", "on", "be", "by", "from", "at", "it",
    "its", "we", "our", "your", "their", "which", "have", "has", "been",
    "will", "would", "can", "also", "use", "used", "uses", "multiple",
    "configurations", "solutions", "solution", "available", "today", "see",
    "how", "products", "transform", "talent", "strategy", "team", "speak",
    "not", "limited", "potential", "job", "titles", "include", "but",
    "candidates", "sample", "tasks", "these", "positions",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ── BM25 ──────────────────────────────────────────────────────────────────────


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenized = [_tokenize(doc) for doc in corpus]
        self.N = len(self.tokenized)
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.N, 1)
        self._build_idf()

    def _build_idf(self) -> None:
        df: Dict[str, int] = {}
        for doc in self.tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self.tokenized[doc_idx]
        dl = len(doc)
        tf_map: Dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for qt in query_tokens:
            if qt not in self.idf:
                continue
            tf = tf_map.get(qt, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[qt] * numerator / denominator
        return score

    def get_top_k(self, query: str, k: int = 40) -> List[tuple]:
        """Returns [(idx, score), ...] sorted descending, score > 0 only."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = [(i, self.score(tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:k] if s > 0]


_bm25 = BM25([item["_search_text"] for item in CATALOG])

# ── Job level normalizer ──────────────────────────────────────────────────────

_LEVEL_MAP = {
    "entry": "Entry-Level",
    "junior": "Entry-Level",
    "fresher": "Entry-Level",
    "graduate": "Graduate",
    "grad": "Graduate",
    "intern": "Entry-Level",
    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "midlevel": "Mid-Professional",
    "senior": "Mid-Professional",
    "professional": "Professional Individual Contributor",
    "individual contributor": "Professional Individual Contributor",
    "manager": "Manager",
    "management": "Manager",
    "lead": "Manager",
    "team lead": "Manager",
    "front line": "Front Line Manager",
    "frontline": "Front Line Manager",
    "supervisor": "Manager",
    "director": "Director",
    "vp": "Executive",
    "vice president": "Executive",
    "c-suite": "Executive",
    "executive": "Executive",
    "ceo": "Executive",
    "cto": "Executive",
    "cfo": "Executive",
}


def extract_job_level(text: str) -> Optional[str]:
    lower = text.lower()
    # longest match first to avoid "mid" stealing "mid-level"
    for kw in sorted(_LEVEL_MAP, key=len, reverse=True):
        if kw in lower:
            return _LEVEL_MAP[kw]
    # year-of-experience heuristics
    match = re.search(r"(\d+)\s*(?:\+\s*)?(?:years?|yrs?)", lower)
    if match:
        yrs = int(match.group(1))
        if yrs <= 2:
            return "Entry-Level"
        elif yrs <= 5:
            return "Mid-Professional"
        elif yrs <= 10:
            return "Manager"
        else:
            return "Executive"
    return None


# ── Test type extractor ───────────────────────────────────────────────────────

_TEST_TYPE_MAP = {
    "personality": "P",
    "behaviour": "P",
    "behavioral": "P",
    "behavior": "P",
    "cognitive": "A",
    "ability": "A",
    "aptitude": "A",
    "numerical": "A",
    "verbal": "A",
    "reasoning": "A",
    "inductive": "A",
    "deductive": "A",
    "abstract": "A",
    "knowledge": "K",
    "skills": "K",
    "technical": "K",
    "situational": "S",
    "sjt": "S",
    "judgement": "S",
    "judgment": "S",
    "biodata": "B",
    "biographical": "B",
    "competency": "C",
    "competencies": "C",
    "360": "D",
    "development": "D",
    "feedback": "D",
    "simulation": "E",
    "exercise": "E",
}


def extract_test_types(text: str) -> List[str]:
    lower = text.lower()
    found: List[str] = []
    for kw, code in _TEST_TYPE_MAP.items():
        if kw in lower and code not in found:
            found.append(code)
    return found


# ── Metadata re-ranker ────────────────────────────────────────────────────────


def _rerank_score(
    item: Dict, job_level: Optional[str], test_types: List[str]
) -> float:
    boost = 0.0
    if job_level and job_level in item.get("job_levels", []):
        boost += 2.0
    for tt in test_types:
        if tt in item.get("test_types", []):
            boost += 1.0
    # slight penalty for items with no job-level metadata
    if not item.get("job_levels"):
        boost -= 0.5
    return boost


# ── Public search API ─────────────────────────────────────────────────────────


def search(
    query: str,
    job_level: Optional[str] = None,
    test_types: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict]:
    """
    Returns up to top_k catalog items most relevant to the query.
    Combines BM25 + metadata re-ranking.
    Returns [] if query has no meaningful tokens.
    """
    if test_types is None:
        test_types = []

    candidates = _bm25.get_top_k(query, k=min(60, len(CATALOG)))

    results = []
    for idx, bm25_score in candidates:
        item = CATALOG[idx]
        meta_boost = _rerank_score(item, job_level, test_types)
        final_score = bm25_score + meta_boost
        results.append((final_score, item))

    results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in results[:top_k]]


def get_by_names(names: List[str]) -> List[Dict]:
    """Fetch specific catalog items by name substring match (for comparison)."""
    name_lower = [n.lower().strip() for n in names]
    matched = []
    for item in CATALOG:
        item_name = item["name"].lower()
        if any(nl in item_name or item_name in nl for nl in name_lower):
            matched.append(item)
    return matched