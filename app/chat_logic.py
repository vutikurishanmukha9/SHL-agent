"""
Chat logic — the brain of the SHL Assessment Recommender.

Architecture (per request):
1. Guardrail check  — deterministic, no LLM, fast
2. Intent detection — deterministic heuristics (regex + context signals)
3. Retrieval        — BM25 + metadata re-rank over catalog
4. LLM call         — Anthropic API with catalog context injected in system prompt
5. Parse & validate — extract JSON block, validate URLs against catalog, return

Intent types
────────────
  VAGUE     — not enough info to retrieve; ask one clarifying question
  RECOMMEND — sufficient context; retrieve + suggest
  REFINE    — update an existing shortlist (prior recs exist in history)
  COMPARE   — side-by-side comparison of named assessments
"""

import re
import json
import os
import logging
import httpx
from typing import List, Dict, Optional, Tuple

from app.models import Message, ChatResponse, Recommendation
from app.retriever import search, get_by_names, extract_job_level, extract_test_types, CATALOG
from app.prompts import SYSTEM_PROMPT, build_catalog_context, build_comparison_context
from app.guardrails import check as guardrail_check, INJECTION_REPLY, OFF_TOPIC_REPLY

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

# Pre-build a lowercase name→item lookup for fast validation
_CATALOG_LOOKUP: Dict[str, Dict] = {item["name"].lower(): item for item in CATALOG}

# ── Intent detection ──────────────────────────────────────────────────────────

_COMPARE_RE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(compare|comparison|difference|vs\.?|versus|between)\b",
        r"\bwhat.{0,30}differ",
        r"\bwhich is better\b",
        r"\bhow does .{1,40} (differ|compare)",
    ]
]

_REFINE_RE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(actually|instead|also add|remove|drop|exclude|include|change|update|modify|without)\b",
        r"\badd\b",
        r"\bshorter\b",
        r"\blonger\b",
        r"\bmore (personality|cognitive|technical|skills|ability|aptitude)\b",
        r"\bfewer\b",
        r"\bonly\b.{0,20}\b(cognitive|personality|technical|skills)\b",
    ]
]

_VAGUE_RE = re.compile(
    r"^(i need (an?|the)? (assessment|test|evaluation)|"
    r"what (assessment|test)|"
    r"help me (find|choose|pick|select)|"
    r"give me (an?|the)? (assessment|test)|"
    r"suggest (an?|the)? (assessment|test)|"
    r"recommend (an?|the)? (assessment|test))\.?$",
    re.IGNORECASE,
)

_ROLE_RE = re.compile(
    r"\b(developer|engineer|manager|analyst|designer|writer|sales|nurse|"
    r"admin|assistant|director|executive|lead|officer|specialist|consultant|"
    r"java|python|javascript|typescript|react|angular|node|"
    r"data|software|hardware|marketing|finance|hr|operations|"
    r"hadoop|devops|cloud|frontend|backend|fullstack|ml|ai|accounting|"
    r"customer service|support|recruiter|scientist|researcher|"
    r"pharmacist|teacher|instructor|trainer|architect|product)\b",
    re.IGNORECASE,
)


def _full_text(messages: List[Message]) -> str:
    return " ".join(m.content for m in messages)


def _latest_user(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _has_prior_recommendations(messages: List[Message]) -> bool:
    """True if assistant has already output a recommendations JSON block."""
    for m in messages:
        if m.role == "assistant" and '"recommendations"' in m.content:
            return True
    return False


def _detect_intent(messages: List[Message]) -> str:
    latest = _latest_user(messages)
    full = _full_text(messages)

    # 1. Comparison request?
    if any(p.search(latest) for p in _COMPARE_RE):
        return "COMPARE"

    # 2. Refinement (only if we already gave recommendations)
    if _has_prior_recommendations(messages) and any(
        p.search(latest) for p in _REFINE_RE
    ):
        return "REFINE"

    # 3. Vague — single short message matching vague patterns
    if len(messages) == 1 and _VAGUE_RE.match(latest.strip()):
        return "VAGUE"

    # 4. No role signal and early in conversation → ask
    has_role = bool(_ROLE_RE.search(full))
    if not has_role and len(messages) <= 2:
        return "VAGUE"

    return "RECOMMEND"


# ── Comparison target extractor ───────────────────────────────────────────────

def _extract_comparison_targets(text: str) -> List[str]:
    """Try to extract two named assessments from a comparison request."""
    patterns = [
        r"compare\s+(.+?)\s+and\s+(.+?)(?:\?|$|\n)",
        r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$|\n)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$|\n)",
        r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$|\n)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
    return []


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _call_llm(system: str, messages: List[Dict]) -> str:
    """Call Anthropic claude-sonnet-4 and return raw text response."""
    async with httpx.AsyncClient(timeout=28.0) as client:
        resp = await client.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "system": system,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(
            block["text"]
            for block in data.get("content", [])
            if block.get("type") == "text"
        )


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_llm_response(
    raw: str, catalog_items: List[Dict]
) -> Tuple[str, List[Recommendation], bool]:
    """
    Split raw LLM output into:
      - reply_text  (the human-readable part)
      - recs        (validated Recommendation objects)
      - eoc         (end_of_conversation flag)

    Validation:
      - If the name matches a catalog item exactly → use catalog URL
      - If name doesn't match but URL is shl.com → accept as-is
      - Otherwise drop (hallucinated)
    """
    # Build a local lookup from the retrieval results (for priority)
    local_lookup = {item["name"].lower(): item for item in catalog_items}

    # Try fenced JSON block first, then bare JSON
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{"recommendations"\s*:.*?\})', raw, re.DOTALL)

    reply_text = raw
    recs: List[Recommendation] = []
    eoc = False

    if json_match:
        reply_text = raw[: json_match.start()].strip()
        try:
            parsed = json.loads(json_match.group(1))
            eoc = bool(parsed.get("end_of_conversation", False))

            for rec in parsed.get("recommendations", []):
                name = rec.get("name", "").strip()
                url = rec.get("url", "").strip()
                test_type = rec.get("test_type", "").strip()

                name_lower = name.lower()

                # Priority 1: exact name match in local retrieval results
                matched = local_lookup.get(name_lower)

                # Priority 2: full catalog lookup
                if not matched:
                    matched = _CATALOG_LOOKUP.get(name_lower)

                # Priority 3: partial name match in full catalog
                if not matched:
                    for cat_name, cat_item in _CATALOG_LOOKUP.items():
                        if name_lower in cat_name or cat_name in name_lower:
                            matched = cat_item
                            break

                if matched:
                    recs.append(
                        Recommendation(
                            name=matched["name"],
                            url=matched["url"],
                            test_type="|".join(matched.get("test_types", []))
                            or test_type,
                        )
                    )
                elif url.startswith("https://www.shl.com/"):
                    # URL looks legitimate even if name didn't exactly match
                    recs.append(
                        Recommendation(name=name, url=url, test_type=test_type)
                    )
                else:
                    logger.warning(f"Dropped hallucinated recommendation: {name!r}")

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(f"JSON parse error in LLM response: {exc}")

    # If JSON block was absent, return full raw text as reply with no recs
    return reply_text or raw, recs, eoc


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_chat(messages: List[Message]) -> ChatResponse:
    latest = _latest_user(messages)

    # 1. Guardrail check — fast, deterministic
    blocked, reason = guardrail_check(latest)
    if blocked:
        reply = INJECTION_REPLY if reason == "injection" else OFF_TOPIC_REPLY
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # 2. Detect intent
    intent = _detect_intent(messages)
    logger.info(f"Intent={intent} | latest={latest[:80]!r}")

    # 3. Build retrieval context
    full_text = _full_text(messages)
    job_level = extract_job_level(full_text)
    test_types = extract_test_types(full_text)
    catalog_items: List[Dict] = []
    catalog_context = ""

    if intent == "COMPARE":
        targets = _extract_comparison_targets(latest) or _extract_comparison_targets(full_text)
        if targets:
            catalog_items = get_by_names(targets)
        if not catalog_items:
            # fall back to semantic search so LLM has something to compare
            catalog_items = search(latest, job_level=job_level, test_types=test_types, top_k=10)
        catalog_context = build_comparison_context(catalog_items)

    elif intent == "VAGUE":
        # No catalog context yet — let LLM ask one clarifying question
        catalog_context = ""

    else:  # RECOMMEND or REFINE
        catalog_items = search(
            full_text, job_level=job_level, test_types=test_types, top_k=15
        )
        catalog_context = build_catalog_context(catalog_items)

    # 4. Build system prompt (base + optional catalog context)
    system = SYSTEM_PROMPT
    if catalog_context:
        system = SYSTEM_PROMPT + "\n\n" + catalog_context

    # 5. Convert to Anthropic message format
    anthropic_messages = [{"role": m.role, "content": m.content} for m in messages]

    # 6. LLM call
    raw = await _call_llm(system, anthropic_messages)
    logger.debug(f"LLM raw (first 300): {raw[:300]}")

    # 7. Parse response
    reply, recs, eoc = _parse_llm_response(raw, catalog_items)

    # 8. Safety overrides
    if intent == "VAGUE":
        # Never emit recommendations on a vague turn, regardless of LLM output
        recs = []
        eoc = False

    # Cap to 10 as per spec
    recs = recs[:10]

    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=eoc)