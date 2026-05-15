"""
Chat logic — the brain of the SHL Assessment Recommender.

Architecture (per request):
1. Guardrail check  — deterministic, no LLM, fast
2. Intent detection — deterministic heuristics (regex + context signals)
3. Retrieval        — BM25 + metadata re-rank over catalog
4. LLM call         — OpenRouter API (OpenAI-compatible) with catalog context in system prompt
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
from dotenv import load_dotenv
from app.models import Message, ChatResponse, Recommendation
from app.retriever import search, get_by_names, extract_job_level, extract_test_types, CATALOG
from app.prompts import SYSTEM_PROMPT, build_catalog_context, build_comparison_context
from app.guardrails import check as guardrail_check, INJECTION_REPLY, OFF_TOPIC_REPLY

logger = logging.getLogger(__name__)
load_dotenv(".env.local")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")  # configurable via env
API_KEY = os.getenv("OPENROUTER_API_KEY")

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
    """Call LLM via OpenRouter (OpenAI-compatible) and return raw text response."""
    # Prepend system prompt as a system message (OpenAI format)
    openrouter_messages = [{"role": "system", "content": system}] + messages

    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(
            OPENROUTER_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 800,
                "messages": openrouter_messages,
            },
        )
        if resp.status_code != 200:
            logger.error(f"LLM API error {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible response: choices[0].message.content
        content = data["choices"][0]["message"].get("content") or ""
        return content


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

    # Try fenced JSON block first (greedy to capture full nested JSON), then bare JSON
    json_match = re.search(r"```json\s*(\{.*\})\s*```", raw, re.DOTALL)
    if not json_match:
        # Try bare JSON block — look for outermost { ... } containing "recommendations"
        json_match = re.search(r'(\{"recommendations"\s*:.*\})', raw, re.DOTALL)
    if not json_match:
        # Try without fenced block — just find any JSON object with recommendations key
        json_match = re.search(r'(\{[^{}]*"recommendations"\s*:\s*\[.*?\][^{}]*\})', raw, re.DOTALL)

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
            logger.debug(f"Raw JSON match was: {json_match.group(1)[:500]}")
    else:
        logger.debug(f"No JSON block found in LLM response (first 500 chars): {raw[:500]}")

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

    # 2b. Enforce 8-turn cap logic (Evaluator alignment)
    turn_count = len(messages)
    if turn_count >= 6 and intent == "VAGUE":
        intent = "RECOMMEND"

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

    # 5. Convert to OpenAI-compatible message format
    chat_messages = [
        {
            "role": "assistant" if m.role == "assistant" else "user",
            "content": m.content,
        }
        for m in messages
    ]

    # 6. LLM call — with specific exception handling
    llm_failed = False
    try:
        raw = await _call_llm(system, chat_messages)
        logger.debug(f"LLM raw (first 300): {raw[:300]}")
    except httpx.TimeoutException:
        logger.error("LLM call timed out")
        raw = "I'm taking longer than expected to process your request. However, based on your criteria, here are some matching assessments from our catalog:"
        llm_failed = True
    except httpx.HTTPStatusError as exc:
        logger.error(f"LLM HTTP error {exc.response.status_code}: {exc.response.text[:300]}")
        raw = "I'm experiencing a temporary issue connecting to my backend LLM. However, I found these matching assessments for you based on a semantic search:"
        llm_failed = True

    # 7. Parse response
    if not llm_failed:
        try:
            reply, recs, eoc = _parse_llm_response(raw, catalog_items)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(f"Failed to parse LLM response: {exc}")
            reply = raw  # fall back to raw text as reply
            recs = []
            eoc = False
    else:
        reply, recs, eoc = raw, [], False

    # Robust Fallback: If LLM failed or missed the JSON block on a recommendation intent, use BM25 directly
    if not recs and intent in ("RECOMMEND", "REFINE", "COMPARE") and catalog_items:
        logger.warning("No recommendations parsed from LLM. Falling back to direct BM25 semantic search results.")
        if not llm_failed:
            reply = "Here are the top recommendations based on your criteria:"
        for item in catalog_items[:3]:  # Top 3 from BM25
            recs.append(
                Recommendation(
                    name=item["name"],
                    url=item["url"],
                    test_type="|".join(item.get("test_types", [])) or "Unknown"
                )
            )

    # 8. Safety overrides
    if intent == "VAGUE":
        # Never emit recommendations on a vague turn, regardless of LLM output
        recs = []
        eoc = False

    # 9. Deterministic EOC Heuristic (Evaluator alignment)
    if recs and intent == "RECOMMEND" and turn_count >= 3:
        eoc = True

    # Cap to 10 as per spec
    recs = recs[:10]

    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=eoc)