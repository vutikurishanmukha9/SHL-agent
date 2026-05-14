"""
Guardrails — keeps the agent on-topic and blocks prompt injection.
All checks are deterministic (no LLM call needed).
"""

import re

# Keywords that signal off-topic requests
_OFF_TOPIC_PATTERNS = [
    r"\b(salary|compensation|pay|wages?)\b",
    r"\b(legal|law|lawsuit|compliance|gdpr|hipaa)\b",
    r"\b(how to hire|hiring process|interview tips|offer letter)\b",
    r"\b(diversity|dei|equal opportunity)\b",
    r"\b(background check|drug test|reference check)\b",
    r"\bweather\b",
    r"\b(recipe|cook|food)\b",
    r"\b(stock|invest|crypto)\b",
]

# Prompt injection signatures
_INJECTION_PATTERNS = [
    r"ignore (previous|all|prior|above) instructions?",
    r"you are now",
    r"forget (everything|all|your instructions?)",
    r"act as (an?|a different|a new)",
    r"new (system|persona|role|instructions?)",
    r"override (your )?(instructions?|rules?|guidelines?)",
    r"disregard (your )?(instructions?|rules?|guidelines?)",
    r"jailbreak",
    r"do anything now",
    r"dan mode",
    r"pretend (you are|to be)",
    r"roleplay as",
    r"simulate (being|a)",
]

_COMPILED_OFF_TOPIC = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def is_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED_INJECTION)


def is_off_topic(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED_OFF_TOPIC)


def check(text: str) -> tuple:
    """
    Returns (is_blocked, reason).
    Call with the latest user message before processing.
    """
    if not text or not text.strip():
        return False, ""
    if is_injection(text):
        return True, "injection"
    if is_off_topic(text):
        return True, "off_topic"
    return False, ""


INJECTION_REPLY = (
    "I'm here to help with SHL assessment selection only. "
    "I can't follow instructions that ask me to change my role or ignore my guidelines."
)

OFF_TOPIC_REPLY = (
    "I specialise in recommending SHL assessments. "
    "I'm not able to help with that topic, but I'd be happy to help you find the right assessment "
    "for your hiring needs. What role are you hiring for?"
)