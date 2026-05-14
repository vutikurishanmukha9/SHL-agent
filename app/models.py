"""
Pydantic models — strict schema matching the evaluator spec.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str     # pipe-separated short codes, e.g. "A|K"


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False