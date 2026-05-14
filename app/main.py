"""
SHL Assessment Recommender - FastAPI Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

from app.models import ChatRequest, ChatResponse
from app.chat_logic import run_chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment recommendations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        start = time.time()
        response = await run_chat(request.messages)
        elapsed = time.time() - start
        logger.info(f"Chat handled in {elapsed:.2f}s | eoc={response.end_of_conversation} | recs={len(response.recommendations)}")
        return response
    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e))