"""OpenAI-compatible endpoints.

Standard chat clients (Open WebUI, LobeChat, custom widgets, ...) expect
`GET /v1/models` and `POST /v1/chat/completions`. These routes expose the
same grounded RAG agent through that protocol so no client-side changes are
needed beyond pointing it at this backend.
"""

import logging
import time
import uuid

from fastapi import APIRouter

from app.api.errors import run_chat
from app.core.config import get_settings
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai-compatible"])


@router.get("/models")
def list_models():
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.llm_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": settings.llm_provider,
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(payload: ChatRequest):
    result = run_chat(payload)

    settings = get_settings()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.llm_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["answer"]},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "sources": result["sources"],
        "grounded": result["grounded"],
        "session_id": result["session_id"],
    }
