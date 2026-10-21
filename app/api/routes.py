from fastapi import APIRouter

from app.api.errors import run_chat
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(payload: ChatRequest):
    return ChatResponse(**run_chat(payload))
