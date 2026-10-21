import logging

from fastapi import HTTPException

from app.agent.graph import run_agent
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)

# Substrings that identify an upstream quota, credit, or rate-limit rejection.
QUOTA_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "prepayment credits",
    "quota",
    "429",
    "402",
)


def run_chat(payload: ChatRequest) -> dict:
    """Run the agent and convert provider failures into a safe 503 response."""
    question = (payload.question or "").strip()
    try:
        return run_agent(question, payload.session_id)
    except Exception as exc:
        logger.exception("Chat request failed")
        if any(marker in str(exc) for marker in QUOTA_MARKERS):
            logger.error(
                "Upstream provider rejected the request for quota/billing reasons. "
                "Check the provider account (Gemini: https://ai.studio/projects)."
            )
        # Do not leak provider errors, paths, or credentials to public clients.
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again shortly.",
        ) from exc
