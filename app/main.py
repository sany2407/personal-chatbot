from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.openai_routes import router as openai_router
from app.api.routes import router
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep startup lightweight; Chroma and LLM clients are initialized on demand.
    yield


settings = get_settings()
app = FastAPI(
    title="Portfolio RAG Assistant API",
    description="Grounded, agentic Q&A over public portfolio and project information.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router, prefix="/api/v1")
# OpenAI-compatible surface: GET /v1/models, POST /v1/chat/completions
app.include_router(openai_router, prefix="/v1")


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "service": "portfolio-rag-assistant"}
