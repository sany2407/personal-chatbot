import asyncio
import os
import platform
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.openai_routes import router as openai_router
from app.api.routes import router
from app.core.config import get_settings
from app.core.ratelimit import SlidingWindowLimiter
from app.rag.index import ensure_index, index_notes, index_status
from app.rag.vectorstore import store_paths


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chroma and the LLM client stay lazy, but the vector index is verified
    # eagerly: hosted platforms hand us a fresh filesystem on every boot.
    settings = get_settings()
    if settings.ensure_index_on_startup:
        try:
            app.state.index = await asyncio.to_thread(ensure_index)
        except Exception as exc:  # never block boot on the index
            print(f"[index] startup check failed: {exc}", flush=True)
            app.state.index = {"status": "error", "chunks": 0, "error": str(exc)}
    else:
        app.state.index = index_status()
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

# Only the paid, model-backed routes are limited; health checks stay free.
RATE_LIMITED_PATHS = {"/api/v1/chat", "/v1/chat/completions"}
limiter = SlidingWindowLimiter(
    limit=settings.rate_limit_per_minute,
    window_seconds=settings.rate_limit_window_seconds,
)


def client_ip(request: Request) -> str:
    """Resolve the caller IP, honouring the hosting proxy's header."""
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def enforce_rate_limit(request: Request, call_next):
    if (
        settings.rate_limit_per_minute > 0
        and request.method == "POST"
        and request.url.path in RATE_LIMITED_PATHS
    ):
        allowed, retry_after = limiter.check(client_ip(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please wait a moment and try again."
                },
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


app.include_router(router, prefix="/api/v1")
# OpenAI-compatible surface: GET /v1/models, POST /v1/chat/completions
app.include_router(openai_router, prefix="/v1")


@app.get("/health", tags=["health"])
def health():
    index = index_status()
    paths = store_paths()
    payload = {
        "status": "ok",
        "service": "portfolio-rag-assistant",
        "index": {
            "status": index.get("status", "unknown"),
            "chunks": index.get("chunks", 0),
            "embedding_model": index.get("embedding_model"),
            "dir": paths["resolved"],
            "dir_writable": paths["resolved_writable"],
        },
        # Reported for monitoring: the API is up even if ingestion failed.
        "index_ready": index.get("status") == "ready" and index.get("chunks", 0) > 0,
        "environment": os.getenv("APP_ENV", settings.app_env),
        "build": {
            "commit": (os.getenv("RENDER_GIT_COMMIT") or "local")[:7],
            "python": platform.python_version(),
            "chromadb": _package_version("chromadb"),
            "uid": paths["uid"],
        },
        # Short trail of index decisions, useful when a deploy misbehaves.
        "notes": index_notes(),
    }
    if index.get("error"):
        # Surfaced so a broken deployment is diagnosable from the outside.
        payload["index"]["error"] = index["error"]
    return payload
