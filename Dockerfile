FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY knowledge ./knowledge

RUN useradd --create-home appuser && \
    mkdir -p /app/.chroma /app/.memory && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# Works locally (PORT unset -> 8000) and on hosts that inject $PORT (Render).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
