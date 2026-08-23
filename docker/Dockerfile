# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

# System deps: sqlite3 CLI is handy for debugging; build-essential covers any
# source builds pulled in transitively (e.g. by chromadb's dependencies).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY pyproject.toml ./

# Data directory for the SQLite file, Chroma persistence, and logs.
RUN mkdir -p /srv/app/data

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /srv/app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# The app auto-seeds the SQLite DB and FAQ vector store on first boot if
# empty (see app/main.py's lifespan handler) -- no separate init step is
# strictly required, though `python scripts/seed_db.py` can be run as a
# one-off job before scaling to multiple instances sharing a volume.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
