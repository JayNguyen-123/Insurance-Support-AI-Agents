"""FAQ vector store (ChromaDB) for the General Help agent's RAG lookups.

The original notebook always pulled the `deccan-ai/insuranceQA-v2` dataset
from Hugging Face at import time -- meaning the notebook (and, transitively,
this app) simply doesn't start without network access to the Hub. That's not
acceptable for a production service that needs to boot in an offline
container or CI environment.

This module seeds from a small bundled CSV (`data/faq_seed.csv`, general
insurance FAQ content) by default, and will only attempt the Hugging Face
download when `FAQ_USE_HUGGINGFACE_DATASET=true` is set -- falling back to
the bundled CSV automatically if the download fails for any reason (no
network, package not installed, dataset unavailable, etc.).
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_BUNDLED_FAQ_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "faq_seed.csv"


@lru_cache
def _get_client_for_path(persist_dir: str) -> chromadb.ClientAPI:
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _get_client() -> chromadb.ClientAPI:
    """Return the Chroma client for the *current* configured persist dir.

    Caching is keyed by path (via `_get_client_for_path`), not globally --
    a bare `@lru_cache` on a zero-arg function here would silently pin the
    client to whichever `CHROMA_PERSIST_DIR` was in effect on the first call
    for the lifetime of the process, ignoring any later config change (this
    was caught by cross-test contamination: two tests with different
    `isolated_env` tmp dirs ended up sharing the first test's client).
    """
    settings = get_settings()
    return _get_client_for_path(settings.chroma_persist_dir)


def get_collection() -> Any:
    """Return the FAQ collection, creating it (empty) if it doesn't exist yet.

    Callers should treat a freshly-created (empty) collection as "no FAQs
    indexed yet" -- run `seed_faq_collection()` (or `scripts/seed_db.py`) to
    populate it.
    """
    settings = get_settings()
    client = _get_client()
    return client.get_or_create_collection(name=settings.chroma_collection_name)


def _load_bundled_faqs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(_BUNDLED_FAQ_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if question and answer:
                rows.append({"question": question, "answer": answer})
    return rows


def _load_huggingface_faqs(sample_size: int) -> list[dict[str, str]] | None:
    """Best-effort download of the deccan-ai/insuranceQA-v2 dataset.

    Returns None (never raises) if the `datasets` package is missing, there's
    no network access, or anything else goes wrong -- callers fall back to
    the bundled CSV in that case.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("`datasets` package not installed; skipping Hugging Face FAQ download.")
        return None

    try:
        import pandas as pd

        ds = load_dataset("deccan-ai/insuranceQA-v2")
        df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        return [
            {"question": str(q), "answer": str(a)}
            for q, a in zip(df["input"], df["output"], strict=False)
        ]
    except Exception:
        logger.exception(
            "Failed to load deccan-ai/insuranceQA-v2 from Hugging Face; "
            "falling back to the bundled FAQ set."
        )
        return None


def seed_faq_collection(batch_size: int = 100) -> int:
    """(Re)populate the FAQ collection. Returns the number of documents indexed."""
    settings = get_settings()
    client = _get_client()

    # Start clean so re-running the seed script doesn't duplicate documents.
    try:
        client.delete_collection(name=settings.chroma_collection_name)
    except Exception:
        pass  # collection didn't exist yet
    collection = client.get_or_create_collection(name=settings.chroma_collection_name)

    faqs: list[dict[str, str]] | None = None
    if settings.faq_use_huggingface_dataset:
        faqs = _load_huggingface_faqs(settings.faq_sample_size)

    if not faqs:
        faqs = _load_bundled_faqs()
        logger.info("Seeding FAQ collection from bundled sample dataset (%d rows).", len(faqs))
    else:
        logger.info("Seeding FAQ collection from Hugging Face dataset (%d rows).", len(faqs))

    for i in range(0, len(faqs), batch_size):
        batch = faqs[i : i + batch_size]
        collection.add(
            documents=[f"Question: {r['question']} \n Answer: {r['answer']}" for r in batch],
            metadatas=[{"question": r["question"], "answer": r["answer"]} for r in batch],
            ids=[str(i + j) for j in range(len(batch))],
        )

    return len(faqs)


def query_faqs(query_text: str, n_results: int = 3) -> list[dict[str, Any]]:
    """Query the FAQ collection; returns a list of {question, answer, distance}."""
    collection = get_collection()
    if collection.count() == 0:
        logger.warning("FAQ collection is empty; run scripts/seed_db.py to populate it.")
        return []

    n_results = min(n_results, collection.count())
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    out: list[dict[str, Any]] = []
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]
    if metadatas and metadatas[0]:
        for meta, dist in zip(metadatas[0], distances[0], strict=False):
            out.append({
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
                "distance": dist,
            })
    return out
