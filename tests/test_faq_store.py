from __future__ import annotations

from app.vectorstore.faq_store import query_faqs, seed_faq_collection


def test_seed_faq_collection_uses_bundled_fallback(isolated_env):
    count = seed_faq_collection()
    assert count > 0


def test_query_faqs_returns_relevant_results(seeded_faqs):
    results = query_faqs("What does life insurance cover?", n_results=3)
    assert len(results) > 0
    assert all({"question", "answer", "distance"} <= set(r.keys()) for r in results)


def test_query_faqs_empty_collection_returns_empty_list(isolated_env):
    # Collection exists (auto-created) but was never seeded.
    results = query_faqs("anything", n_results=3)
    assert results == []
