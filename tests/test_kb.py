"""Tests for Policy Knowledge Base ingestion and semantic retrieval."""
from src.kb.store import PolicyStore


def test_kb_ingestion_and_search():
    """Validates markdown parsing, in-memory Qdrant ingestion, and semantic matching."""
    store = PolicyStore(url=":memory:")
    count = store.ingest_markdown_policies("data/policies")
    assert count >= 6, f"Expected at least 6 sections ingested, got {count}"

    # Test refund query
    refund_matches = store.search_policies("Can I get a refund for a charge of $40 within 14 days?", limit=2)
    assert len(refund_matches) > 0
    top_refund = refund_matches[0]
    assert "refund" in top_refund.content.lower() or "refund" in top_refund.title.lower()
    assert top_refund.score > 0.50

    # Test cancellation query
    cancel_matches = store.search_policies("I want to terminate my subscription at end of cycle", limit=2)
    assert len(cancel_matches) > 0
    top_cancel = cancel_matches[0]
    assert "cancel" in top_cancel.content.lower() or "subscription" in top_cancel.content.lower()

    # Test security P0 query
    security_matches = store.search_policies("Urgent SSO lockout Okta login failure", limit=2)
    assert len(security_matches) > 0
    top_sec = security_matches[0]
    assert "security" in top_sec.content.lower() or "sso" in top_sec.content.lower()
