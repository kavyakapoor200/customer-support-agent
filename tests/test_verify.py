"""Tests for response policy verification."""
from src.cognition.verify import verify_reply


def test_verify_reply_valid():
    """Validates that a compliant draft passes policy verification."""
    draft = "We have initiated your refund. It will take 3-5 business days to appear on your bank statement."
    policy = "Section 1: Standard bank processing takes 3-5 business days."
    res = verify_reply(draft=draft, policy_snippet=policy)
    assert res["verified"] is True
    assert res["policy_grounding_score"] == 0.95
    assert len(res["issues"]) == 0


def test_verify_reply_instant_refund_violation():
    """Validates that promising instant refund violates 3-5 business days SLA policy."""
    draft = "Don't worry, we gave you an instant refund immediately!"
    policy = "Section 1: Standard bank processing takes 3-5 business days."
    res = verify_reply(draft=draft, policy_snippet=policy)
    assert res["verified"] is False
    assert res["policy_grounding_score"] == 0.40
    assert len(res["issues"]) == 1
    assert "violates the 3-5 business days SLA" in res["issues"][0]
