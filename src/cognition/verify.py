"""Response verification against policy invariants."""
from typing import Any


def verify_reply(draft: str, policy_snippet: str) -> dict[str, Any]:
    """Verifies that a drafted customer response aligns with company policy invariants.

    Checks for unauthorized promises of instant refunds against policies that
    require standard 3-5 business days bank settlement.

    Args:
        draft: Generated customer response text.
        policy_snippet: Retrieved policy document content.
    """
    is_valid = True
    issues: list[str] = []

    # Check for unauthorized promise of instant bank deposit
    if "instant refund" in draft.lower() and "3-5 business days" in policy_snippet.lower():
        is_valid = False
        issues.append("Draft promises instant refund, which violates the 3-5 business days SLA.")

    return {
        "verified": is_valid,
        "policy_grounding_score": 0.95 if is_valid else 0.40,
        "issues": issues,
    }
