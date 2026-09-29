"""Unit tests for the DecisionEngine adapter layer."""
import pytest
from pydantic import ValidationError

from src.decision_engine.backends.kev import KevDecisionEngine
from src.decision_engine.backends.mock import MockDecisionEngine
from src.decision_engine.base import DecisionOutput
from src.decision_engine.factory import get_decision_engine

CANDIDATES = [
    "refund",
    "cancel_subscription",
    "billing_dispute",
    "account_escalation",
    "general_inquiry"
]


@pytest.mark.asyncio
async def test_mock_engine_probability_distribution():
    """Validates that MockDecisionEngine produces normalized probabilities summing to 1.0."""
    engine = MockDecisionEngine()
    result = await engine.decide("I was charged twice, please refund my money", CANDIDATES)

    assert result.action == "refund"
    assert 0.0 <= result.confidence <= 1.0
    assert result.engine_name == "mock"
    assert result.latency_ms >= 0.0

    # Ensure all candidates are present and probabilities sum to 1.0
    for candidate in CANDIDATES:
        assert candidate in result.probabilities
        assert 0.0 <= result.probabilities[candidate] <= 1.0

    total_prob = sum(result.probabilities.values())
    assert abs(total_prob - 1.0) < 1e-3


@pytest.mark.asyncio
async def test_mock_engine_option_order_invariance():
    """Asserts that permuting candidate action list yields identical top action and scores."""
    engine = MockDecisionEngine()
    text = "Mera yearly plan cancel kardo bhai, auto debit nahi chahiye"

    permuted_1 = ["refund", "cancel_subscription", "billing_dispute"]
    permuted_2 = ["billing_dispute", "cancel_subscription", "refund"]

    res_1 = await engine.decide(text, permuted_1)
    res_2 = await engine.decide(text, permuted_2)

    assert res_1.action == "cancel_subscription"
    assert res_2.action == "cancel_subscription"
    assert abs(res_1.probabilities["cancel_subscription"] - res_2.probabilities["cancel_subscription"]) < 1e-4


@pytest.mark.asyncio
async def test_mock_engine_taxonomy_classification():
    """Tests all standard support actions for correct classification."""
    engine = MockDecisionEngine()

    test_cases = [
        ("Please issue a refund for this accidental purchase", "refund"),
        ("I need to terminate and cancel my membership immediately", "cancel_subscription"),
        ("This is an unauthorized charge that I never made, stolen card dispute", "billing_dispute"),
        ("Emergency P0 security breach, whole team locked out of Okta SSO", "account_escalation"),
        ("Where can I find the documentation guide for API endpoints?", "general_inquiry"),
    ]

    for text, expected_action in test_cases:
        res = await engine.decide(text, CANDIDATES)
        assert res.action == expected_action, f"Expected {expected_action} for text '{text}', got {res.action}"


def test_decision_output_validation_rejects_unnormalized():
    """Asserts that DecisionOutput raises ValidationError if probabilities don't sum to ~1.0."""
    with pytest.raises(ValidationError):
        DecisionOutput(
            action="refund",
            confidence=0.90,
            probabilities={"refund": 0.50, "cancel_subscription": 0.10}, # Sums to 0.60 (invalid)
            engine_name="mock",
            latency_ms=1.5,
        )


@pytest.mark.asyncio
async def test_kev_offline_fallback():
    """Asserts that KevDecisionEngine gracefully falls back to mock when explicitly requested."""
    offline_kev = KevDecisionEngine(endpoint_url="http://localhost:59999", fallback_to_mock=True, timeout=0.5)
    result = await offline_kev.decide("Duplicate payment made, send refund", CANDIDATES)

    assert result.action == "refund"
    assert result.engine_name == "kev-fallback"


@pytest.mark.asyncio
async def test_kev_strict_offline_raises():
    """Asserts that KevDecisionEngine fails loudly by default (fallback_to_mock=False) when offline."""
    strict_kev = KevDecisionEngine(endpoint_url="http://localhost:59999", timeout=0.5)
    assert strict_kev.fallback_to_mock is False

    with pytest.raises(RuntimeError, match="Kev-0.8B inference call.*failed"):
        await strict_kev.decide("Duplicate payment made, send refund", CANDIDATES)


def test_factory_engine_instantiation():
    """Validates that factory returns correct engine types."""
    mock_eng = get_decision_engine("mock")
    assert isinstance(mock_eng, MockDecisionEngine)

    with pytest.raises(ValueError):
        get_decision_engine("unsupported_backend_xyz")
