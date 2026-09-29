"""Pytest automated evaluation suite validating production quality gates."""
import json
from pathlib import Path

import pytest

from eval.generate_report import run_benchmark

DATASET_PATH = Path("eval/data/saas_tickets_eval.json")


@pytest.fixture(scope="module")
def eval_dataset():
    assert DATASET_PATH.is_file(), f"Dataset missing at {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_eval_accuracy_threshold():
    """Asserts overall classification accuracy meets or exceeds the 90.0% production gate."""
    results = await run_benchmark(backend="mock")
    accuracy_pct = results["accuracy"] * 100.0
    assert accuracy_pct >= 90.0, f"Expected accuracy >= 90.0%, achieved: {accuracy_pct:.1f}%"


@pytest.mark.asyncio
async def test_eval_expected_calibration_error():
    """Asserts that the statistical Expected Calibration Error (ECE) is <= 0.1500."""
    results = await run_benchmark(backend="mock")
    ece = results["ece"]
    assert ece <= 0.1500, f"Expected ECE <= 0.1500, achieved: {ece:.4f}"


@pytest.mark.asyncio
async def test_eval_option_order_invariance():
    """Asserts that reversing candidate actions does not flip predictions by more than 5.0%."""
    results = await run_benchmark(backend="mock")
    flip_rate_pct = results["flip_rate"] * 100.0
    assert flip_rate_pct <= 5.0, f"Option-order flip rate too high: {flip_rate_pct:.1f}%"


@pytest.mark.asyncio
async def test_eval_zero_false_positive_refunds_on_disputes(eval_dataset):
    """Safety Invariant: Unauthorized billing disputes must NEVER be auto-refunded without review."""
    from src.cognition.gating import evaluate_gating
    from src.decision_engine.factory import get_decision_engine

    engine = get_decision_engine("mock")
    candidates = ["refund", "cancel_subscription", "billing_dispute", "account_escalation", "general_inquiry"]

    dispute_tickets = [t for t in eval_dataset if t.get("is_fraud_or_dispute")]
    assert len(dispute_tickets) >= 10

    unauthorized_auto_refunds = 0
    for ticket in dispute_tickets:
        res = await engine.decide(ticket["text"], candidates)
        gating = evaluate_gating(res.action, res.confidence, ticket.get("amount_usd"))

        # If a dispute ticket was auto-executed as refund, that's a critical safety violation!
        if gating.outcome == "auto_execute" and res.action == "refund":
            unauthorized_auto_refunds += 1

    assert unauthorized_auto_refunds == 0, (
        f"Critical safety violation! {unauthorized_auto_refunds} dispute tickets were auto-refunded."
    )


@pytest.mark.asyncio
async def test_eval_p50_latency():
    """Asserts that decision inference P50 latency is under 50.0 ms."""
    results = await run_benchmark(backend="mock")
    assert results["p50_latency_ms"] < 50.0


def test_temperature_scaling_calibration():
    """Validates that temperature scaling adjusts overconfident probabilities and reduces ECE."""
    from eval.metrics.calibration import (
        apply_temperature_scaling,
        compute_ece,
        fit_temperature_scaling,
    )

    candidates = ["refund", "cancel_subscription", "billing_dispute"]
    # Simulated overconfident model: 98% confidence on all, but only 66% correct
    probs = [
        {"refund": 0.98, "cancel_subscription": 0.01, "billing_dispute": 0.01},
        {"refund": 0.98, "cancel_subscription": 0.01, "billing_dispute": 0.01},
        {"refund": 0.98, "cancel_subscription": 0.01, "billing_dispute": 0.01},
    ]
    labels = ["refund", "refund", "cancel_subscription"]  # 2 correct, 1 wrong

    raw_ece = compute_ece([0.98, 0.98, 0.98], [True, True, False])
    assert raw_ece > 0.30

    temp = fit_temperature_scaling(probs, labels, candidates)
    assert temp > 1.0  # Overconfidence requires T > 1.0 to soften

    scaled_probs = [apply_temperature_scaling(p, candidates, temp) for p in probs]
    scaled_confs = [p["refund"] for p in scaled_probs]
    calib_ece = compute_ece(scaled_confs, [True, True, False])

    assert calib_ece < raw_ece

