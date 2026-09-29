"""Threshold sweep analyzing trade-offs between auto-action precision and review rate."""
import math
from collections.abc import Sequence


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes the 95% Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of correct auto-actions.
        trials: Total number of auto-actioned samples.
        confidence: Statistical confidence level (default: 0.95 -> z ~ 1.96).

    Returns:
        Tuple of (lower_bound_pct, upper_bound_pct) in [0.0, 100.0].
    """
    if trials == 0:
        return (0.0, 100.0)

    # For 95% confidence, z ~ 1.95996
    z = 1.95996
    p = successes / trials
    denom = 1.0 + (z**2) / trials
    center = (p + (z**2) / (2.0 * trials)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / trials) + (z**2) / (4.0 * (trials**2)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (round(lower * 100.0, 1), round(upper * 100.0, 1))


def sweep_thresholds(
    predictions: Sequence[dict],
    min_thresh: float = 0.50,
    max_thresh: float = 0.95,
    steps: int = 10,
) -> list[dict]:
    """Sweeps confidence thresholds from min_thresh to max_thresh.

    Args:
        predictions: List of dicts with:
            - 'confidence': float
            - 'is_correct': bool
            - 'is_fraud': bool
            - 'action': optional predicted action string
        min_thresh: Starting confidence threshold.
        max_thresh: Ending confidence threshold.
        steps: Number of evaluation steps.

    Returns:
        List of trade-off metrics per threshold step including Wilson 95% CIs.
    """
    total = len(predictions)
    if total == 0:
        return []

    step_size = (max_thresh - min_thresh) / max(steps - 1, 1)
    results = []

    for i in range(steps):
        tau = round(min_thresh + i * step_size, 2)

        # Auto executed: confidence >= tau and not fraud
        auto_samples = [p for p in predictions if p["confidence"] >= tau and not p.get("is_fraud", False)]
        auto_count = len(auto_samples)
        review_count = total - auto_count

        auto_rate = round(auto_count / total, 3)
        review_rate = round(review_count / total, 3)

        auto_correct = sum(1 for p in auto_samples if p["is_correct"])
        auto_precision = round(auto_correct / max(auto_count, 1), 3) if auto_count > 0 else 1.0

        ci_lower, ci_upper = wilson_score_interval(auto_correct, auto_count)

        # Count any unauthorized auto-refund on dispute tickets (must be 0 by code gate)
        disputed_auto_refunds = sum(
            1 for p in predictions
            if p.get("is_fraud", False) and p["confidence"] >= tau and p.get("action") == "refund"
        )

        results.append({
            "threshold": tau,
            "auto_rate": auto_rate,
            "review_rate": review_rate,
            "auto_precision": auto_precision,
            "auto_count": auto_count,
            "review_count": review_count,
            "auto_correct": auto_correct,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_str": f"[{ci_lower:.1f}%, {ci_upper:.1f}%]" if auto_count > 0 else "N/A",
            "disputed_auto_refunds": disputed_auto_refunds,
        })

    return results

