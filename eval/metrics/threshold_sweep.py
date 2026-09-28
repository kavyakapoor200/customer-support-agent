"""Threshold sweep analyzing trade-offs between auto-action precision and review rate."""
from collections.abc import Sequence


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
        min_thresh: Starting confidence threshold.
        max_thresh: Ending confidence threshold.
        steps: Number of evaluation steps.

    Returns:
        List of trade-off metrics per threshold step.
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

        results.append({
            "threshold": tau,
            "auto_rate": auto_rate,
            "review_rate": review_rate,
            "auto_precision": auto_precision,
            "auto_count": auto_count,
            "review_count": review_count,
        })

    return results
