"""Option-order invariance and decision flip rate metric."""
from collections.abc import Sequence

from src.decision_engine.base import BaseDecisionEngine


async def compute_flip_rate(
    engine: BaseDecisionEngine,
    samples: Sequence[dict],
    candidate_actions: list[str],
) -> float:
    """Measures the proportion of predictions where reversing candidate action ordering

    causes the top classified action to flip.

    Args:
        engine: DecisionEngine backend to evaluate.
        samples: List of sample dictionaries containing "text".
        candidate_actions: Standard list of candidate actions.

    Returns:
        Flip rate float in [0.0, 1.0]. Ideal score is 0.0 (perfect invariance).
    """
    if not samples:
        return 0.0

    reversed_candidates = list(reversed(candidate_actions))
    flips = 0

    for sample in samples:
        text = sample["text"]
        res_standard = await engine.decide(text, candidate_actions)
        res_reversed = await engine.decide(text, reversed_candidates)

        if res_standard.action != res_reversed.action:
            flips += 1

    return round(flips / len(samples), 4)
