"""Temperature scaling calibration utility applied at decision engine inference runtime."""
import math
from collections.abc import Sequence


def apply_temperature_scaling(
    probabilities: dict[str, float],
    candidate_actions: Sequence[str],
    temperature: float,
) -> dict[str, float]:
    """Scales a probability distribution using temperature scalar T > 0.

    When T < 1.0, softens/sharpens the distribution (counteracting underconfidence).
    When T > 1.0, flattens the distribution (counteracting overconfidence).
    Preserves argmax ordering so predicted action remains invariant.

    Args:
        probabilities: Mapping of action candidate to probability score.
        candidate_actions: List of valid candidate actions.
        temperature: Positive scalar temperature parameter T.

    Returns:
        Calibrated probability distribution mapping.
    """
    if temperature <= 0.0 or abs(temperature - 1.0) < 1e-6:
        return probabilities

    classes = list(candidate_actions)
    # Convert probabilities to unnormalized logits: z_i = log(p_i + eps)
    logits = [math.log(max(probabilities.get(c, 1e-12), 1e-12)) for c in classes]
    scaled_logits = [z / temperature for z in logits]
    max_logit = max(scaled_logits)
    exp_logits = [math.exp(z - max_logit) for z in scaled_logits]
    sum_exp = sum(exp_logits)
    calibrated = [e / sum_exp for e in exp_logits]

    return {c: round(calibrated[i], 4) for i, c in enumerate(classes)}
