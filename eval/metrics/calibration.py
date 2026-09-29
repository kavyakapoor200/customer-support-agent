"""Expected Calibration Error (ECE) metric calculation."""
from collections.abc import Sequence


def compute_ece(confidences: Sequence[float], accuracies: Sequence[bool], n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) across confidence bins.

    Args:
        confidences: List of predicted confidence probabilities in [0.0, 1.0].
        accuracies: List of booleans indicating whether each prediction was correct.
        n_bins: Number of equal-width bins between 0.0 and 1.0 (default: 10).

    Returns:
        Scalar ECE value in [0.0, 1.0]. Lower values indicate better statistical calibration.
    """
    if len(confidences) != len(accuracies) or len(confidences) == 0:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    total_samples = len(confidences)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Filter samples falling into the current bin
        bin_indices = [
            idx for idx, conf in enumerate(confidences)
            if (bin_lower <= conf < bin_upper) or (i == n_bins - 1 and conf == bin_upper)
        ]

        bin_size = len(bin_indices)
        if bin_size == 0:
            continue

        bin_conf = sum(confidences[idx] for idx in bin_indices) / bin_size
        bin_acc = sum(1 for idx in bin_indices if accuracies[idx]) / bin_size

        ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return round(ece, 4)


def fit_temperature_scaling(
    probabilities_list: Sequence[dict[str, float]],
    expected_actions: Sequence[str],
    candidate_actions: Sequence[str],
) -> float:
    """Fits optimal temperature scalar T > 0 by minimizing Negative Log-Likelihood (NLL)

    on a held-out calibration split.

    Args:
        probabilities_list: List of predicted probability distributions.
        expected_actions: Ground truth label per ticket.
        candidate_actions: List of possible candidate labels.

    Returns:
        Optimal temperature parameter T > 0.
    """
    import numpy as np

    if not probabilities_list or len(probabilities_list) != len(expected_actions):
        return 1.0

    classes = list(candidate_actions)

    def nll(temp: float) -> float:
        loss = 0.0
        for probs, expected in zip(probabilities_list, expected_actions, strict=False):
            p_vec = np.array([probs.get(c, 1e-12) for c in classes], dtype=float)
            p_vec = np.clip(p_vec, 1e-12, 1.0)
            logits = np.log(p_vec)
            scaled = logits / temp
            scaled -= np.max(scaled)
            exp_scaled = np.exp(scaled)
            calibrated_p = exp_scaled / np.sum(exp_scaled)
            target_idx = classes.index(expected) if expected in classes else 0
            loss -= np.log(max(calibrated_p[target_idx], 1e-12))
        return float(loss / len(probabilities_list))

    best_t = 1.0
    best_loss = float("inf")
    # Fine 1D search over physically plausible temperature bounds
    for t_cand in np.linspace(0.1, 5.0, 491):
        loss_val = nll(float(t_cand))
        if loss_val < best_loss:
            best_loss = loss_val
            best_t = float(t_cand)

    return round(best_t, 4)


def apply_temperature_scaling(
    probabilities: dict[str, float],
    candidate_actions: Sequence[str],
    temperature: float,
) -> dict[str, float]:
    """Applies temperature scaling to a single probability distribution.

    Args:
        probabilities: Mapping of candidate action to raw probability.
        candidate_actions: Order of classes.
        temperature: Calibrated scalar T > 0.

    Returns:
        Calibrated probability distribution dictionary.
    """
    import numpy as np

    if temperature <= 0.0 or abs(temperature - 1.0) < 1e-6:
        return probabilities

    classes = list(candidate_actions)
    p_vec = np.array([probabilities.get(c, 1e-12) for c in classes], dtype=float)
    p_vec = np.clip(p_vec, 1e-12, 1.0)
    logits = np.log(p_vec)
    scaled = logits / temperature
    scaled -= np.max(scaled)
    exp_scaled = np.exp(scaled)
    calibrated_vec = exp_scaled / np.sum(exp_scaled)

    return {c: round(float(calibrated_vec[i]), 4) for i, c in enumerate(classes)}

