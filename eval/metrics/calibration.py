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
