"""Feature and prediction drift detection using PSI and KS tests."""

import numpy as np
from scipy import stats


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI < 0.10: No significant change
    PSI 0.10-0.25: Moderate change, investigate
    PSI > 0.25: Significant change, retrain recommended

    Args:
        reference: Training distribution values.
        current: Production distribution values.
        n_bins: Number of bins for histogram comparison.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    # Convert to proportions (avoid zero division)
    ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
    cur_pct = (cur_counts + 1) / (len(current) + n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    significance: float = 0.05,
) -> dict:
    """Kolmogorov-Smirnov test for distribution drift.

    Returns:
        Dict with statistic, p_value, and is_drifted flag.
    """
    if len(reference) < 5 or len(current) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "is_drifted": False}

    stat, p_value = stats.ks_2samp(reference, current)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_drifted": p_value < significance,
    }


def check_feature_drift(
    reference_features: dict[str, np.ndarray],
    current_features: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Check drift for all features. Returns per-feature PSI and KS results."""
    results = {}
    for feature_name in reference_features:
        if feature_name not in current_features:
            continue

        ref = reference_features[feature_name]
        cur = current_features[feature_name]

        psi = population_stability_index(ref, cur)
        ks = ks_drift_test(ref, cur)

        results[feature_name] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks["statistic"], 4),
            "ks_p_value": round(ks["p_value"], 4),
            "is_drifted": psi > 0.25 or ks["is_drifted"],
            "severity": "high" if psi > 0.25 else "moderate" if psi > 0.10 else "low",
        }

    return results
