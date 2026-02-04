from __future__ import annotations

import math


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def probability_over(line: float, mean: float, std: float, *, min_std: float = 0.5) -> float:
    if std <= 0:
        std = min_std
    std = max(std, min_std)
    z = (line - mean) / std
    return 1.0 - normal_cdf(z)


def confidence_from_probability(prob_over: float) -> float:
    return max(prob_over, 1.0 - prob_over)
