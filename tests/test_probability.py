import pytest

from app.modeling.probability import confidence_from_probability, normal_cdf, probability_over


def test_normal_cdf_midpoint():
    assert normal_cdf(0.0) == pytest.approx(0.5, rel=1e-6)


def test_probability_over_increases_with_mean():
    low = probability_over(10.0, mean=8.0, std=2.0)
    high = probability_over(10.0, mean=12.0, std=2.0)
    assert high > low


def test_confidence_is_symmetric():
    assert confidence_from_probability(0.8) == pytest.approx(0.8)
    assert confidence_from_probability(0.2) == pytest.approx(0.8)


def test_confidence_handles_nonfinite_probabilities():
    assert confidence_from_probability(float("nan")) == pytest.approx(0.5)
