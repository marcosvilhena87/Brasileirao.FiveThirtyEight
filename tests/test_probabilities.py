"""Testes da distribuicao Davidson e holdout temporal."""

from math import log

import pytest

from brasileirao538.elo import EloConfig
from brasileirao538.experiment1x2 import DRAW_WEIGHTS, run_experiment_1x2
from brasileirao538.probabilities import (
    Probabilities,
    brier_multiclass,
    log_loss,
    predict_1x2,
)


def test_symmetry_and_sum():
    p = predict_1x2(1500, 1500, 1, neutral=True)
    assert p.as_tuple() == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert sum(p.as_tuple()) == pytest.approx(1)


def test_home_advantage():
    cfg = EloConfig(home_advantage=80)
    assert predict_1x2(1500, 1500, 1, cfg).home > 1 / 3
    assert predict_1x2(1500, 1500, 1, cfg, neutral=True).home == pytest.approx(1 / 3)


def test_extreme_ratings_stable():
    p = predict_1x2(1e6, -1e6, 1)
    assert p.home == pytest.approx(1)


def test_scoring():
    p = Probabilities(0.5, 0.3, 0.2)
    assert brier_multiclass(p, 0) == pytest.approx(0.38)
    assert log_loss(p, 0) == pytest.approx(-log(0.5))


@pytest.mark.parametrize("weight", [0, -1, float("inf")])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        predict_1x2(1500, 1500, weight)


def test_invalid_distribution():
    with pytest.raises(ValueError):
        Probabilities(0.4, 0.4, 0.4)


def test_holdout_evaluation():
    result = run_experiment_1x2()
    assert result.draw_weight in DRAW_WEIGHTS
    assert result.validation.n == 380
    assert result.test_2025.n == 380
    assert result.baseline_historical.n == 380
    assert result.test_2025.brier > 0
    assert result.test_2025.logloss > 0
