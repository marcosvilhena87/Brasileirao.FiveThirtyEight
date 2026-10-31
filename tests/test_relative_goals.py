"""Valida modelo de intensidade especifica e separacao temporal."""

import pytest

from brasileirao538.probabilities import Probabilities
from brasileirao538.relative_goals import (
    evaluate_variable,
    fit_slope,
    fixture_lambdas,
    run_experiment,
)
from brasileirao538.scores import conditioned_poisson


def test_relative_lambdas():
    p = Probabilities(0.6, 0.2, 0.2)
    h, a = fixture_lambdas(p, 1.3, 1.1, 0.2)
    assert h > 1.3
    assert a < 1.1
    assert h * a == pytest.approx(1.3 * 1.1)
    assert fixture_lambdas(p, 1.3, 1.1, 0) == (1.3, 1.1)


def test_outcome_probabilities_preserved():
    p = Probabilities(0.6, 0.2, 0.2)
    h, a = fixture_lambdas(p, 1.3, 1.1, 0.2)
    dist = conditioned_poisson(p, home_lambda=h, away_lambda=a)
    assert dist.outcome_probabilities().as_tuple() == pytest.approx(p.as_tuple())


def test_full_temporal_evaluation():
    report = run_experiment()
    assert report.validation_relative.count == 380
    assert report.holdout_relative.count == 380
    assert -0.6 <= report.beta_2023 <= 0.6
    assert -0.6 <= report.beta_final <= 0.6
    assert report.holdout_relative.exact_logloss > 0


def test_empty_records():
    with pytest.raises(ValueError):
        fit_slope([], 1.3, 1.1)
    with pytest.raises(ValueError):
        evaluate_variable([], 1.3, 1.1, 0.2)
