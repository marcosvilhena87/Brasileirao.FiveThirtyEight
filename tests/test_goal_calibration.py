"""Verificacao da calibracao de gols com 2025 reservado para teste."""

import pytest

from brasileirao538.goal_calibration import (
    evaluate_scores,
    fit_lambdas,
    run_goal_experiment,
)
from brasileirao538.probabilities import Probabilities
from brasileirao538.training import read_season


def test_fit_uses_past_matches_and_returns_positive_values():
    from brasileirao538.elo import EloConfig
    from brasileirao538.experiment1x2 import replay_1x2

    matches = read_season(2023)
    records, _ = replay_1x2(matches, EloConfig(), {}, 0.75)
    home, away = fit_lambdas(records)
    assert 0.2 <= home <= 5
    assert 0.2 <= away <= 5


def test_metrics_finite_and_2025_held_out():
    report = run_goal_experiment()
    assert report.trained_matches == 760
    assert report.holdout.count == 380
    assert report.historical_baseline.count == 380
    assert report.home_lambda > 0
    assert report.away_lambda > 0
    assert report.holdout.exact_logloss > 0
    assert report.holdout.difference_variance_predicted >= 0


def test_empty_training_rejected():
    with pytest.raises(ValueError):
        fit_lambdas([])


def test_outcome_probabilities_do_not_change_with_goal_fit():
    from brasileirao538.scores import conditioned_poisson

    p = Probabilities(0.48, 0.27, 0.25)
    calibrated = conditioned_poisson(p, home_lambda=1.65, away_lambda=1.23)
    assert calibrated.outcome_probabilities().as_tuple() == pytest.approx(p.as_tuple())


def test_evaluate_scores_rejects_empty():
    with pytest.raises(ValueError):
        evaluate_scores([], 1.45, 1.15)
