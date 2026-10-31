"""Treino temporal sem vazamento e validacao dos datasets."""

from brasileirao538.elo import EloConfig
from brasileirao538.training import (
    evaluate_year,
    read_season,
    regress_ratings,
    run_experiment,
)


def test_seasons_complete():
    for year in (2023, 2024, 2025):
        matches = read_season(year)
        assert len(matches) == 380
        assert len({m.home for m in matches}) == 20


def test_priors_for_promoted_teams():
    priors = regress_ratings({"A": 1600}, {"A", "B"})
    assert priors == {"A": 1575, "B": 1500}


def test_2023_scores_and_2024_title():
    from brasileirao538.season2025 import standings

    table23 = standings(read_season(2023))
    table24 = standings(read_season(2024))
    assert table23[0]["team"] == "Palmeiras"
    assert table24[0]["team"] == "Botafogo"


def test_evaluation_is_deterministic():
    matches = read_season(2023)
    config = EloConfig()
    score1, ratings1 = evaluate_year(matches, config)
    score2, ratings2 = evaluate_year(list(reversed(matches)), config)
    assert score1 == score2
    assert ratings1 == ratings2


def test_2025_is_never_used_to_choose_parameters():
    report = run_experiment()
    assert report.train_2023.matches == 380
    assert report.validation_2024.matches == 380
    assert report.holdout_2025.matches == 380
    assert report.k in (10, 20, 30, 40)
    assert report.home_advantage in (0, 50, 80, 100)
