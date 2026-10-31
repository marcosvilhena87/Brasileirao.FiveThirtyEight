"""Testes da primeira versao Elo."""

import pytest

from brasileirao538.elo import (
    EloConfig,
    EloRatings,
    expected_home_score,
    update_ratings,
)


def test_equal_strength_neutral_is_half():
    assert expected_home_score(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_home_advantage_increases_expectation():
    assert expected_home_score(1500, 1500) > 0.5


def test_expected_score_is_not_win_probability():
    # Expectativa Elo agrega vitoria, empate e derrota numa pontuacao.
    assert 0 < expected_home_score(1500, 1500) < 1


@pytest.mark.parametrize("home_goals,away_goals,score", [(2, 0, 1), (1, 1, 0.5), (0, 2, 0)])
def test_results_preserve_total_rating(home_goals, away_goals, score):
    result = update_ratings(1500, 1500, home_goals, away_goals, neutral=True)
    assert result.score_home == score
    assert result.home_after + result.away_after == pytest.approx(3000)


def test_win_increases_rating_and_loss_decreases():
    result = update_ratings(1500, 1500, 2, 0, neutral=True)
    assert result.home_after == pytest.approx(1510)
    assert result.away_after == pytest.approx(1490)


def test_draw_with_home_advantage_penalizes_home():
    result = update_ratings(1500, 1500, 1, 1)
    assert result.home_after < 1500
    assert result.away_after > 1500


def test_neutral_ignores_home_advantage():
    config = EloConfig(home_advantage=100)
    assert expected_home_score(1500, 1500, config, neutral=True) == pytest.approx(0.5)


def test_record_updates_state_sequentially():
    ratings = EloRatings(EloConfig(home_advantage=0))
    first = ratings.record("Flamengo", "Palmeiras", 2, 0)
    second = ratings.record("Palmeiras", "Flamengo", 1, 1)
    assert first.home_after == pytest.approx(1510)
    assert second.away_before == pytest.approx(1510)
    assert ratings.rating("Flamengo") == pytest.approx(second.away_after)


def test_expected_does_not_create_clubs():
    ratings = EloRatings()
    ratings.expected("A", "B")
    assert ratings.ratings == {}


def test_invalid_match_does_not_change_state():
    ratings = EloRatings()
    with pytest.raises(ValueError):
        ratings.record("A", "B", -1, 0)
    assert ratings.ratings == {}


@pytest.mark.parametrize("home,away", [("A", "A"), ("", "B"), ("A", "")])
def test_invalid_teams(home, away):
    with pytest.raises(ValueError):
        EloRatings().record(home, away, 1, 0)


@pytest.mark.parametrize("goals", [-1, 1.5, True, "2"])
def test_invalid_goals(goals):
    with pytest.raises(ValueError):
        update_ratings(1500, 1500, goals, 1)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        EloConfig(k_factor=0)
    with pytest.raises(ValueError):
        EloConfig(scale=0)
    with pytest.raises(ValueError):
        EloConfig(home_advantage=-1)


def test_extreme_rating_gap_is_stable():
    assert expected_home_score(1e6, -1e6) == pytest.approx(1)
    assert expected_home_score(-1e6, 1e6) == pytest.approx(0)
