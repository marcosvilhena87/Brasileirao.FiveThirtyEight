"""Invariantes do motor Monte Carlo 1X2."""

import pytest

from brasileirao538.elo import EloConfig
from brasileirao538.probabilities import Probabilities
from brasileirao538.simulation import Fixture, TeamState, fixtures_from_elo, simulate


def test_deterministic_300000_simulations():
    teams = [TeamState("A"), TeamState("B")]
    games = [Fixture("A", "B", Probabilities(0.5, 0.25, 0.25))]
    a = simulate(teams, games, simulations=300_000, seed=2026, relegation_slots=1)
    b = simulate(teams, games, simulations=300_000, seed=2026)
    assert a == b
    assert a.title["A"] == pytest.approx(0.75, abs=0.004)
    assert a.title["B"] == pytest.approx(0.25, abs=0.004)
    assert sum(a.title.values()) == pytest.approx(1)
    assert a.expected_points["A"] == pytest.approx(1.75, abs=0.02)
    assert a.expected_points["B"] == pytest.approx(1.0, abs=0.02)


def test_position_marginals_and_relegation():
    teams = [TeamState("A", points=20), TeamState("B", points=10), TeamState("C")]
    r = simulate(teams, [], simulations=40, relegation_slots=1)
    assert r.title == {"A": 1, "B": 0, "C": 0}
    assert r.relegation == {"A": 0, "B": 0, "C": 1}
    for team in r.teams:
        assert sum(r.position_probabilities[team]) == pytest.approx(1)


def test_wins_then_frozen_goal_difference_tiebreak():
    teams = [
        TeamState("A", points=10, wins=1, goal_difference=100),
        TeamState("B", points=10, wins=2, goal_difference=-100),
        TeamState("C", points=10, wins=2, goal_difference=0),
    ]
    r = simulate(teams, [], simulations=4, relegation_slots=1)
    assert r.title["C"] == 1
    assert r.relegation["A"] == 1


def test_fixed_one_game_result():
    teams = [TeamState("A"), TeamState("B")]
    result = simulate(
        teams,
        [Fixture("A", "B", Probabilities(1, 0, 0))],
        simulations=100,
        batch_size=7,
        relegation_slots=1,
    )
    assert result.title["A"] == 1
    assert result.expected_points["A"] == 3
    assert result.expected_points["B"] == 0


def test_elo_fixture_adapter_and_home_advantage():
    fixture = fixtures_from_elo(
        [("A", "B")],
        {"A": 1500, "B": 1500},
        config=EloConfig(home_advantage=80),
    )[0]
    assert fixture.probabilities.home > fixture.probabilities.away
    assert sum(fixture.probabilities.as_tuple()) == pytest.approx(1)


def test_invalid_fixture_and_team():
    with pytest.raises(ValueError):
        simulate([TeamState("A")], [Fixture("A", "X", Probabilities(1, 0, 0))])
    with pytest.raises(ValueError):
        simulate([TeamState("A"), TeamState("A")], [])
    with pytest.raises(ValueError):
        simulate([TeamState("A")], [], simulations=0)
    with pytest.raises(ValueError):
        fixtures_from_elo([("A", "B")], {"A": 1500})


def test_positions_sum_for_each_rank():
    teams = [TeamState(str(i)) for i in range(4)]
    fixtures = [
        Fixture("0", "1", Probabilities(0.4, 0.3, 0.3)),
        Fixture("2", "3", Probabilities(0.3, 0.3, 0.4)),
    ]
    r = simulate(teams, fixtures, simulations=1234, batch_size=123)
    for position in range(4):
        assert sum(r.position_probabilities[team][position] for team in r.teams) == pytest.approx(1)
