"""Verifica marginais 1X2 e atualizacao de saldo no Monte Carlo."""

import numpy as np
import pytest

from brasileirao538.probabilities import Probabilities
from brasileirao538.scores import conditioned_poisson
from brasileirao538.simulation import Fixture, TeamState, simulate


@pytest.mark.parametrize(
    "p",
    [
        Probabilities(0.5, 0.25, 0.25),
        Probabilities(1, 0, 0),
        Probabilities(0, 1, 0),
        Probabilities(0, 0, 1),
    ],
)
def test_poisson_marginals(p):
    model = conditioned_poisson(p)
    assert model.outcome_probabilities().as_tuple() == pytest.approx(p.as_tuple(), abs=1e-12)


def test_score_sampling_consistent_with_1x2():
    rng = np.random.default_rng(123)
    model = conditioned_poisson(Probabilities(0.5, 0.25, 0.25))
    h, a = model.sample(rng, 100_000)
    assert np.mean(h > a) == pytest.approx(0.5, abs=0.006)
    assert np.mean(h == a) == pytest.approx(0.25, abs=0.006)
    assert np.mean(h < a) == pytest.approx(0.25, abs=0.006)
    assert np.all((h >= 0) & (a >= 0))


def test_invalid_lambda():
    with pytest.raises(ValueError):
        conditioned_poisson(Probabilities(0.5, 0.25, 0.25), home_lambda=-1)


def test_score_tiebreak_changes_ranking():
    teams = [
        TeamState("A", points=3, wins=1, goal_difference=0),
        TeamState("B", points=0, wins=0, goal_difference=0),
        TeamState("C", points=3, wins=1, goal_difference=0),
    ]
    matches = [Fixture("B", "C", Probabilities(1, 0, 0))]
    result = simulate(
        teams, matches, simulations=3000, seed=11, relegation_slots=1
    )
    assert result.title["A"] == 1.0
    assert result.expected_points["B"] == 3.0
    assert result.expected_points["C"] == 3.0


def test_legacy_mode_still_supported():
    teams = [TeamState("A"), TeamState("B")]
    match = Fixture("A", "B", Probabilities(0.5, 0.25, 0.25))
    result = simulate(
        teams, [match], simulations=1000, relegation_slots=1, simulate_scores=False
    )
    assert sum(result.title.values()) == pytest.approx(1.0)
