"""Monte Carlo vetorizado com probabilidades 1X2 e memoria limitada.

Somente pontos e vitorias futuras podem ser inferidos diretamente do 1X2.
Saldo e gols marcados permanecem congelados na entrada: desempates que
dependem desses campos sao aproximados; os demais seguem ordem de entrada.
"""

from dataclasses import dataclass

import numpy as np

from brasileirao538.elo import EloConfig
from brasileirao538.probabilities import Probabilities, predict_1x2


@dataclass(frozen=True)
class TeamState:
    name: str
    points: int = 0
    wins: int = 0
    goal_difference: int = 0
    goals_for: int = 0


@dataclass(frozen=True)
class Fixture:
    home: str
    away: str
    probabilities: Probabilities


@dataclass(frozen=True)
class SimulationResult:
    teams: tuple[str, ...]
    simulations: int
    seed: int
    title: dict[str, float]
    relegation: dict[str, float]
    expected_points: dict[str, float]
    position_probabilities: dict[str, tuple[float, ...]]


def fixtures_from_elo(
    fixtures: list[tuple[str, str]],
    ratings: dict[str, float],
    *,
    draw_weight: float = 0.75,
    config: EloConfig | None = None,
) -> list[Fixture]:
    """Gera probabilidades pre-torneio fixas; nao atualiza Elo durante o sorteio."""
    cfg = config or EloConfig(k_factor=30, home_advantage=80)
    if any(team not in ratings for fixture in fixtures for team in fixture):
        raise ValueError("Todos os times precisam de rating inicial")
    return [
        Fixture(h, a, predict_1x2(ratings[h], ratings[a], draw_weight, cfg))
        for h, a in fixtures
    ]


def simulate(
    teams: list[TeamState],
    fixtures: list[Fixture],
    *,
    simulations: int = 300_000,
    seed: int = 2026,
    relegation_slots: int = 4,
    batch_size: int = 10_000,
) -> SimulationResult:
    """Distribuicao de posicoes e pontos por sorteio de partidas independentes.

    Criterio: pontos, vitorias, saldo congelado, gols pro congelados,
    e indice de entrada como desempate tecnico residual.
    """
    if not teams or len({t.name for t in teams}) != len(teams):
        raise ValueError("Times vazios ou repetidos")
    if any(not t.name or t.points < 0 or t.wins < 0 or t.goals_for < 0 for t in teams):
        raise ValueError("Estados de times invalidos")
    if simulations <= 0 or batch_size <= 0:
        raise ValueError("simulations e batch_size devem ser positivos")
    n = len(teams)
    if not 0 <= relegation_slots <= n:
        raise ValueError("Quantidade de rebaixados invalida")
    index = {t.name: i for i, t in enumerate(teams)}
    for f in fixtures:
        if f.home not in index or f.away not in index or f.home == f.away:
            raise ValueError("Confronto invalido")
    home_idx = np.array([index[f.home] for f in fixtures], dtype=np.intp)
    away_idx = np.array([index[f.away] for f in fixtures], dtype=np.intp)
    home_prob = np.array([f.probabilities.home for f in fixtures])
    draw_prob = np.array([f.probabilities.draw for f in fixtures])
    initial_points = np.array([t.points for t in teams], dtype=np.int32)
    initial_wins = np.array([t.wins for t in teams], dtype=np.int32)
    gd = np.array([t.goal_difference for t in teams])
    gf = np.array([t.goals_for for t in teams])
    rng = np.random.default_rng(seed)
    positions = np.zeros((n, n), dtype=np.int64)
    point_totals = np.zeros(n, dtype=np.int64)
    processed = 0
    while processed < simulations:
        batch = min(batch_size, simulations - processed)
        points = np.broadcast_to(initial_points, (batch, n)).copy()
        wins = np.broadcast_to(initial_wins, (batch, n)).copy()
        # Uma chamada por partida; semente e ordem do calendario devem ser fixas.
        for j in range(len(fixtures)):
            random_values = rng.random(batch)
            home_win = random_values < home_prob[j]
            draw = (random_values >= home_prob[j]) & (
                random_values < home_prob[j] + draw_prob[j]
            )
            away_win = ~(home_win | draw)
            points[:, home_idx[j]] += 3 * home_win + draw
            points[:, away_idx[j]] += 3 * away_win + draw
            wins[:, home_idx[j]] += home_win
            wins[:, away_idx[j]] += away_win
        # Stable argsort preserva a ordem de entrada quando todos os critérios empatam.
        ranks = np.broadcast_to(np.arange(n), (batch, n)).copy()
        for criterion in (gf, gd):
            ranks = np.take_along_axis(
                ranks,
                np.argsort(-np.broadcast_to(criterion, (batch, n)), axis=1, stable=True),
                axis=1,
            )
        for criterion in (wins, points):
            ranks = np.take_along_axis(
                ranks,
                np.argsort(-np.take_along_axis(criterion, ranks, axis=1), axis=1, stable=True),
                axis=1,
            )
        # Inverter classificacao de clubes para contagem por posicao.
        for place in range(n):
            positions[:, place] += np.bincount(ranks[:, place], minlength=n)
        point_totals += points.sum(axis=0, dtype=np.int64)
        processed += batch
    names = tuple(t.name for t in teams)
    return SimulationResult(
        teams=names,
        simulations=simulations,
        seed=seed,
        title={team: float(positions[i, 0] / simulations) for i, team in enumerate(names)},
        relegation={
            team: float(positions[i, n - relegation_slots :].sum() / simulations)
            if relegation_slots else 0.0
            for i, team in enumerate(names)
        },
        expected_points={
            team: float(point_totals[i] / simulations) for i, team in enumerate(names)
        },
        position_probabilities={
            team: tuple(float(value / simulations) for value in positions[i])
            for i, team in enumerate(names)
        },
    )
