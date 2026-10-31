"""Avalia Davidson 1X2 em 2025, com escolha de peso de empate em 2024."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from brasileirao538.elo import EloConfig, EloRatings, update_ratings
from brasileirao538.probabilities import (
    Probabilities,
    brier_multiclass,
    log_loss,
    predict_1x2,
)
from brasileirao538.training import DATA_DIR, read_season, regress_ratings, run_experiment

DRAW_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def outcome(match: object) -> int:
    """0 mandante, 1 empate, 2 visitante."""
    if match.home_goals > match.away_goals:
        return 0
    if match.home_goals < match.away_goals:
        return 2
    return 1


def replay_1x2(matches: list, config: EloConfig, priors: dict, draw_weight: float):
    """Registra a previsao antes de incorporar resultados da mesma data."""
    model = EloRatings(config, dict(priors))
    days = defaultdict(list)
    for match in matches:
        days[match.date].append(match)
    records = []
    for day in sorted(days):
        snapshot = dict(model.ratings)
        deltas = defaultdict(float)
        for match in sorted(days[day], key=lambda m: (m.round, m.home)):
            home = snapshot.get(match.home, config.initial_rating)
            away = snapshot.get(match.away, config.initial_rating)
            records.append((match, predict_1x2(home, away, draw_weight, config)))
            result = update_ratings(home, away, match.home_goals, match.away_goals, config)
            deltas[match.home] += result.home_after - home
            deltas[match.away] += result.away_after - away
        for team, delta in deltas.items():
            model.ratings[team] = snapshot.get(team, config.initial_rating) + delta
    return records, dict(model.ratings)


@dataclass(frozen=True)
class Scores:
    n: int
    brier: float
    logloss: float


def evaluate(records: list) -> Scores:
    if not records:
        raise ValueError("Nao ha partidas")
    n = len(records)
    return Scores(
        n,
        sum(brier_multiclass(p, outcome(m)) for m, p in records) / n,
        sum(log_loss(p, outcome(m)) for m, p in records) / n,
    )


def historical_probabilities(matches: list) -> Probabilities:
    n = len(matches)
    return Probabilities(*(sum(outcome(m) == i for m in matches) / n for i in range(3)))


@dataclass(frozen=True)
class Report:
    k: float
    home_advantage: float
    draw_weight: float
    validation: Scores
    test_2025: Scores
    baseline_historical: Scores
    baseline_uniform: Scores


def run_experiment_1x2(data_dir: Path = DATA_DIR) -> Report:
    """2025 nunca participa de ajuste de parametro nem frequencia historica."""
    s23, s24, s25 = [read_season(y, data_dir) for y in (2023, 2024, 2025)]
    elo = run_experiment(data_dir)
    cfg = EloConfig(k_factor=elo.k, home_advantage=elo.home_advantage)
    _, end23 = replay_1x2(s23, cfg, {}, 1.0)
    teams24 = {t for m in s24 for t in (m.home, m.away)}
    priors24 = regress_ratings(end23, teams24)
    candidate_scores = [
        (evaluate(replay_1x2(s24, cfg, priors24, weight)[0]).logloss, weight)
        for weight in DRAW_WEIGHTS
    ]
    _, selected_weight = min(candidate_scores)
    validation_records, end24 = replay_1x2(s24, cfg, priors24, selected_weight)
    teams25 = {t for m in s25 for t in (m.home, m.away)}
    priors25 = regress_ratings(end24, teams25)
    test_records, _ = replay_1x2(s25, cfg, priors25, selected_weight)
    historical = historical_probabilities(s23 + s24)
    uniform = Probabilities(1 / 3, 1 / 3, 1 / 3)
    return Report(
        elo.k,
        elo.home_advantage,
        selected_weight,
        evaluate(validation_records),
        evaluate(test_records),
        evaluate([(m, historical) for m in s25]),
        evaluate([(m, uniform) for m in s25]),
    )


def main() -> None:
    report = run_experiment_1x2()
    print(
        f"Elo K={report.k:g}; mando={report.home_advantage:g}; "
        f"peso de empate={report.draw_weight:g}"
    )
    for name, score in (
        ("Validacao 2024", report.validation),
        ("Teste 2025", report.test_2025),
        ("Base historica 2023-24", report.baseline_historical),
        ("Uniforme", report.baseline_uniform),
    ):
        print(f"{name}: {score.n} jogos, Brier={score.brier:.6f}, LogLoss={score.logloss:.6f}")


if __name__ == "__main__":
    main()
