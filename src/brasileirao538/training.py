"""Calibracao cronologica: 2023 treino, 2024 validacao, 2025 teste cego.

O erro quadratico e aplicado ao escore Elo 1/0.5/0; nao equivale ao
Brier score multiclasses das probabilidades 1X2.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from brasileirao538.elo import EloConfig, EloRatings
from brasileirao538.season2025 import Match, elo_history

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def read_season(year: int, data_dir: Path = DATA_DIR) -> list[Match]:
    """Le temporada completa, sem supor sequencia pelo numero da rodada."""
    path = data_dir / f"brasileirao_{year}.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"season", "round", "date", "home", "away", "home_goals", "away_goals"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Colunas invalidas: {path}")
        from datetime import date

        matches = [
            Match(
                season=int(r["season"]),
                round=int(r["round"]),
                date=date.fromisoformat(r["date"]),
                home=r["home"],
                away=r["away"],
                home_goals=int(r["home_goals"]),
                away_goals=int(r["away_goals"]),
            )
            for r in reader
        ]
    teams = {name for m in matches for name in (m.home, m.away)}
    pairs = {(m.home, m.away) for m in matches}
    if (
        len(matches) != 380
        or len(teams) != 20
        or len(pairs) != 380
        or any(m.season != year or m.home == m.away for m in matches)
        or any(m.home_goals < 0 or m.away_goals < 0 for m in matches)
        or any(sum(m.home == t for m in matches) != 19 for t in teams)
        or any(sum(m.away == t for m in matches) != 19 for t in teams)
    ):
        raise ValueError(f"Calendario inconsistente: {year}")
    return sorted(matches, key=lambda m: (m.date, m.round, m.home))


def regress_ratings(
    previous: dict[str, float], teams: set[str], strength: float = 0.75
) -> dict[str, float]:
    """Regressao a 1500; promovidos entram em 1500 (sem informacao da Serie B)."""
    if not 0 <= strength <= 1:
        raise ValueError("strength deve estar entre 0 e 1")
    return {t: 1500 + strength * (previous.get(t, 1500) - 1500) for t in teams}


@dataclass(frozen=True)
class Evaluation:
    year: int
    mse: float
    mae: float
    matches: int


def evaluate_year(
    matches: list[Match], config: EloConfig, priors: dict[str, float] | None = None
) -> tuple[Evaluation, dict[str, float]]:
    """Mede erros pre-jogo e atualiza rating apenas apos observar o dia.

    A avaliacao e prequential: previsoes usam resultados anteriores, nunca
    futuros; o escore de um jogo e observado somente apos sua previsao.
    """
    model = EloRatings(config, dict(priors or {}))
    model, timeline = elo_history(matches, config=config) if not priors else _replay(model, matches)
    score_by_pair = {
        (m.date.isoformat(), m.home, m.away): (
            1.0 if m.home_goals > m.away_goals else 0.0 if m.home_goals < m.away_goals else 0.5
        )
        for m in matches
    }
    errors = [
        score_by_pair[(row["date"], row["home"], row["away"])] - row["expected_home"]
        for row in timeline
    ]
    return (
        Evaluation(
            matches[0].season,
            sum(e * e for e in errors) / len(errors),
            sum(abs(e) for e in errors) / len(errors),
            len(errors),
        ),
        dict(model.ratings),
    )


def _replay(model: EloRatings, matches: list[Match]) -> tuple[EloRatings, list[dict]]:
    """Reaproveita replay de dias em estado inicial fornecido."""
    from collections import defaultdict

    from brasileirao538.elo import update_ratings

    days = defaultdict(list)
    for m in matches:
        days[m.date].append(m)
    timeline = []
    for day in sorted(days):
        before = dict(model.ratings)
        deltas = defaultdict(float)
        for m in sorted(days[day], key=lambda x: (x.round, x.home)):
            rh = before.get(m.home, model.config.initial_rating)
            ra = before.get(m.away, model.config.initial_rating)
            result = update_ratings(rh, ra, m.home_goals, m.away_goals, model.config)
            deltas[m.home] += result.home_after - rh
            deltas[m.away] += result.away_after - ra
            timeline.append(
                {
                    "date": day.isoformat(),
                    "home": m.home,
                    "away": m.away,
                    "expected_home": result.expected_home,
                }
            )
        for team, delta in deltas.items():
            model.ratings[team] = before.get(team, model.config.initial_rating) + delta
    return model, timeline


@dataclass(frozen=True)
class Report:
    k: float
    home_advantage: float
    train_2023: Evaluation
    validation_2024: Evaluation
    holdout_2025: Evaluation
    final_ratings: dict[str, float]


def run_experiment(data_dir: Path = DATA_DIR) -> Report:
    """Seleciona hiperparametros por 2024; avalia 2025 uma unica vez."""
    season23 = read_season(2023, data_dir)
    season24 = read_season(2024, data_dir)
    season25 = read_season(2025, data_dir)
    candidates = []
    for k in (10.0, 20.0, 30.0, 40.0):
        for home in (0.0, 50.0, 80.0, 100.0):
            cfg = EloConfig(k_factor=k, home_advantage=home)
            tr, end23 = evaluate_year(season23, cfg)
            priors24 = regress_ratings(end23, {t for m in season24 for t in (m.home, m.away)})
            val, _ = evaluate_year(season24, cfg, priors24)
            candidates.append((val.mse, k, home, tr, val))
    _, best_k, best_home, train, validation = min(candidates)
    best = EloConfig(k_factor=best_k, home_advantage=best_home)
    _, end23 = evaluate_year(season23, best)
    priors24 = regress_ratings(end23, {t for m in season24 for t in (m.home, m.away)})
    _, end24 = evaluate_year(season24, best, priors24)
    priors25 = regress_ratings(end24, {t for m in season25 for t in (m.home, m.away)})
    holdout, final = evaluate_year(season25, best, priors25)
    return Report(best_k, best_home, train, validation, holdout, final)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibracao Elo e avaliacao holdout Brasileirao 2025"
    )
    parser.parse_args()
    report = run_experiment()
    print(f"Parametros por validacao 2024: K={report.k:g}; mando={report.home_advantage:g}")
    for label, item in (
        ("Treino 2023", report.train_2023),
        ("Validacao 2024", report.validation_2024),
        ("Teste 2025", report.holdout_2025),
    ):
        print(f"{label}: n={item.matches} MSE={item.mse:.5f} MAE={item.mae:.5f}")
    print("Ratings finais 2025 (retrospectivos):")
    for team, rating in sorted(report.final_ratings.items(), key=lambda x: -x[1]):
        print(f"  {team:18} {rating:.1f}")
    print("MSE de escore Elo; NAO Brier 1X2 nem probabilidades de titulo.")


if __name__ == "__main__":
    main()
