"""Auditoria reproduzivel de 2025 com baselines independentes do holdout."""

from collections import defaultdict

from brasileirao538.elo import EloConfig, EloRatings
from brasileirao538.training import (
    _replay,
    evaluate_year,
    read_season,
    regress_ratings,
    run_experiment,
)


def squared_error(score: float, expected: float) -> float:
    """Erro quadratico do escore Elo 0/0.5/1 (nao Brier 1X2)."""
    return (score - expected) ** 2


def score_for_match(match: object) -> float:
    if match.home_goals > match.away_goals:
        return 1.0
    if match.home_goals < match.away_goals:
        return 0.0
    return 0.5


def report_metrics(data_dir=None) -> dict:
    """Usa 2024 para selecionar configuracao e 2025 apenas para avaliar."""
    if data_dir is None:
        matches = {year: read_season(year) for year in (2023, 2024, 2025)}
        selection = run_experiment()
    else:
        matches = {year: read_season(year, data_dir) for year in (2023, 2024, 2025)}
        selection = run_experiment(data_dir)
    cfg = EloConfig(k_factor=selection.k, home_advantage=selection.home_advantage)
    _, end23 = evaluate_year(matches[2023], cfg)
    teams24 = {t for m in matches[2024] for t in (m.home, m.away)}
    _, end24 = evaluate_year(matches[2024], cfg, regress_ratings(end23, teams24))
    teams25 = {t for m in matches[2025] for t in (m.home, m.away)}
    priors25 = regress_ratings(end24, teams25)
    _, predictions = _replay(EloRatings(cfg, priors25), matches[2025])
    outcome = {(m.date.isoformat(), m.home, m.away): score_for_match(m) for m in matches[2025]}
    base_historical = sum(score_for_match(m) for y in (2023, 2024) for m in matches[y]) / 760
    losses = defaultdict(list)
    losses["elo"] = []
    losses["historical"] = []
    losses["neutral"] = []
    for row in predictions:
        value = outcome[(row["date"], row["home"], row["away"])]
        losses["elo"].append(squared_error(value, row["expected_home"]))
        losses["historical"].append(squared_error(value, base_historical))
        losses["neutral"].append(squared_error(value, 0.5))
    return {
        "k": selection.k,
        "home_advantage": selection.home_advantage,
        "historical_expectation": base_historical,
        "matches": len(predictions),
        "mse": {model: sum(values) / len(values) for model, values in losses.items()},
    }


def main() -> None:
    result = report_metrics()
    print(f"2025: {result['matches']} jogos; K={result['k']:g}, mando={result['home_advantage']:g}")
    print(f"Baseline historico: {result['historical_expectation']:.6f}")
    for name, mse in result["mse"].items():
        print(f"{name}: MSE={mse:.6f}")
    print("MSE de escore Elo; nao probabilidades de vitoria/empate/derrota.")


if __name__ == "__main__":
    main()
