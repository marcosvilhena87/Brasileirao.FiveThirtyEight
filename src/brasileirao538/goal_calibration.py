"""Calibracao de gols em 2023-24 e avaliacao temporal fora da amostra em 2025."""

from dataclasses import dataclass
from math import log
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from brasileirao538.elo import EloConfig
from brasileirao538.experiment1x2 import replay_1x2, run_experiment_1x2
from brasileirao538.scores import conditioned_poisson
from brasileirao538.training import DATA_DIR, read_season, regress_ratings

MAX_GOALS = 10


@dataclass(frozen=True)
class GoalMetrics:
    count: int
    exact_logloss: float
    home_mae: float
    away_mae: float
    difference_mae: float
    difference_bias: float
    difference_variance_predicted: float
    difference_variance_observed: float


@dataclass(frozen=True)
class GoalReport:
    home_lambda: float
    away_lambda: float
    trained_matches: int
    holdout: GoalMetrics
    historical_baseline: GoalMetrics


def _observations(records: list) -> tuple[np.ndarray, np.ndarray]:
    home = np.array([m.home_goals for m, _ in records], dtype=int)
    away = np.array([m.away_goals for m, _ in records], dtype=int)
    if (home > MAX_GOALS).any() or (away > MAX_GOALS).any():
        raise ValueError("Placares fora do suporte truncado")
    return home, away


def fit_lambdas(records: list) -> tuple[float, float]:
    """MLE condicional ao 1X2, sem usar nenhuma partida do teste de 2025.

    A parcela log P(1X2) nao depende dos lambdas e nao entra na otimizacao.
    """
    if not records:
        raise ValueError("Nao ha jogos para treinamento")
    home, away = _observations(records)
    outcomes = np.sign(home - away)
    def loss(parameters: np.ndarray) -> float:
        lh, la = np.exp(parameters)
        h = np.arange(MAX_GOALS + 1)
        ph = poisson.pmf(h, lh)
        pa = poisson.pmf(h, la)
        raw = ph[:, None] * pa[None, :]
        grid_h, grid_a = np.indices(raw.shape)
        zhome = raw[grid_h > grid_a].sum()
        zdraw = raw[grid_h == grid_a].sum()
        zaway = raw[grid_h < grid_a].sum()
        normalizers = np.where(outcomes > 0, zhome, np.where(outcomes < 0, zaway, zdraw))
        likelihood = raw[home, away] / normalizers
        if np.any(likelihood <= 0):
            return float("inf")
        return float(-np.log(likelihood).mean())
    result = minimize(
        loss,
        np.log([1.45, 1.15]),
        method="L-BFGS-B",
        bounds=[(log(0.2), log(5.0)), (log(0.2), log(5.0))],
    )
    if not result.success:
        raise RuntimeError(f"Falha de otimizacao: {result.message}")
    return float(np.exp(result.x[0])), float(np.exp(result.x[1]))


def evaluate_scores(records: list, home_lambda: float, away_lambda: float) -> GoalMetrics:
    """Mede log loss exato, MAE dos gols e momentos da diferenca de gols."""
    if not records:
        raise ValueError("Amostra vazia")
    losses = []
    errors_h = []
    errors_a = []
    errors_d = []
    biases = []
    predicted_variances = []
    observed_differences = []
    support = np.arange(MAX_GOALS + 1)
    for match, p in records:
        model = conditioned_poisson(
            p, home_lambda=home_lambda, away_lambda=away_lambda, max_goals=MAX_GOALS
        )
        matrix = model.probabilities
        actual_h, actual_a = match.home_goals, match.away_goals
        if not 0 <= actual_h <= MAX_GOALS or not 0 <= actual_a <= MAX_GOALS:
            raise ValueError("Resultado fora do suporte")
        likelihood = matrix[actual_h, actual_a]
        losses.append(-np.log(likelihood) if likelihood > 0 else float("inf"))
        expected_h = float((matrix * support[:, None]).sum())
        expected_a = float((matrix * support[None, :]).sum())
        expected_d = expected_h - expected_a
        diff_grid = support[:, None] - support[None, :]
        predicted_variances.append(float((matrix * (diff_grid - expected_d) ** 2).sum()))
        actual_d = actual_h - actual_a
        errors_h.append(abs(actual_h - expected_h))
        errors_a.append(abs(actual_a - expected_a))
        errors_d.append(abs(actual_d - expected_d))
        biases.append(expected_d - actual_d)
        observed_differences.append(actual_d)
    return GoalMetrics(
        count=len(records),
        exact_logloss=float(np.mean(losses)),
        home_mae=float(np.mean(errors_h)),
        away_mae=float(np.mean(errors_a)),
        difference_mae=float(np.mean(errors_d)),
        difference_bias=float(np.mean(biases)),
        difference_variance_predicted=float(
            np.mean(predicted_variances)
            + np.var(np.array(observed_differences) + np.array(biases))
        ),
        difference_variance_observed=float(np.var(observed_differences)),
    )


def evaluate_independent_baseline(
    records: list, home_lambda: float, away_lambda: float
) -> GoalMetrics:
    """Baseline: duas Poisson independentes, parametros treinados em 2023-24."""
    h = np.arange(MAX_GOALS + 1)
    matrix = poisson.pmf(h, home_lambda)[:, None] * poisson.pmf(h, away_lambda)[None, :]
    matrix /= matrix.sum()
    observed_h = np.array([m.home_goals for m, _ in records])
    observed_a = np.array([m.away_goals for m, _ in records])
    likelihood = matrix[observed_h, observed_a]
    expected_h = float((matrix * h[:, None]).sum())
    expected_a = float((matrix * h[None, :]).sum())
    d = h[:, None] - h[None, :]
    mean_d = expected_h - expected_a
    return GoalMetrics(
        len(records),
        float(np.mean(-np.log(likelihood))),
        float(np.mean(np.abs(observed_h - expected_h))),
        float(np.mean(np.abs(observed_a - expected_a))),
        float(np.mean(np.abs(observed_h - observed_a - mean_d))),
        float(mean_d - np.mean(observed_h - observed_a)),
        float((matrix * (d - mean_d) ** 2).sum()),
        float(np.var(observed_h - observed_a)),
    )


def run_goal_experiment(data_dir: Path = DATA_DIR) -> GoalReport:
    """Lambdas ajustados apenas em 2023-24, 2025 reservado para teste."""
    s23, s24, s25 = (read_season(year, data_dir) for year in (2023, 2024, 2025))
    chosen = run_experiment_1x2(data_dir)
    cfg = EloConfig(k_factor=chosen.k, home_advantage=chosen.home_advantage)
    rec23, end23 = replay_1x2(s23, cfg, {}, chosen.draw_weight)
    teams24 = {t for m in s24 for t in (m.home, m.away)}
    rec24, end24 = replay_1x2(
        s24, cfg, regress_ratings(end23, teams24), chosen.draw_weight
    )
    lh, la = fit_lambdas(rec23 + rec24)
    teams25 = {t for m in s25 for t in (m.home, m.away)}
    rec25, _ = replay_1x2(
        s25, cfg, regress_ratings(end24, teams25), chosen.draw_weight
    )
    return GoalReport(
        lh,
        la,
        len(rec23) + len(rec24),
        evaluate_scores(rec25, lh, la),
        evaluate_independent_baseline(
            rec25,
            sum(m.home_goals for m in s23 + s24) / 760,
            sum(m.away_goals for m in s23 + s24) / 760,
        ),
    )


def main() -> None:
    report = run_goal_experiment()
    print(f"Treino: {report.trained_matches} jogos (2023-24)")
    print(f"Lambda mandante: {report.home_lambda:.5f}; visitante: {report.away_lambda:.5f}")
    for name, metrics in (
        ("Poisson condicionado + Davidson", report.holdout),
        ("Poisson historico constante", report.historical_baseline),
    ):
        print(
            f"{name}: log loss exato={metrics.exact_logloss:.5f}; "
            f"MAE gols casa={metrics.home_mae:.5f}; fora={metrics.away_mae:.5f}; "
            f"MAE saldo={metrics.difference_mae:.5f}; bias saldo={metrics.difference_bias:.5f}; "
            f"var saldo prevista={metrics.difference_variance_predicted:.5f}; "
            f"observada={metrics.difference_variance_observed:.5f}"
        )


if __name__ == "__main__":
    main()
