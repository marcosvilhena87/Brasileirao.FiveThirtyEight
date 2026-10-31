"""Intensidades Poisson especificas por confronto, derivadas do Elo-Davidson.

O logaritmo da razao p(casa)/p(fora) e o diferencial Elo ajustado ao mando,
independentemente do parametro de empate. O coeficiente beta e calibrado
sem observar a temporada de teste.
"""

from dataclasses import dataclass
from math import exp, log
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from brasileirao538.elo import EloConfig
from brasileirao538.experiment1x2 import replay_1x2, run_experiment_1x2
from brasileirao538.goal_calibration import GoalMetrics, evaluate_scores, fit_lambdas
from brasileirao538.scores import conditioned_poisson
from brasileirao538.training import DATA_DIR, read_season, regress_ratings


def fixture_lambdas(probabilities, home_lambda: float, away_lambda: float, beta: float):
    """Lambdas positivos e simetricos sob a inclinacao da forca relativa."""
    x = log(probabilities.home / probabilities.away)
    return home_lambda * exp(beta * x), away_lambda * exp(-beta * x)


def fit_slope(records: list, home_lambda: float, away_lambda: float) -> float:
    """Minimiza NLL de placar *condicional* em 1X2, com lambda global fixo."""
    if not records:
        raise ValueError("Amostra vazia")
    observed_h = np.array([m.home_goals for m, _ in records], dtype=int)
    observed_a = np.array([m.away_goals for m, _ in records], dtype=int)
    if np.any(observed_h > 10) or np.any(observed_a > 10):
        raise ValueError("Placares fora do suporte")
    x = np.array([log(p.home / p.away) for _, p in records])
    g = np.arange(11)
    mask_h, mask_a = np.indices((11, 11))
    groups = np.sign(mask_h - mask_a)
    observed_groups = np.sign(observed_h - observed_a)
    fact = gammaln(g + 1)

    def loss(beta: float) -> float:
        lh = home_lambda * np.exp(beta * x)
        la = away_lambda * np.exp(-beta * x)
        if np.any(lh > 30) or np.any(la > 30):
            return float("inf")
        log_h = g[None, :] * np.log(lh[:, None]) - lh[:, None] - fact[None, :]
        log_a = g[None, :] * np.log(la[:, None]) - la[:, None] - fact[None, :]
        raw = np.exp(log_h[:, :, None] + log_a[:, None, :])
        z = np.empty(len(records))
        for outcome in (-1, 0, 1):
            selected = observed_groups == outcome
            if selected.any():
                z[selected] = raw[selected][:, groups == outcome].sum(axis=1)
        actual = raw[np.arange(len(records)), observed_h, observed_a]
        if np.any(z <= 0) or np.any(actual <= 0):
            return float("inf")
        return float(np.mean(-np.log(actual / z)))

    result = minimize_scalar(loss, method="bounded", bounds=(-0.6, 0.6))
    if not result.success:
        raise RuntimeError("Falha na calibracao da inclinacao")
    return float(result.x)


def evaluate_variable(records: list, home_lambda: float, away_lambda: float, beta: float):
    """Mede log loss, MAEs e dispersao de saldo por confronto."""
    if not records:
        raise ValueError("Amostra vazia")
    observed_d = []
    expected_d = []
    variance_d = []
    errors = []
    support = np.arange(11)
    delta_grid = support[:, None] - support[None, :]
    for match, p in records:
        lh, la = fixture_lambdas(p, home_lambda, away_lambda, beta)
        matrix = conditioned_poisson(p, home_lambda=lh, away_lambda=la).probabilities
        actual_h, actual_a = match.home_goals, match.away_goals
        if max(actual_h, actual_a) > 10:
            raise ValueError("Placar excede suporte")
        mu_h = float((matrix * support[:, None]).sum())
        mu_a = float((matrix * support[None, :]).sum())
        mu_d = mu_h - mu_a
        observed_d.append(actual_h - actual_a)
        expected_d.append(mu_d)
        variance_d.append(float((matrix * (delta_grid - mu_d) ** 2).sum()))
        errors.append((
            -log(float(matrix[actual_h, actual_a])),
            abs(actual_h - mu_h),
            abs(actual_a - mu_a),
            abs(actual_h - actual_a - mu_d),
        ))
    a = np.array(errors)
    obs = np.array(observed_d)
    exp_d = np.array(expected_d)
    return GoalMetrics(
        len(records),
        float(a[:, 0].mean()),
        float(a[:, 1].mean()),
        float(a[:, 2].mean()),
        float(a[:, 3].mean()),
        float((exp_d - obs).mean()),
        float(np.mean(variance_d) + np.var(exp_d)),
        float(np.var(obs)),
    )


@dataclass(frozen=True)
class RelativeStrengthReport:
    home_lambda: float
    away_lambda: float
    beta_2023: float
    beta_final: float
    validation_global: GoalMetrics
    validation_relative: GoalMetrics
    holdout_global: GoalMetrics
    holdout_relative: GoalMetrics


def run_experiment(data_dir: Path = DATA_DIR) -> RelativeStrengthReport:
    """2023 treina; 2024 valida; 2023-24 retreina; 2025 testa uma vez."""
    s23, s24, s25 = (read_season(year, data_dir) for year in (2023, 2024, 2025))
    selected = run_experiment_1x2(data_dir)
    config = EloConfig(k_factor=selected.k, home_advantage=selected.home_advantage)
    rec23, end23 = replay_1x2(s23, config, {}, selected.draw_weight)
    teams24 = {t for m in s24 for t in (m.home, m.away)}
    rec24, end24 = replay_1x2(
        s24, config, regress_ratings(end23, teams24), selected.draw_weight
    )
    h23, a23 = fit_lambdas(rec23)
    beta23 = fit_slope(rec23, h23, a23)
    validation_global = evaluate_scores(rec24, h23, a23)
    validation_relative = evaluate_variable(rec24, h23, a23, beta23)
    # Ajuste final sem usar nenhum gol ou resultado de 2025.
    home_lambda, away_lambda = fit_lambdas(rec23 + rec24)
    beta_final = fit_slope(rec23 + rec24, home_lambda, away_lambda)
    teams25 = {t for m in s25 for t in (m.home, m.away)}
    rec25, _ = replay_1x2(
        s25, config, regress_ratings(end24, teams25), selected.draw_weight
    )
    return RelativeStrengthReport(
        home_lambda,
        away_lambda,
        beta23,
        beta_final,
        validation_global,
        validation_relative,
        evaluate_scores(rec25, home_lambda, away_lambda),
        evaluate_variable(rec25, home_lambda, away_lambda, beta_final),
    )


def main() -> None:
    report = run_experiment()
    print(
        f"Treino 2023 beta={report.beta_2023:.6f}; treino 2023-24 beta={report.beta_final:.6f}"
    )
    print(f"Lambda base casa={report.home_lambda:.5f}; fora={report.away_lambda:.5f}")
    for name, m in (
        ("2024 Poisson global", report.validation_global),
        ("2024 Poisson relativo", report.validation_relative),
        ("2025 Poisson global", report.holdout_global),
        ("2025 Poisson relativo", report.holdout_relative),
    ):
        print(
            f"{name}: NLL={m.exact_logloss:.5f}; MAE gols casa={m.home_mae:.5f}; "
            f"fora={m.away_mae:.5f}; MAE saldo={m.difference_mae:.5f}; "
            f"bias saldo={m.difference_bias:.5f}; var saldo prev={m.difference_variance_predicted:.5f}; "
            f"obs={m.difference_variance_observed:.5f}"
        )


if __name__ == "__main__":
    main()
