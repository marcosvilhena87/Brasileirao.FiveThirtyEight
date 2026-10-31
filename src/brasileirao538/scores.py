"""Placares Poisson independentes condicionados ao resultado 1X2.

A distribuicao conjunta soma exatamente as probabilidades 1X2 informadas,
dentro do suporte truncado. Lambdas governam somente placares dentro de
cada classe; precisam ser calibrados antes de uso preditivo real.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.stats import poisson

from brasileirao538.probabilities import Probabilities


@dataclass(frozen=True)
class ScoreModel:
    """Distribuicao discreta de placares, incluindo marginais 1X2."""

    probabilities: np.ndarray
    max_goals: int

    def __post_init__(self) -> None:
        p = self.probabilities
        if p.shape != (self.max_goals + 1, self.max_goals + 1):
            raise ValueError("Dimensoes invalidas")
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Massas invalidas")
        if not np.isclose(p.sum(), 1.0, atol=1e-12):
            raise ValueError("Massas precisam somar 1")

    def outcome_probabilities(self) -> Probabilities:
        h, a = np.indices(self.probabilities.shape)
        values = (
            float(self.probabilities[h > a].sum()),
            float(self.probabilities[h == a].sum()),
            float(self.probabilities[h < a].sum()),
        )
        # Evita 1+epsilon devido apenas a arredondamento flutuante.
        return Probabilities(*(min(1.0, max(0.0, p)) for p in values))

    def sample(self, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Sorteia pares (gols_casa, gols_fora)."""
        choices = rng.choice(self.probabilities.size, size=count, p=self.probabilities.ravel())
        return np.divmod(choices, self.max_goals + 1)


def conditioned_poisson(
    probabilities: Probabilities,
    *,
    home_lambda: float = 1.29610,
    away_lambda: float = 1.15824,
    max_goals: int = 10,
) -> ScoreModel:
    """Repondera Poisson por classe para respeitar 1X2 exatamente.

    Suporte 0..max_goals por equipe. Se uma classe possui probabilidade
    positiva, precisa haver ao menos uma celula dessa classe no grid.
    """
    if (
        not isfinite(home_lambda)
        or not isfinite(away_lambda)
        or home_lambda <= 0
        or away_lambda <= 0
        or not isinstance(max_goals, int)
        or isinstance(max_goals, bool)
        or max_goals < 1
    ):
        raise ValueError("Parametros Poisson invalidos")
    goals = np.arange(max_goals + 1)
    home, away = np.meshgrid(goals, goals, indexing="ij")
    raw = poisson.pmf(home, home_lambda) * poisson.pmf(away, away_lambda)
    result = np.zeros_like(raw, dtype=float)
    for mask, target in (
        (home > away, probabilities.home),
        (home == away, probabilities.draw),
        (home < away, probabilities.away),
    ):
        total = raw[mask].sum()
        if target > 0 and total <= 0:
            raise ValueError("Suporte insuficiente para a classe")
        if total > 0:
            result[mask] = raw[mask] * target / total
    result /= result.sum()
    return ScoreModel(result, max_goals)
