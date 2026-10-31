"""Distribuicao Davidson 1X2: vitoria mandante, empate, vitoria visitante."""

from dataclasses import dataclass
from math import exp, isfinite, log

from brasileirao538.elo import EloConfig


@dataclass(frozen=True)
class Probabilities:
    home: float
    draw: float
    away: float

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(not isfinite(p) or p < 0 or p > 1 for p in values):
            raise ValueError("Probabilidades invalidas")
        if abs(sum(values) - 1) > 1e-10:
            raise ValueError("Probabilidades precisam somar 1")

    def as_tuple(self) -> tuple[float, float, float]:
        return self.home, self.draw, self.away


def predict_1x2(
    home_rating: float,
    away_rating: float,
    draw_weight: float,
    config: EloConfig | None = None,
    *,
    neutral: bool = False,
) -> Probabilities:
    """Modelo Davidson numericamente estavel, com peso de empate positivo."""
    cfg = config or EloConfig()
    if not isfinite(home_rating) or not isfinite(away_rating):
        raise ValueError("Ratings invalidos")
    if not isfinite(draw_weight) or draw_weight <= 0:
        raise ValueError("Peso de empate deve ser positivo e finito")
    diff = home_rating - away_rating + (0.0 if neutral else cfg.home_advantage)
    x = diff * log(10) / cfg.scale
    logits = (x / 2, log(draw_weight), -x / 2)
    top = max(logits)
    weights = [exp(v - top) for v in logits]
    return Probabilities(*(w / sum(weights) for w in weights))


def brier_multiclass(probabilities: Probabilities, outcome: int) -> float:
    """Soma de tres erros quadraticos, escala 0 a 2."""
    if outcome not in (0, 1, 2):
        raise ValueError("Classe invalida")
    return sum((p - int(i == outcome)) ** 2 for i, p in enumerate(probabilities.as_tuple()))


def log_loss(probabilities: Probabilities, outcome: int) -> float:
    """Entropia cruzada com logaritmo natural."""
    if outcome not in (0, 1, 2):
        raise ValueError("Classe invalida")
    p = probabilities.as_tuple()[outcome]
    return -log(p) if p else float("inf")
