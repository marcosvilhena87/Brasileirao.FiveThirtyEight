"""Calibracao 1X2 e bootstrap temporal em 2025, sem recalibrar o modelo."""

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from brasileirao538.elo import EloConfig
from brasileirao538.experiment1x2 import (
    historical_probabilities,
    outcome,
    replay_1x2,
    run_experiment_1x2,
)
from brasileirao538.probabilities import brier_multiclass, log_loss
from brasileirao538.training import read_season, regress_ratings


@dataclass(frozen=True)
class Bin:
    category: str
    lower: float
    upper: float
    count: int
    predicted: float
    observed: float


def calibration_bins(
    records: list, edges: tuple[float, ...] = (0, 0.2, 0.4, 0.6, 0.8, 1)
) -> list[Bin]:
    """Binagem fixa por classe; faixas finais incluem p=1."""
    if edges[0] != 0 or edges[-1] != 1 or any(a >= b for a, b in pairwise(edges)):
        raise ValueError("Faixas invalidas")
    output = []
    for index, category in enumerate(("home", "draw", "away")):
        for low, high in pairwise(edges):
            selected = [
                (p.as_tuple()[index], int(outcome(m) == index))
                for m, p in records
                if low <= p.as_tuple()[index] < high or (high == 1 and p.as_tuple()[index] == 1)
            ]
            if selected:
                output.append(
                    Bin(
                        category,
                        low,
                        high,
                        len(selected),
                        sum(p for p, _ in selected) / len(selected),
                        sum(y for _, y in selected) / len(selected),
                    )
                )
    return output


def bootstrap_advantage(
    records: list,
    baseline,
    *,
    repetitions: int = 5000,
    seed: int = 2025,
) -> dict:
    """IC percentil 95% da diferenca de perda baseline-modelo, por blocos de datas."""
    if repetitions < 2:
        raise ValueError("repetitions deve ser >=2")
    days = defaultdict(list)
    for match, p in records:
        y = outcome(match)
        days[match.date].append(
            (
                brier_multiclass(baseline, y) - brier_multiclass(p, y),
                log_loss(baseline, y) - log_loss(p, y),
            )
        )
    blocks = list(days.values())
    count = len(blocks)
    rng = random.Random(seed)
    samples = [[], []]
    for _ in range(repetitions):
        sums = [0.0, 0.0]
        n = 0
        for _ in range(count):
            block = blocks[rng.randrange(count)]
            n += len(block)
            for brier_delta, log_delta in block:
                sums[0] += brier_delta
                sums[1] += log_delta
        for index in (0, 1):
            samples[index].append(sums[index] / n)

    def interval(values: list[float]) -> tuple[float, float]:
        ordered = sorted(values)
        return ordered[int(0.025 * repetitions)], ordered[
            min(repetitions - 1, int(0.975 * repetitions))
        ]

    observed = [
        sum(item[index] for block in blocks for item in block) / len(records) for index in (0, 1)
    ]
    return {
        "date_blocks": count,
        "matches": len(records),
        "brier_gain": observed[0],
        "logloss_gain": observed[1],
        "brier_ci95": interval(samples[0]),
        "logloss_ci95": interval(samples[1]),
    }


def audit(repetitions: int = 5000, seed: int = 2025) -> dict:
    """Congela hiperparametros via validacao 2024; avalia somente em 2025."""
    s23, s24, s25 = (read_season(y) for y in (2023, 2024, 2025))
    selected = run_experiment_1x2()
    cfg = EloConfig(k_factor=selected.k, home_advantage=selected.home_advantage)
    _, end23 = replay_1x2(s23, cfg, {}, selected.draw_weight)
    priors24 = regress_ratings(end23, {t for m in s24 for t in (m.home, m.away)})
    _, end24 = replay_1x2(s24, cfg, priors24, selected.draw_weight)
    priors25 = regress_ratings(end24, {t for m in s25 for t in (m.home, m.away)})
    records, _ = replay_1x2(s25, cfg, priors25, selected.draw_weight)
    baseline = historical_probabilities(s23 + s24)
    return {
        "bins": calibration_bins(records),
        "bootstrap": bootstrap_advantage(records, baseline, repetitions=repetitions, seed=seed),
    }


def main() -> None:
    report = audit()
    print("Categoria, faixa, n, probabilidade media, frequencia observada")
    for row in report["bins"]:
        print(
            f"{row.category:4} [{row.lower:.1f},{row.upper:.1f}] "
            f"n={row.count:3} previsto={row.predicted:.3f} observado={row.observed:.3f}"
        )
    print("Ganho = perda do baseline historico - perda do Elo Davidson")
    for name, value in report["bootstrap"].items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
