"""Leitura e analise reprodutivel do Brasileirao 2025 com Elo."""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from brasileirao538.elo import EloConfig, EloRatings, update_ratings


@dataclass(frozen=True)
class Match:
    season: int
    round: int
    date: date
    home: str
    away: str
    home_goals: int
    away_goals: int


def load_matches(path: str | Path) -> list[Match]:
    """Carrega e valida o conjunto de jogos de uma temporada completa."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"season", "round", "date", "home", "away", "home_goals", "away_goals"}
        if not reader.fieldnames or set(reader.fieldnames) != required:
            raise ValueError("Cabecalho CSV incompatível")
        matches = [
            Match(
                int(row["season"]), int(row["round"]), date.fromisoformat(row["date"]),
                row["home"], row["away"], int(row["home_goals"]), int(row["away_goals"])
            )
            for row in reader
        ]
    if len(matches) != 380:
        raise ValueError(f"Esperadas 380 partidas; recebidas {len(matches)}")
    teams = {team for m in matches for team in (m.home, m.away)}
    if len(teams) != 20 or {m.round for m in matches} != set(range(1, 39)):
        raise ValueError("Clubes ou rodadas inconsistentes")
    counts = Counter((m.home, m.away) for m in matches)
    if any(m.season != 2025 or m.home == m.away for m in matches):
        raise ValueError("Temporada ou confronto invalido")
    if any(c != 1 for c in counts.values()):
        raise ValueError("Confrontos duplicados")
    for team in teams:
        if sum(m.home == team for m in matches) != 19:
            raise ValueError(f"Mandos invalidos para {team}")
        if sum(m.away == team for m in matches) != 19:
            raise ValueError(f"Visitantes invalidos para {team}")
    if any(m.home_goals < 0 or m.away_goals < 0 for m in matches):
        raise ValueError("Placar negativo")
    return sorted(matches, key=lambda m: (m.date, m.round, m.home))


def standings(matches: list[Match]) -> list[dict]:
    """Calcula a classificacao sem desempates de confronto direto."""
    stats: dict[str, dict] = {}
    for m in matches:
        for team in (m.home, m.away):
            stats.setdefault(team, {"team": team, "points": 0, "played": 0,
                                    "wins": 0, "gf": 0, "ga": 0})
        h, a = stats[m.home], stats[m.away]
        h["played"] += 1
        a["played"] += 1
        h["gf"] += m.home_goals
        h["ga"] += m.away_goals
        a["gf"] += m.away_goals
        a["ga"] += m.home_goals
        if m.home_goals > m.away_goals:
            h["points"] += 3
            h["wins"] += 1
        elif m.home_goals < m.away_goals:
            a["points"] += 3
            a["wins"] += 1
        else:
            h["points"] += 1
            a["points"] += 1
    return sorted(
        stats.values(),
        key=lambda r: (-r["points"], -r["wins"], -(r["gf"] - r["ga"]), -r["gf"], r["team"]),
    )


def elo_history(matches: list[Match], config: EloConfig | None = None) -> tuple[EloRatings, list[dict]]:
    """Atualiza Elo por data; jogos no mesmo dia usam ratings do inicio do dia.

    Sem horarios completos no dataset, nao se presume ordem intradiaria.
    """
    model = EloRatings(config or EloConfig())
    timeline: list[dict] = []
    days: dict[date, list[Match]] = defaultdict(list)
    for m in matches:
        days[m.date].append(m)

    for day in sorted(days):
        snapshot = dict(model.ratings)
        changes: dict[str, float] = defaultdict(float)
        for m in sorted(days[day], key=lambda x: (x.round, x.home)):
            rh = snapshot.get(m.home, model.config.initial_rating)
            ra = snapshot.get(m.away, model.config.initial_rating)
            result = update_ratings(rh, ra, m.home_goals, m.away_goals, model.config)
            changes[m.home] += result.home_after - rh
            changes[m.away] += result.away_after - ra
            timeline.append({
                "date": day.isoformat(), "round": m.round, "home": m.home, "away": m.away,
                "expected_home": result.expected_home, "home_before": rh, "away_before": ra,
                "home_after": result.home_after, "away_after": result.away_after,
            })
        for team, delta in changes.items():
            model.ratings[team] = snapshot.get(team, model.config.initial_rating) + delta
    return model, timeline


def main() -> None:
    """Resumo retrospectivo: pontuacao oficial e ratings Elo finais."""
    path = Path(__file__).resolve().parents[2] / "data/processed/brasileirao_2025.csv"
    matches = load_matches(path)
    model, _ = elo_history(matches)
    print("Brasileirao 2025 | 380 partidas | Elo v1 retrospectivo")
    for place, item in enumerate(standings(matches), 1):
        print(f'{place:2}. {item["team"]:16} {item["points"]:2} pts | Elo {model.rating(item["team"]):.1f}')
    print("Parametros nao calibrados. Resultados conhecidos; nao e backtest preditivo.")


if __name__ == "__main__":
    main()
