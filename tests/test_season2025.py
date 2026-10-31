"""Validacao da temporada historica e da cronologia do Elo."""

from datetime import date
from pathlib import Path

import pytest

from brasileirao538.season2025 import Match, elo_history, load_matches, standings

CSV = Path(__file__).resolve().parents[1] / "data/processed/brasileirao_2025.csv"


def test_full_2025_dataset():
    matches = load_matches(CSV)
    assert len(matches) == 380
    assert len({x.home for x in matches} | {x.away for x in matches}) == 20
    assert len({(x.home, x.away) for x in matches}) == 380


def test_2025_final_table():
    table = standings(load_matches(CSV))
    assert table[0]["team"] == "Flamengo"
    assert table[0]["points"] == 79
    assert table[1]["team"] == "Palmeiras"
    assert table[1]["points"] == 76
    assert all(item["played"] == 38 for item in table)
    assert sum(x["points"] for x in table) == 3 * 281 + 2 * 99


def test_elo_history_deterministic_and_zero_sum():
    matches = load_matches(CSV)
    model, timeline = elo_history(matches)
    other, again = elo_history(list(reversed(matches)))
    assert len(timeline) == 380
    assert model.ratings == other.ratings
    assert timeline == again
    assert sum(model.ratings.values()) == pytest.approx(20 * 1500)


def test_same_day_matches_use_pre_day_ratings():
    m1 = Match(2025, 1, date(2025, 3, 29), "A", "B", 1, 0)
    m2 = Match(2025, 2, date(2025, 3, 29), "A", "C", 0, 0)
    _, timeline = elo_history([m1, m2])
    assert timeline[0]["home_before"] == 1500
    assert timeline[1]["home_before"] == 1500
