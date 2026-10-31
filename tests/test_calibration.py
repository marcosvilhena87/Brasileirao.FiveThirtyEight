"""Invariantes da binagem e da reamostragem por dias."""

import pytest

from brasileirao538.calibration import bootstrap_advantage, calibration_bins
from brasileirao538.probabilities import Probabilities
from brasileirao538.training import read_season


def test_all_classes_cover_every_game():
    from brasileirao538.calibration import audit

    report = audit(repetitions=100, seed=7)
    for category in ("home", "draw", "away"):
        assert sum(x.count for x in report["bins"] if x.category == category) == 380
    assert report["bootstrap"]["matches"] == 380
    assert report["bootstrap"]["date_blocks"] > 20


def test_bootstrap_is_reproducible():
    from brasileirao538.elo import EloConfig
    from brasileirao538.experiment1x2 import replay_1x2

    matches = read_season(2025)[:10]
    records, _ = replay_1x2(matches, EloConfig(), {}, 0.75)
    baseline = Probabilities(0.5, 0.25, 0.25)
    a = bootstrap_advantage(records, baseline, repetitions=100, seed=42)
    b = bootstrap_advantage(records, baseline, repetitions=100, seed=42)
    assert a == b


def test_invalid_bins():
    with pytest.raises(ValueError):
        calibration_bins([], edges=(0.2, 1))
