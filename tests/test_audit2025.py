"""Auditoria dos baselines sem ajustar parametros em 2025."""

import pytest

from brasileirao538.audit2025 import report_metrics, squared_error


def test_square_loss():
    assert squared_error(1, 0.6) == pytest.approx(0.16)


def test_holdout_vs_baselines():
    metrics = report_metrics()
    assert metrics["matches"] == 380
    assert metrics["k"] == 30
    assert metrics["home_advantage"] == 80
    assert metrics["mse"]["elo"] == pytest.approx(0.1552087529, abs=1e-7)
    assert metrics["mse"]["historical"] == pytest.approx(0.1681635215, abs=1e-7)
    assert metrics["mse"]["neutral"] == pytest.approx(0.1848684211, abs=1e-7)
    assert metrics["mse"]["elo"] < metrics["mse"]["historical"]
