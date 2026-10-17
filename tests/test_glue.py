"""Verification: rules.py, runner.py, and model.py plumbing."""
import numpy as np
import pytest

import model
import params as P
import rules
import runner


def test_rule_D8_swaps_shares_on_real_slice():
    _, rep = rules.hot_frit_dominated(P.T19_ELEMENT, model.nominal_slice)
    assert rep["swapped"]["cold_share"] == pytest.approx(1 - rep["original"]["cold_share"], abs=1e-3)


def test_runner_self_test_runs_every_check():
    results = runner.run_all(runner._StandIn())
    assert {r["id"] for r in results} == set(P.VALIDATION)
    assert not any(r["verdict"] == "error" for r in results)


def test_runner_fallback_marks_channel_checks_not_run():
    results = runner.run_all(runner._StandIn(), channel_off=True)
    for r in results:
        if P.VALIDATION[r["id"]]["needs_channel"] and r["tier"] == 1:
            assert r["verdict"] == "not run"


APP_KEYS = {"fake", "status", "status_reason", "x", "r", "m_dot", "gas_T", "fuel_T", "gas_outlet_T",
            "T_fuel_max_slice", "peak_fuel_T", "mixed_outlet_T", "dp_total", "dp_cold_frit",
            "dp_hot_frit", "m_dot_total", "flags", "solve_time_s"}


def test_app_contract_keys():
    out = model.solve(1.0, "cosine", True, 0.5, False, 1e-4, 0.0, 9, False)
    assert APP_KEYS <= set(out)
    assert out["fake"] is False
    assert out["status"] in ("stable", "runaway", "out_of_range", "no_solution", "unknown")
    assert out["fuel_T"].shape == (20, 21)


def test_app_reports_unbuilt_channel():
    if not model.CHANNEL_AVAILABLE:
        with pytest.raises(NotImplementedError):
            model.solve(1.0, "uniform", True, 0.5, False, 1e-4, 0.0, 9, True)