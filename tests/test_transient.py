"""Verification of transient.py against the steady solver and the eigenvalue test."""
import numpy as np
import pytest

import params as P
import stability
import steady
import transient as TR

F = 0.4


@pytest.mark.parametrize("cold", [None, "none"])
def test_steady_state_does_not_drift(cold):
    res = steady.solve_element(F, P.nominal_mass_flow(F), cold_profile=cold, channel=False, N=10, n_r=10)
    _, cap = TR._setup(res)
    dTdt, m, _ = TR.rhs(res, cap, np.array(res["T_gas_out"]), res["x"])
    assert np.abs(dTdt).max() < 1e-6
    assert np.allclose(m, res["m"], rtol=1e-8)


@pytest.mark.parametrize("cold", [None, "none"])
def test_growth_rate_matches_eigenvalue(cold):
    res = steady.solve_element(F, P.nominal_mass_flow(F), cold_profile=cold, channel=False, N=10, n_r=10)
    lam = stability.eigen_check(res)["max_real"]
    out = TR.march(res, 4, 0.5, 0.05, 2.0)
    dev = [np.linalg.norm(np.array(f["T"]) - np.array(res["T_gas_out"])) for f in out["frames"]]
    t = [f["t"] for f in out["frames"]]
    k = len(t) // 2
    rate = np.polyfit(t[k:], np.log(dev[k:]), 1)[0]
    assert rate == pytest.approx(lam, rel=0.01)
