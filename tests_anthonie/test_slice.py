"""Verification tests for slice.py (docs/model.md section 1).
No validation targets are tested here."""
import math

import pytest

import params as P
import properties as pr
import slice as sl

N = 20
COLD = (P.COLD_FRIT_PORE, P.COLD_FRIT_EPS)
HOT = (P.HOT_FRIT_PORE, P.HOT_FRIT_EPS)


def nominal(n_r=20, cold=COLD, hot=HOT):
    """One nominal slice: N = 20, inputs from params."""
    return sl.slice_dp(P.nominal_mass_flow() / N, P.total_power() / N,
                       P.P_OUTLET, P.T_INLET, cold, hot, P.L_ELEMENT / N, n_r)


VALUE_KEYS = ("dp", "T_gas_out", "T_fuel_max", "p_out", "dp_cold_frit", "dp_bed", "dp_hot_frit")


# 1) No cold frit
def test_no_cold_frit_zero_drop():
    res = nominal(cold=None)
    assert res["dp_cold_frit"] == 0.0
    assert math.isfinite(res["dp"])


# 2) Constant-property Ergun vs closed-form radial integral
def test_cold_frit_constant_property_ergun(monkeypatch):
    rho0 = float(pr.rho(P.T_INLET, P.P_OUTLET))
    mu0 = float(pr.mu(P.T_INLET))
    monkeypatch.setattr(pr, "rho", lambda T, p=None: rho0)
    monkeypatch.setattr(pr, "mu", lambda T: mu0)

    m, dz = P.nominal_mass_flow() / N, P.L_ELEMENT / N
    d, e = COLD
    r_out, r_in = P.R_COLD_FRIT_OUT, P.R_BED_OUT
    viscous = (150 * mu0 * (1 - e)**2 / (e**3 * d**2)
               * m / (2 * math.pi * dz * rho0) * math.log(r_out / r_in))
    inertial = (1.75 * (1 - e) / (e**3 * d)
                * (m / (2 * math.pi * dz))**2 / rho0 * (1 / r_in - 1 / r_out))

    res = sl.slice_dp(m, P.total_power() / N, P.P_OUTLET, P.T_INLET,
                      COLD, HOT, dz, n_r=200)
    assert res["dp_cold_frit"] == pytest.approx(viscous + inertial, rel=1e-4)


# 3) Pressure collapse
def test_pressure_collapse_extreme_cold_frit():
    res = nominal(cold=(1e-9, P.COLD_FRIT_EPS))
    assert res["pressure_collapse"] is True
    assert res["out_of_range"] is False
    assert all(math.isnan(res[k]) for k in VALUE_KEYS)


# 4) Tsinghua geometries run and give finite values
def _t19_element_inputs():
    g = P.T19_ELEMENT
    r_hf = g["r_hot_chan"] + g["t_hot_frit"]
    r_bed = r_hf + g["t_bed"]
    power = g["power_density"] * math.pi * (r_bed**2 - r_hf**2) * g["L"]
    return dict(m_dot=g["m_dot"] / N, q_slice=power / N, p_in=g["P"], T_in=g["T_inlet"],
                cold=(g["cold_frit_pore"], g["cold_frit_eps"]),
                hot=(g["hot_frit_pore"], g["hot_frit_eps"]), dz=g["L"] / N, geom=g)


def _t19_large_inputs():
    g = P.T19_LARGE_REACTOR
    L, m_dot = g["L_options"][0], g["m_dot_options"][0]     # paired by index
    return dict(m_dot=m_dot / N, q_slice=g["power"] / N, p_in=g["P"], T_in=g["T_inlet"],
                cold=(g["frit_pore"], g["cold_frit_eps"]),
                hot=(g["frit_pore"], g["hot_frit_eps"]), dz=L / N, geom=g)


@pytest.mark.parametrize("make_inputs", [_t19_element_inputs, _t19_large_inputs],
                         ids=["T19_ELEMENT", "T19_LARGE_REACTOR"])
def test_tsinghua_geometries_finite(make_inputs):
    res = sl.slice_dp(n_r=20, **make_inputs())
    assert not res["out_of_range"] and not res["pressure_collapse"]
    assert all(math.isfinite(res[k]) for k in VALUE_KEYS)


# 5) Grid convergence in n_r
def test_grid_doubling_nominal():
    a, b = nominal(n_r=20), nominal(n_r=40)
    assert abs(b["dp"] - a["dp"]) / abs(b["dp"]) < P.GRID_TOLERANCE
    assert abs(b["T_fuel_max"] - a["T_fuel_max"]) / b["T_fuel_max"] < P.GRID_TOLERANCE


# 6) Fuel hotter than gas
def test_fuel_not_below_gas_nominal():
    res = nominal()
    assert res["T_fuel_max"] >= res["T_gas_out"]
