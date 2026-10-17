"""Verification: steady.py and stability.py solve their own equations.
No validation targets are tested here (those run only through runner.py)."""
import numpy as np
import pytest
from scipy.optimize import root

import params as P
import properties as pr
import slice as sl
import stability as stb
from steady import solve_element

M = P.nominal_mass_flow()


@pytest.fixture(scope="module")
def cosine():
    return solve_element(1.0, M, power_shape="cosine")


def test_mass_balance(cosine):
    assert cosine["m"].sum() == pytest.approx(M, rel=1e-8)


def test_energy_balance(cosine):
    absorbed = np.sum(cosine["m"] * (pr.h(cosine["T_gas_out"]) - pr.h(P.T_INLET)))
    assert absorbed == pytest.approx(cosine["q"].sum(), rel=1e-6)


def test_all_slices_share_one_pressure_drop(cosine):
    assert np.allclose(cosine["dp"], cosine["dp_element"], rtol=1e-6)


def test_symmetry_uniform_power_channel_off():
    r = solve_element(1.0, M)
    assert np.allclose(r["m"], r["m"].mean(), rtol=1e-8)


def test_power_multipliers_add_power():
    mult = np.ones(20); mult[9] = 1.2
    r = solve_element(1.0, M, power_multipliers=mult)
    assert r["q"].sum() == pytest.approx(P.total_power() * (1 + 0.2 / 20), rel=1e-9)


def test_inlet_reference_fixes_inlet_pressure():
    r = solve_element(1.0, P.T19_ELEMENT["m_dot"], geom=P.T19_ELEMENT, pressure_ref="inlet")
    assert r["p_in"] == pytest.approx(P.T19_ELEMENT["P"], rel=1e-12)
    o = solve_element(1.0, P.T19_ELEMENT["m_dot"], geom=P.T19_ELEMENT, pressure_ref="outlet")
    assert o["p_out"] == pytest.approx(P.T19_ELEMENT["P"], rel=1e-12)


def test_range_flags():
    lo = sl.slice_dp(0.05 * M / 20, 0.05 * P.total_power() / 20, P.P_OUTLET, P.T_INLET,
                     (P.COLD_FRIT_PORE, P.COLD_FRIT_EPS), (P.HOT_FRIT_PORE, P.HOT_FRIT_EPS),
                     P.L_ELEMENT / 20, 20)
    hi = sl.slice_dp(M / 20, P.total_power() / 20, P.P_OUTLET, P.T_INLET,
                     (P.COLD_FRIT_PORE, P.COLD_FRIT_EPS), (P.HOT_FRIT_PORE, P.HOT_FRIT_EPS),
                     P.L_ELEMENT / 20, 20)
    assert lo["re_m_low"] and not hi["re_m_low"]        # v3: Re_m at hot face ~4.6 at 5%, ~92 at 100%
    assert hi["re_ht_low"]                               # v3: Re ~57 at hot face, below 100


def test_profile_matches_summary():
    r = sl.slice_dp(M / 20, P.total_power() / 20, P.P_OUTLET, P.T_INLET,
                    (P.COLD_FRIT_PORE, P.COLD_FRIT_EPS), (P.HOT_FRIT_PORE, P.HOT_FRIT_EPS),
                    P.L_ELEMENT / 20, 20, profile=True)
    assert len(r["r_nodes"]) == 21
    assert max(r["T_fuel_nodes"]) == pytest.approx(r["T_fuel_max"])
    assert r["T_gas_nodes"][-1] == pytest.approx(r["T_gas_out"])


def test_static_check_ledinegg_matches_slopes(cosine):
    st = stb.static_check(cosine)
    assert st["ledinegg"] == bool(np.any(cosine["ddp_dm"] < 0))
    assert st["det_sign"] in (-1, 1)


def test_eigen_jacobian_matches_brute_force():
    # Small grid (N = 4) so brute-force re-solves are cheap. Not a validation case.
    N = 4
    m_tot = M
    res = solve_element(1.0, m_tot, power_shape="cosine", N=N, n_r=10)
    si = res["slice_inputs"]; m0, T0 = res["m"], res["T_gas_out"]

    def flows_given_T(T):
        def R(x):
            dp = np.array([stb._dp(si, i, x[i] * m0.mean(), T[i]) for i in range(N)])
            return np.r_[(dp[:-1] - dp[1:]) / 1e5, (x.sum() * m0.mean() - m_tot) / m_tot]
        return root(R, m0 / m0.mean(), method="hybr", tol=1e-13).x * m0.mean()

    _, r_hf, r_bed, _, _, alpha = sl._geometry(None)
    C = (P.FUEL_DENSITY * pr.fuel_cp(0.5 * (si["T_in"] + T0)) * (1 - alpha)
         * np.pi * (r_bed**2 - r_hf**2) * si["dz"] * 0.5)
    f = lambda T, m: (si["q"] - m * (pr.h(T) - pr.h(si["T_in"]))) / C
    J = np.empty((N, N))
    for j in range(N):
        d = 1e-3 * T0[j]; Tp, Tm = T0.copy(), T0.copy(); Tp[j] += d; Tm[j] -= d
        J[:, j] = (f(Tp, flows_given_T(Tp)) - f(Tm, flows_given_T(Tm))) / (2 * d)
    brute = np.linalg.eigvals(J).real.max()
    assert stb.eigen_check(res)["max_real"] == pytest.approx(brute, rel=1e-4)


def test_continuation_reports_where_it_stopped():
    r = solve_element(0.1, P.nominal_mass_flow(0.1), power_shape="cosine", cold_profile="none")
    if not r["converged"]:
        assert r["path"] == "continuation"
        assert "no solution found" in r["message"]