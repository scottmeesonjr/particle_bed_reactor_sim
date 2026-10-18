"""Verification of the channel and plenum model (docs/model.md section 2, Amendment A2).
Uses cosine power at 80% with rule-D2 flow, which is not a validation case."""
import numpy as np
import pytest
from scipy.optimize import root

import params as P
import properties as pr
import slice as sl
import stability as stb
import steady as S

PF = 0.8
M = P.nominal_mass_flow(PF)


@pytest.fixture(scope="module")
def res():
    r = S.solve_element(PF, M, power_shape="cosine", channel=True)
    assert r["converged"]
    return r


def test_slice_equations_satisfied(res):
    assert np.max(np.abs(res["p_plenum"] - res["p_channel"] - res["dp"])) < 1.0   # Pa


def test_mass_and_energy_balance(res):
    st = res["slice_inputs"]["setup"]
    assert res["m"].sum() == pytest.approx(M, rel=1e-8)
    absorbed = np.sum(res["m"] * (pr.h(res["T_gas_out"]) - pr.h(st["T_in"])))
    assert absorbed == pytest.approx(res["q"].sum(), rel=1e-6)


def test_end_pressures(res):
    assert res["p_out"] == pytest.approx(P.P_OUTLET, rel=1e-12)
    assert res["dp_element"] == pytest.approx(res["p_in"] - res["p_out"], rel=1e-12)


def test_momentum_terms_telescope(res):
    st = res["slice_inputs"]["setup"]; m, T = res["m"], res["T_gas_out"]
    p0, pe = res["p_in"], res["p_out"]
    ppl = S.plenum_pressures(m, p0, st, friction=False)
    rho_last = float(pr.rho(st["T_in"], ppl[-2]))
    pred = p0 - ((0.5 * m[-1]) ** 2 / rho_last - m.sum() ** 2 / float(pr.rho(st["T_in"], p0))) / st["A_pl"] ** 2
    assert ppl[-1] == pytest.approx(pred, rel=1e-12)
    pch = S.channel_pressures(m, T, pe, st, friction=False)
    h = pr.h(T)
    rho0 = float(pr.rho(float(pr.T_from_h(h[0])), pch[1]))
    rhoe = float(pr.rho(float(pr.T_from_h(np.sum(m * h) / m.sum())), pe))
    pred = pe + (m.sum() ** 2 / rhoe - (0.5 * m[0]) ** 2 / rho0) / st["A_ch"] ** 2
    assert pch[0] == pytest.approx(pred, rel=1e-12)


def test_friction_continuous_and_laminar():
    Re = np.array([10.0, 100.0, 1000.0, 1450.0, 1452.0, 2999.0, 3001.0, 1e5])
    f = np.array([S.darcy_friction(r) for r in Re])
    assert f[0] == pytest.approx(6.4) and f[1] == pytest.approx(0.64)
    assert abs(f[3] - f[4]) / f[3] < 0.01 and abs(f[5] - f[6]) / f[5] < 0.01


def test_jacobian_matches_brute_force():
    N = 6
    r = S.solve_element(PF, M, power_shape="cosine", channel=True, N=N, n_r=10)
    st = r["slice_inputs"]["setup"]; si = r["slice_inputs"]; x = r["x"]

    def R(xx):
        m = xx[:N] * si["m_avg"]; p0, pe = S.end_pressures(xx[N], st)
        ppl = S.plenum_pressures(m, p0, st)
        out = [sl.slice_dp(m[i], si["q"][i], ppl[i], si["T_in"], si["colds"][i], si["hot"], si["dz"], 10)
               for i in range(N)]
        pch = S.channel_pressures(m, np.array([o["T_gas_out"] for o in out]), pe, st)
        return np.r_[(ppl - pch - np.array([o["dp"] for o in out])) / S.P_SCALE, (m.sum() - M) / M]

    J = np.empty((N + 1, N + 1))
    for j in range(N + 1):
        h = 1e-5 * max(abs(x[j]), 1.0); xp, xm = x.copy(), x.copy(); xp[j] += h; xm[j] -= h
        J[:, j] = (R(xp) - R(xm)) / (2 * h)
    assert np.max(np.abs(r["jacobian"] - J)) < 1e-4


def test_eigen_matches_brute_force_channel_on():
    N = 4
    r = S.solve_element(PF, M, power_shape="cosine", channel=True, N=N, n_r=10)
    si = r["slice_inputs"]; st = si["setup"]; m0, T0 = r["m"], r["T_gas_out"]

    def flows_given_T(T):
        def R(x):
            m = x[:N] * si["m_avg"]; p0, pe = S.end_pressures(x[N], st)
            ppl = S.plenum_pressures(m, p0, st); pch = S.channel_pressures(m, T, pe, st)
            dp = np.array([stb._dp(si, i, m[i], T[i], ppl[i]) for i in range(N)])
            return np.r_[(ppl - pch - dp) / S.P_SCALE, (m.sum() - M) / M]
        return root(R, r["x"], method="hybr", tol=1e-13).x[:N] * si["m_avg"]

    _, r_hf, r_bed, _, _, alpha = sl._geometry(None)
    C = (P.FUEL_DENSITY * pr.fuel_cp(0.5 * (si["T_in"] + T0)) * (1 - alpha)
         * np.pi * (r_bed**2 - r_hf**2) * si["dz"] * 0.5)
    f = lambda T, m: (si["q"] - m * (pr.h(T) - pr.h(si["T_in"]))) / C
    J = np.empty((N, N))
    for j in range(N):
        d = 1e-3 * T0[j]; Tp, Tm = T0.copy(), T0.copy(); Tp[j] += d; Tm[j] -= d
        J[:, j] = (f(Tp, flows_given_T(Tp)) - f(Tm, flows_given_T(Tm))) / (2 * d)
    assert stb.eigen_check(r)["max_real"] == pytest.approx(np.linalg.eigvals(J).real.max(), rel=1e-4)


@pytest.fixture(scope="module")
def res_u():
    r = S.solve_element(PF, M, power_shape="cosine", channel=True, pattern="U")
    assert r["converged"]
    return r


def test_u_pattern_satisfies_equations(res_u):
    st = res_u["slice_inputs"]["setup"]
    assert np.max(np.abs(res_u["p_plenum"] - res_u["p_channel"] - res_u["dp"])) < 1.0
    assert res_u["m"].sum() == pytest.approx(M, rel=1e-8)
    absorbed = np.sum(res_u["m"] * (pr.h(res_u["T_gas_out"]) - pr.h(st["T_in"])))
    assert absorbed == pytest.approx(res_u["q"].sum(), rel=1e-6)


def test_u_plenum_is_fed_from_the_top(res_u):
    # Friction off: momentum telescopes from the feed at z = L down to the bottom slice.
    st = res_u["slice_inputs"]["setup"]; m = res_u["m"]; p0 = res_u["p_in"]
    ppl = S.plenum_pressures(m, p0, st, friction=False)
    rho_last = float(pr.rho(st["T_in"], ppl[1]))
    pred = p0 - ((0.5 * m[0]) ** 2 / rho_last - m.sum() ** 2 / float(pr.rho(st["T_in"], p0))) / st["A_pl"] ** 2
    assert ppl[0] == pytest.approx(pred, rel=1e-12)


def test_u_equals_z_with_channel_off():
    z = S.solve_element(PF, M, power_shape="cosine", channel=False, pattern="Z")
    u = S.solve_element(PF, M, power_shape="cosine", channel=False, pattern="U")
    assert np.allclose(z["m"], u["m"], rtol=1e-12)
