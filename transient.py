"""
transient.py -- nonlinear time march of the fuel-temperature model, docs/model.md
section 5 (and 5a: C_i uses fuel_cp at the mean of inlet and outlet temperature).

State: one temperature per slice, the slice outlet gas temperature T_i.
    C_i dT_i/dt = Q_i - m_i (h(T_i) - h_in)
Flows adjust instantly: at every instant the flows and element pressure drop solve
the same equations as the steady model (slice drop = plenum minus channel pressure,
flows sum to the total), with each slice's drop evaluated at its current outlet
temperature (heat to the gas q = m_i (h(T_i) - h_in), as in stability.eigen_check).
stability.eigen_check is the linearization of exactly this system, which the
verification tests use: small disturbances must grow or decay at its rate.

    python transient.py            # runs the demonstrations and writes results/transient_<commit>.*

DEMONSTRATIONS (fixed here before the first run; presentation, not validation)
  - Ji et al. element, uniform power, 8 MW/L (40%), flow by rule D2, channel effects off
    (Kerrebrock's parallel-passage setup, where Q2 found a converged unstable state).
  - Case "with cold frit": the published cold frit. Case "no cold frit": cold frit removed.
  - Disturbance: the middle slice (slice 10 of 20) starts 20 K hotter than its steady
    state, as after a brief local power pulse. Nothing else changes.
  - March: Heun's method (second order), dt = 0.1 s, up to 30 s.
  - The run stops early, and is labeled a runaway, when any slice outlet temperature
    leaves the property range (above 5000 K) or no flow split exists (a slice's flow
    would have to reverse). The time any slice outlet first passes the 3800 K fuel limit
    is recorded (the gas leaving a slice is cooler than its fuel, so this is conservative).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

import params as P
import properties as pr
import slice as sl
import stability
import steady

RESULTS = Path(__file__).resolve().parent / "results"
S = steady.P_SCALE
FD = 1.0e-6
RESIDUAL_TOL = 1.0e-8   # scaled residual (x 5e5 Pa = 0.005 Pa): equations hold

DEMO = dict(power_fraction=0.40, power_shape="uniform", channel=False, slice=9,
            kick_K=20.0, dt=0.1, t_end=30.0)


class Runaway(Exception):
    """No physical flow split or temperature in range: the slice has run away."""


def _setup(res):
    si = res["slice_inputs"]
    _, r_hf, r_bed, _, _, alpha = sl._geometry(si["geom"])
    v_bed = np.pi * (r_bed**2 - r_hf**2) * si["dz"]
    return si, (1.0 - alpha) * v_bed * 0.5 * P.FUEL_DENSITY


def heat_capacity(si, cap_factor, T):
    return cap_factor * pr.fuel_cp(0.5 * (si["T_in"] + np.asarray(T)))


def flows_given_T(res, T, x0):
    """Flows m (N,) and scaled unknowns x (N+1,) with slice outlet temperatures held at T."""
    si = res["slice_inputs"]; st = si["setup"]
    N, m_avg, m_total = len(T), si["m_avg"], si["m_total"]
    if np.any(T > pr.T_MAX - 1.0):
        raise Runaway("a slice outlet temperature left the property range (above 5000 K)")

    def dp(i, m, p):
        if not m > 0.0:
            raise Runaway(f"slice {i + 1}'s flow would have to reverse")
        try:
            return stability._dp(si, i, m, T[i], p)
        except (RuntimeError, pr.PropertyRangeError):
            raise Runaway(f"slice {i + 1} left the property range")

    def state(x):
        m = x[:N] * m_avg
        p0, p_exit = steady.end_pressures(x[N], st)
        p_pl = steady.plenum_pressures(m, p0, st)
        p_ch = steady.channel_pressures(m, T, p_exit, st)
        return m, p_pl, p_ch

    def residual(x):
        m, p_pl, p_ch = state(x)
        d = np.array([dp(i, m[i], p_pl[i]) for i in range(N)])
        return np.r_[(p_pl - p_ch - d) / S, (m.sum() - m_total) / m_total]

    def jacobian(x):
        m, p_pl, _ = state(x)
        d0 = np.array([dp(i, m[i], p_pl[i]) for i in range(N)])
        hm = FD * np.maximum(m, m_avg)
        hp = FD * S
        a = (np.array([dp(i, m[i] + hm[i], p_pl[i]) for i in range(N)]) - d0) / hm
        c = (np.array([dp(i, m[i], p_pl[i] + hp) for i in range(N)]) - d0) / hp
        _, _, dPpl, dPch_x, _ = steady.manifold_derivatives(m, T, x[N], st, m_avg)
        J = np.zeros((N + 1, N + 1))
        J[:N] = (dPpl - dPch_x - c[:, None] * dPpl) / S
        J[np.arange(N), np.arange(N)] -= a * m_avg / S
        J[N, :N] = m_avg / m_total
        return J

    try:
        sol = root(residual, x0, jac=jacobian, method="hybr")
        # Judge success by whether the equations hold, not by the solver's progress test:
        # near a converged state hybr can report "not making good progress" while the
        # residual is already at noise level.
        ok = sol.success or float(np.max(np.abs(residual(sol.x)))) < RESIDUAL_TOL
    except Runaway:
        raise
    except pr.PropertyRangeError:
        raise Runaway("the channel gas left the property range")
    if not ok:
        raise Runaway(f"no flow split exists ({sol.message})")
    return sol.x[:N] * m_avg, sol.x


def rhs(res, cap_factor, T, x0):
    si = res["slice_inputs"]
    m, x = flows_given_T(res, T, x0)
    dTdt = (si["q"] - m * (pr.h(T) - pr.h(si["T_in"]))) / heat_capacity(si, cap_factor, T)
    return dTdt, m, x


def march(res, kick_slice, kick_K, dt, t_end):
    """Heun's method from the steady state with one slice kicked. Returns a dict of frames."""
    si, cap_factor = _setup(res)
    T = np.array(res["T_gas_out"], dtype=float)
    T[kick_slice] += kick_K
    x = np.array(res["x"], dtype=float)
    frames, t, outcome, reason, t_limit = [], 0.0, "settled", "", None
    try:
        while True:
            k1, m, x = rhs(res, cap_factor, T, x)
            frames.append(dict(t=round(t, 6), T=T.tolist(), m=m.tolist()))
            if t_limit is None and T.max() > P.T_LIMIT:
                t_limit = t
            if t >= t_end - 1e-9:
                break
            T_pred = T + dt * k1
            k2, _, _ = rhs(res, cap_factor, T_pred, x)
            T = T + 0.5 * dt * (k1 + k2)
            t += dt
    except Runaway as e:
        outcome, reason = "runaway", str(e)
    if outcome == "settled":
        dev = np.max(np.abs(np.array(frames[-1]["T"]) - np.array(res["T_gas_out"])))
        outcome = "settled" if dev < 1.0 else ("runaway" if dev > kick_K else "still moving")
    return dict(frames=frames, outcome=outcome, reason=reason, t_end=frames[-1]["t"],
                t_fuel_limit=t_limit, T_steady=list(map(float, res["T_gas_out"])),
                m_steady=list(map(float, res["m"])))


def commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", ".",
                                              ":(exclude)results"], stderr=subprocess.DEVNULL, text=True).strip())
        return h, dirty
    except Exception:
        return "nocommit", True


def demo_case(cold_profile):
    f = DEMO["power_fraction"]
    res = steady.solve_element(f, P.nominal_mass_flow(f), power_shape=DEMO["power_shape"],
                               cold_profile=cold_profile, channel=DEMO["channel"])
    if not res["converged"]:
        raise RuntimeError(f"steady state not found: {res['message']}")
    lam = float(stability.eigen_check(res)["max_real"])
    out = march(res, DEMO["slice"], DEMO["kick_K"], DEMO["dt"], DEMO["t_end"])
    out.update(eigen_max_real=lam, power_MW_per_L=f * P.POWER_DENSITY / 1e9)
    return out


if __name__ == "__main__":
    cases = {"with_cold_frit": None, "no_cold_frit": "none"}
    out = dict(demo=DEMO, cases={})
    for name, cold in cases.items():
        r = demo_case(cold)
        out["cases"][name] = r
        print(f"{name}: {r['outcome']} at t = {r['t_end']:.1f} s "
              f"({r['reason'] or 'ran to the end'}); eigenvalue {r['eigen_max_real']:+.3f} 1/s; "
              f"first above {P.T_LIMIT:.0f} K at {r['t_fuel_limit']} s", flush=True)
    h, dirty = commit()
    out.update(commit=h, uncommitted_changes=dirty)
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"transient_{h}.json"
    p.write_text(json.dumps(out, default=float))
    print(f"wrote {p.relative_to(RESULTS.parent)}")
