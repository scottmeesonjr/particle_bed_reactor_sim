"""
steady.py -- full-element steady state, docs/model.md sections 2-4.

Unknowns (scaled):  x_i = m_i / m_avg (i = 0..N-1),  x_N = (p_in - p_out) / P_SCALE
                    pressure_ref="outlet": p_out fixed at the reference pressure;
                    pressure_ref="inlet":  p_in fixed at the reference pressure.
Residuals:          R_i = (p_pl(z_i) - p_ch(z_i) - dp_slice_i) / P_SCALE
                    R_N = (sum m_i - m_total) / m_total
Solved with scipy.optimize.root(method="hybr").

channel=False: p_pl = p_in and p_ch = p_out for every slice. Block Jacobian.
channel=True (Z pattern, section 2): plenum marched up from p_in at z = 0,
  channel marched down from p_out at z = L, slices at their centres z_i.
  Full finite-difference Jacobian (every residual depends on every flow).
  Momentum: channel inflow carries no axial momentum, -Delta(W^2/(rho A^2));
  plenum outflow takes its momentum, -(W/A) Delta(W/(rho A)) (recovery 1/2 rho u^2).
  Discretisation: the flow at slice centre i excludes slice i ("flows below z",
  "already drawn"); segments between centres carry the flow after slice i;
  half segments z = 0 -> z_0 (plenum) and z_{N-1} -> L (channel).
  The implicit density in each segment step is solved by fixed-point iteration.

A slice that returns out_of_range or pressure_collapse, at the solution or at
any trial point the solver visits, stops the solve with converged=False and
that reason. Nothing is extrapolated.
"""
import math
import time

import numpy as np
from scipy.optimize import root

import params as P
import properties as pr
from slice import slice_dp

P_SCALE = 5.0e5        # Pa, unknown and residual scale (docs/model.md section 3)
N_SHAPE_AVG = 50       # points per slice for the slice-averaged power shape (section 3)
FD_REL_STEP = 1.0e-6   # relative finite-difference step for the Jacobians
RE_MIN_PETUKHOV = 3000.0   # flag manifold segments below this (Petukhov range, D6)
# TODO: move RE_MIN_PETUKHOV to params.py (Scott's file); docs/model.md section 2.
SEG_ITER_MAX = 50      # fixed-point iterations per manifold segment
SEG_ITER_TOL = 1.0e-12 # relative pressure change to stop iterating
STOP_FLAGS = ("out_of_range", "pressure_collapse")
FLAG_KEYS = ("re_m_low", "re_ht_low", "out_of_range", "pressure_collapse", "fuel_over_limit")


class _Failure(Exception):
    """Stops the solve; reported as converged=False with this reason."""
    def __init__(self, reason, message, slice_index=None):
        super().__init__(message)
        self.reason, self.message, self.slice_index = reason, message, slice_index


# ----------------------------------------------------------------------------
# Element description: nominal (params) or a params.py geometry dict
# ----------------------------------------------------------------------------
def _element(geom):
    """Length, radii, power at power_frac = 1, inlet T, reference pressure, published frits."""
    if geom is None:
        return dict(L=P.L_ELEMENT, r_ch=P.R_HOT_CHAN, r_cf=P.R_COLD_FRIT_OUT, r_el=P.R_ELEMENT,
                    power=P.total_power(1.0), T_in=P.T_INLET, p_ref=P.P_OUTLET,
                    cold=(P.COLD_FRIT_PORE, P.COLD_FRIT_EPS),
                    hot=(P.HOT_FRIT_PORE, P.HOT_FRIT_EPS))
    if "L" not in geom:
        raise ValueError("geom needs one length 'L'. For T19_LARGE_REACTOR use "
                         "steady.t19_large_reactor(L), which returns (geom, m_total).")
    if ("power" in geom) == ("power_density" in geom):
        raise ValueError("geom must give exactly one of 'power' or 'power_density'")
    r_ch = geom["r_hot_chan"]
    r_hf = r_ch + geom["t_hot_frit"]
    r_bed = r_hf + geom["t_bed"]
    r_cf = r_bed + geom["t_cold_frit"]
    r_el = r_cf + geom["t_plenum"]
    L = geom["L"]
    power = geom["power"] if "power" in geom else geom["power_density"] * math.pi * (r_bed**2 - r_hf**2) * L
    cold_pore = geom.get("cold_frit_pore", geom.get("frit_pore"))
    hot_pore = geom.get("hot_frit_pore", geom.get("frit_pore"))
    return dict(L=L, r_ch=r_ch, r_cf=r_cf, r_el=r_el, power=power,
                T_in=geom["T_inlet"], p_ref=geom["P"],
                cold=(cold_pore, geom["cold_frit_eps"]), hot=(hot_pore, geom["hot_frit_eps"]))


def t19_large_reactor(L):
    """params.T19_LARGE_REACTOR at core length L, with its matching mass flow.
    Returns (geom, m_total). L must be one of the published L_options."""
    g = P.T19_LARGE_REACTOR
    for L_opt, m_opt in zip(g["L_options"], g["m_dot_options"]):
        if math.isclose(L, L_opt, rel_tol=0.0, abs_tol=1e-12):
            return dict(g, L=L_opt), m_opt
    raise ValueError(f"L = {L} is not one of T19_LARGE_REACTOR['L_options'] = {g['L_options']}")


def _slice_inputs(el, power_frac, power_shape, cold_profile, hot, N):
    s = (np.arange(N) + 0.5) / N                     # slice centres, z_i / L
    # Slice-averaged shape: 50 sub-interval midpoints per slice (docs/model.md section 3)
    sub = (np.arange(N)[:, None] + (np.arange(N_SHAPE_AVG) + 0.5)[None, :] / N_SHAPE_AVG) / N
    F = P.power_shape(power_shape, sub).mean(axis=1)
    F = F / F.mean()                                 # discrete mean exactly 1
    q = el["power"] * power_frac * F / N

    if cold_profile is None:
        colds = [el["cold"]] * N
    elif isinstance(cold_profile, str) and cold_profile == "none":
        colds = [None] * N
    elif callable(cold_profile):
        colds = [tuple(cold_profile(si)) for si in s]
    else:
        raise ValueError('cold_profile must be None, "none", or a callable s -> (pore, porosity)')

    hot = el["hot"] if hot is None else tuple(hot)
    return s, q, colds, hot


# ----------------------------------------------------------------------------
# Plenum and channel (docs/model.md section 2)
# ----------------------------------------------------------------------------
def _petukhov(Re):
    a = 0.790 * math.log(Re) - 1.64 if Re > 0.0 else -1.0
    if not a > 0.0:
        raise _Failure("friction_undefined", f"Petukhov friction undefined at Re = {Re:.4g}")
    return a**-2


def _segment(kind, p_known, upstream_known, W_a, W_b, W_s, T_a, T_b, T_s, ell, A, D_h):
    """One manifold segment from a (upstream) to b (downstream) in the flow direction
    (docs/model.md section 2):
      channel: p_b = p_a - (W_b^2/(rho_b A^2) - W_a^2/(rho_a A^2)) - friction
      plenum:  p_b = p_a - ((W_a + W_b)/(2A)) (W_b/(rho_b A) - W_a/(rho_a A)) - friction
      friction = f_D W_s^2 / (2 rho_s A^2 D_h) ell
    One end pressure is known; the other is found by fixed-point iteration.
    Returns (unknown end pressure, segment Re)."""
    Re = W_s * D_h / (A * float(pr.mu(T_s)))
    f = _petukhov(Re)

    def vel(W, T, p):                                # u = W / (rho A)
        return 0.0 if W == 0.0 else W / (float(pr.rho(T, p)) * A)

    def momentum(p_a, p_b):
        u_a, u_b = vel(W_a, T_a, p_a), vel(W_b, T_b, p_b)
        if kind == "channel":                        # inflow carries no axial momentum
            return (W_b * u_b - W_a * u_a) / A
        return 0.5 * (W_a + W_b) / A * (u_b - u_a)   # plenum: outflow takes its momentum

    p_other = p_known
    for _ in range(SEG_ITER_MAX):
        p_a, p_b = (p_known, p_other) if upstream_known else (p_other, p_known)
        if not (p_a > 0.0 and p_b > 0.0):
            raise _Failure("pressure_collapse", "manifold pressure reached 0 or below")
        mom = momentum(p_a, p_b)
        fric = f * W_s * W_s / (2.0 * float(pr.rho(T_s, 0.5 * (p_a + p_b))) * A * A * D_h) * ell
        new = p_a - mom - fric if upstream_known else p_b + mom + fric
        if abs(new - p_other) <= SEG_ITER_TOL * abs(new):
            if not new > 0.0:
                raise _Failure("pressure_collapse", "manifold pressure reached 0 or below")
            return new, Re
        p_other = new
    raise _Failure("manifold_iteration", "manifold segment pressure did not converge")


def _plenum(el, m, p_in, dz):
    """Plenum pressures at slice centres (before each slice draws), marched up from z = 0."""
    N = len(m)
    A = math.pi * (el["r_el"]**2 - el["r_cf"]**2)
    D_h = 2.0 * (el["r_el"] - el["r_cf"])
    T = el["T_in"]
    W = m.sum() - np.concatenate([[0.0], np.cumsum(m)[:-1]])   # W at centre i, slice i not drawn
    p = np.empty(N)
    Re = np.empty(N)
    p[0], Re[0] = _segment("plenum", p_in, True, W[0], W[0], W[0], T, T, T, 0.5 * dz, A, D_h)
    for i in range(N - 1):
        p[i + 1], Re[i + 1] = _segment("plenum", p[i], True, W[i], W[i + 1], W[i + 1], T, T, T, dz, A, D_h)
    return p, Re


def _channel(el, m, h_out, p_out, dz):
    """Channel pressures at slice centres (before each slice enters), marched down from z = L.
    Returns (p, Re per segment, T at exit)."""
    N = len(m)
    A = math.pi * el["r_ch"]**2
    D_h = 2.0 * el["r_ch"]
    W_after = np.cumsum(m)                           # flow after slice i has entered
    T_after = np.array([float(pr.T_from_h(H / Wa))
                        for H, Wa in zip(np.cumsum(m * h_out), W_after)])
    W = np.concatenate([[0.0], W_after[:-1]])        # W at centre i, slice i not entered
    T = np.concatenate([[T_after[0]], T_after[:-1]]) # (T at centre 0 unused: W = 0)
    p = np.empty(N)
    Re = np.empty(N)
    p[N - 1], Re[N - 1] = _segment("channel", p_out, False, W[N - 1], W_after[-1], W_after[-1],
                                   T[N - 1], T_after[-1], T_after[-1], 0.5 * dz, A, D_h)
    for i in range(N - 2, -1, -1):
        p[i], Re[i] = _segment("channel", p[i + 1], False, W[i], W[i + 1], W[i + 1],
                               T[i], T[i + 1], T_after[i], dz, A, D_h)
    return p, Re, T_after[-1]


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
def solve_element(power_frac, m_total, geom=None, power_shape="uniform", cold_profile=None,
                  hot=None, N=20, n_r=20, channel=False, pattern="Z", pressure_ref="outlet",
                  x0=None):
    """docs/model.md sections 2-3. Returns a result dict."""
    if pattern == "U":
        raise NotImplementedError('pattern="U" is Tier 2 and not implemented yet')
    if pattern != "Z":
        raise ValueError(f'pattern must be "Z" or "U" (got {pattern!r})')
    if pressure_ref not in ("outlet", "inlet"):
        raise ValueError(f'pressure_ref must be "outlet" or "inlet" (got {pressure_ref!r})')

    t0 = time.perf_counter()
    el = _element(geom)
    dz = el["L"] / N
    m_avg = m_total / N
    s, q, colds, hot = _slice_inputs(el, power_frac, power_shape, cold_profile, hot, N)
    n_slice_calls = 0
    last_x = None                                    # last trial point evaluated

    def one(i, m_i, p_in_i):
        nonlocal n_slice_calls
        if not m_i > 0.0:
            raise _Failure("nonpositive_flow",
                           "no solution found (model cannot represent reverse flow): "
                           f"slice {i} at m = {m_i:.6g} kg/s", i)
        n_slice_calls += 1
        r = slice_dp(m_i, q[i], p_in_i, el["T_in"], colds[i], hot, dz, n_r, geom=geom)
        for flag in STOP_FLAGS:
            if r[flag]:
                raise _Failure(flag, f"slice {i} returned {flag} at m = {m_i:.6g} kg/s, "
                                     f"p_in = {p_in_i:.6g} Pa", i)
        return r

    def unpack(x):
        nonlocal last_x
        last_x = np.array(x, dtype=float)
        m = x[:N] * m_avg
        if pressure_ref == "outlet":
            p_out = el["p_ref"]
            return m, p_out + x[N] * P_SCALE, p_out
        p_in = el["p_ref"]
        return m, p_in, p_in - x[N] * P_SCALE

    def evaluate(x):
        m, p_in, p_out = unpack(x)
        if channel:
            p_pl, re_pl = _plenum(el, m, p_in, dz)
        else:
            p_pl, re_pl = np.full(N, p_in), None
        slices = [one(i, m[i], p_pl[i]) for i in range(N)]
        if channel:
            h_out = np.array([float(pr.h(r["T_gas_out"])) for r in slices])
            p_ch, re_ch, T_mix = _channel(el, m, h_out, p_out, dz)
        else:
            p_ch, re_ch, T_mix = np.full(N, p_out), None, None
        dp = np.array([r["dp"] for r in slices])
        R = np.empty(N + 1)
        R[:N] = (p_pl - p_ch - dp) / P_SCALE
        R[N] = (m.sum() - m_total) / m_total
        return dict(R=R, m=m, p_in=p_in, p_out=p_out, p_pl=p_pl, p_ch=p_ch,
                    re_pl=re_pl, re_ch=re_ch, T_mix=T_mix, slices=slices)

    def residual(x):
        return evaluate(x)["R"]

    def block_jacobian(x):
        # channel=False only: R_i depends on x_i and x_N, R_N on the x_i.
        m, p_in, _ = unpack(x)
        dp0 = np.array([one(i, m[i], p_in)["dp"] for i in range(N)])
        hx = FD_REL_STEP * np.maximum(np.abs(x[:N]), 1.0)
        dp_m = np.array([one(i, m[i] + hx[i] * m_avg, p_in)["dp"] for i in range(N)])
        hp = FD_REL_STEP * max(abs(x[N]), 1.0)
        xp = np.array(x, dtype=float)
        xp[N] += hp
        p_in_p = unpack(xp)[1]
        dp_p = dp0 if p_in_p == p_in else np.array([one(i, m[i], p_in_p)["dp"] for i in range(N)])
        J = np.zeros((N + 1, N + 1))
        J[np.arange(N), np.arange(N)] = -(dp_m - dp0) / hx / P_SCALE
        J[:N, N] = 1.0 - (dp_p - dp0) / hp / P_SCALE
        J[N, :N] = m_avg / m_total
        return J

    def result(converged, reason, message, x, ev=None, nfev=0):
        sl = ev["slices"] if ev else None
        def col(key):
            return np.array([r[key] for r in sl]) if sl else np.full(N, np.nan)
        m, p_in, p_out = unpack(x)
        if ev:
            h_out = np.array([float(pr.h(r["T_gas_out"])) for r in sl])
            T_mix = ev["T_mix"] if ev["T_mix"] is not None else float(pr.T_from_h((m * h_out).sum() / m.sum()))
        else:
            T_mix = np.nan
        re_ch = ev["re_ch"] if ev else None
        re_pl = ev["re_pl"] if ev else None
        manifold_re_low = bool(channel and ev and (min(re_ch.min(), re_pl.min()) < RE_MIN_PETUKHOV))
        return dict(
            converged=converged, reason=reason, message=message,
            m=m, p_in=p_in, p_out=p_out, dp_element=p_in - p_out,
            p_pl=ev["p_pl"] if ev else None, p_ch=ev["p_ch"] if ev else None,
            T_mix_out=T_mix, re_channel=re_ch, re_plenum=re_pl, manifold_re_low=manifold_re_low,
            q=q, s=s, dp=col("dp"), T_gas_out=col("T_gas_out"), T_fuel_max=col("T_fuel_max"),
            dp_cold_frit=col("dp_cold_frit"), dp_bed=col("dp_bed"), dp_hot_frit=col("dp_hot_frit"),
            flags={k: (np.array([r[k] for r in sl]) if sl else np.zeros(N, bool)) for k in FLAG_KEYS},
            residual_norm=float(np.linalg.norm(ev["R"])) if ev else np.nan,
            nfev=nfev, n_slice_calls=n_slice_calls, wall_time=time.perf_counter() - t0, x=x,
            inputs=dict(power_frac=power_frac, m_total=m_total, geom=geom, power_shape=power_shape,
                        cold_profile=cold_profile, hot=hot, N=N, n_r=n_r, channel=channel,
                        pattern=pattern, pressure_ref=pressure_ref),
        )

    try:
        if x0 is None:
            # Equal flows; pressure difference from the mean slice drop at equal flows,
            # each slice entered at the reference pressure.
            x0 = np.ones(N + 1)
            x0[N] = np.mean([one(i, m_avg, el["p_ref"])["dp"] for i in range(N)]) / P_SCALE
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (N + 1,):
            raise ValueError(f"x0 must have length N + 1 = {N + 1}")

        if channel:
            sol = root(residual, x0, method="hybr",
                       options=dict(eps=FD_REL_STEP**2))         # full FD Jacobian
        else:
            sol = root(residual, x0, jac=block_jacobian, method="hybr")
        ev = evaluate(sol.x)
    except _Failure as f:
        x_fail = last_x if last_x is not None else np.ones(N + 1)
        return result(False, f.reason, f.message, x_fail)

    return result(bool(sol.success), None if sol.success else "solver", sol.message,
                  sol.x, ev, sol.nfev)
