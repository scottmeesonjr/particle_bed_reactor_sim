"""
steady.py -- full-element steady state, docs/model.md sections 3-4.

Only channel=False is implemented so far (section 3): every slice sees
p_ch = P_OUTLET and p_pl = p_in, so all slices share one pressure drop.

Unknowns (scaled):  x_i = m_i / m_avg (i = 0..N-1),  x_N = (p_in - P_OUTLET) / P_SCALE
Residuals:          R_i = (p_pl - p_ch - dp_slice_i) / P_SCALE
                    R_N = (sum m_i - m_total) / m_total
Solved with scipy.optimize.root(method="hybr").

A slice that returns out_of_range or pressure_collapse, at the solution or at
any trial point the solver visits, stops the solve with converged=False and
that reason. Nothing is extrapolated.
"""
import time

import numpy as np
from scipy.optimize import root

import params as P
from slice import slice_dp

P_SCALE = 5.0e5        # Pa, unknown and residual scale (docs/model.md section 3)
N_SHAPE_AVG = 50       # points per slice for the slice-averaged power shape (section 3)
FD_REL_STEP = 1.0e-6   # relative finite-difference step for the Jacobian
STOP_FLAGS = ("out_of_range", "pressure_collapse")
FLAG_KEYS = ("re_m_low", "re_ht_low", "out_of_range", "pressure_collapse", "fuel_over_limit")


class _SliceFailure(Exception):
    def __init__(self, reason, slice_index, m_dot, p_in):
        super().__init__(reason)
        self.reason, self.slice_index, self.m_dot, self.p_in = reason, slice_index, m_dot, p_in


# ----------------------------------------------------------------------------
# Per-slice inputs: power, cold frit, hot frit
# ----------------------------------------------------------------------------
def _slice_inputs(power_frac, power_shape, cold_profile, hot, N):
    s = (np.arange(N) + 0.5) / N                     # slice centres, z_i / L
    # Slice-averaged shape: 50 sub-interval midpoints per slice (docs/model.md section 3)
    sub = (np.arange(N)[:, None] + (np.arange(N_SHAPE_AVG) + 0.5)[None, :] / N_SHAPE_AVG) / N
    F = P.power_shape(power_shape, sub).mean(axis=1)
    F = F / F.mean()                                 # discrete mean exactly 1
    q = P.total_power(power_frac) * F / N

    if cold_profile is None:
        colds = [(P.COLD_FRIT_PORE, P.COLD_FRIT_EPS)] * N
    elif isinstance(cold_profile, str) and cold_profile == "none":
        colds = [None] * N
    elif callable(cold_profile):
        colds = [tuple(cold_profile(si)) for si in s]
    else:
        raise ValueError('cold_profile must be None, "none", or a callable s -> (pore, porosity)')

    hot = (P.HOT_FRIT_PORE, P.HOT_FRIT_EPS) if hot is None else tuple(hot)
    return s, q, colds, hot


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
def solve_element(power_frac, m_total, geom=None, power_shape="uniform", cold_profile=None,
                  hot=None, N=20, n_r=20, channel=False, pattern="Z", pressure_ref="outlet",
                  x0=None):
    """docs/model.md section 3, channel=False only. Returns a result dict."""
    if geom is not None:
        raise NotImplementedError("geom != None: element power and length for non-nominal "
                                  "geometries are not yet decided in docs/model.md")
    if channel:
        raise NotImplementedError("channel=True (docs/model.md section 2) is not implemented yet")
    if pressure_ref != "outlet":
        raise NotImplementedError('pressure_ref="inlet" is not yet decided in docs/model.md')
    if pattern not in ("Z", "U"):
        raise ValueError(f'pattern must be "Z" or "U" (got {pattern!r})')

    t0 = time.perf_counter()
    dz = P.L_ELEMENT / N
    m_avg = m_total / N
    s, q, colds, hot = _slice_inputs(power_frac, power_shape, cold_profile, hot, N)
    n_slice_calls = 0
    last_x = None                                    # last trial point evaluated

    def one(i, m_i, p_in):
        nonlocal n_slice_calls
        if not m_i > 0.0:
            raise _SliceFailure("nonpositive_flow", i, m_i, p_in)
        n_slice_calls += 1
        r = slice_dp(m_i, q[i], p_in, P.T_INLET, colds[i], hot, dz, n_r)
        for flag in STOP_FLAGS:
            if r[flag]:
                raise _SliceFailure(flag, i, m_i, p_in)
        return r

    def unpack(x):
        nonlocal last_x
        last_x = np.array(x, dtype=float)
        return x[:N] * m_avg, P.P_OUTLET + x[N] * P_SCALE

    def residual(x):
        m, p_in = unpack(x)
        dp = np.array([one(i, m[i], p_in)["dp"] for i in range(N)])
        R = np.empty(N + 1)
        R[:N] = (p_in - P.P_OUTLET - dp) / P_SCALE
        R[N] = (m.sum() - m_total) / m_total
        return R

    def jacobian(x):
        # With channel=False, R_i depends only on x_i and x_N, and R_N only
        # on the x_i. Finite differences on that structure: 3N slice calls.
        m, p_in = unpack(x)
        dp0 = np.array([one(i, m[i], p_in)["dp"] for i in range(N)])
        hx = FD_REL_STEP * np.maximum(np.abs(x[:N]), 1.0)
        dp_m = np.array([one(i, m[i] + hx[i] * m_avg, p_in)["dp"] for i in range(N)])
        hp = FD_REL_STEP * max(abs(x[N]), 1.0)
        dp_p = np.array([one(i, m[i], p_in + hp * P_SCALE)["dp"] for i in range(N)])
        J = np.zeros((N + 1, N + 1))
        J[np.arange(N), np.arange(N)] = -(dp_m - dp0) / hx / P_SCALE
        J[:N, N] = 1.0 - (dp_p - dp0) / hp / P_SCALE
        J[N, :N] = m_avg / m_total
        return J

    def result(converged, reason, message, x, slices=None, residual_norm=np.nan, nfev=0):
        m, p_in = unpack(x)
        def col(key):
            return np.array([r[key] for r in slices]) if slices else np.full(N, np.nan)
        out = dict(
            converged=converged, reason=reason, message=message,
            m=m, p_in=p_in, dp_element=p_in - P.P_OUTLET,
            q=q, s=s, dp=col("dp"), T_gas_out=col("T_gas_out"), T_fuel_max=col("T_fuel_max"),
            dp_cold_frit=col("dp_cold_frit"), dp_bed=col("dp_bed"), dp_hot_frit=col("dp_hot_frit"),
            flags={k: (np.array([r[k] for r in slices]) if slices else np.zeros(N, bool))
                   for k in FLAG_KEYS},
            residual_norm=residual_norm, nfev=nfev, n_slice_calls=n_slice_calls,
            wall_time=time.perf_counter() - t0, x=x,
            inputs=dict(power_frac=power_frac, m_total=m_total, power_shape=power_shape,
                        cold_profile=cold_profile, hot=hot, N=N, n_r=n_r, channel=channel,
                        pattern=pattern, pressure_ref=pressure_ref),
        )
        return out

    try:
        if x0 is None:
            # Equal flows; p_in from the mean slice drop at equal flows, p_in = P_OUTLET.
            x0 = np.ones(N + 1)
            dp_guess = np.mean([one(i, m_avg, P.P_OUTLET)["dp"] for i in range(N)])
            x0[N] = dp_guess / P_SCALE
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (N + 1,):
            raise ValueError(f"x0 must have length N + 1 = {N + 1}")

        sol = root(residual, x0, jac=jacobian, method="hybr")
        m, p_in = unpack(sol.x)
        slices = [one(i, m[i], p_in) for i in range(N)]
        R = residual(sol.x)
    except _SliceFailure as f:
        msg = (f"slice {f.slice_index} returned {f.reason} at m = {f.m_dot:.6g} kg/s, "
               f"p_in = {f.p_in:.6g} Pa")
        if f.reason == "nonpositive_flow":
            msg = "no solution found (model cannot represent reverse flow): " + msg
        x_fail = last_x if last_x is not None else np.ones(N + 1)
        if x0 is None:                               # failed during the initial guess
            x_fail = np.ones(N + 1)
            x_fail[N] = 0.0
        return result(False, f.reason, msg, x_fail)

    norm = float(np.linalg.norm(R))
    reason = None if sol.success else "solver"
    return result(bool(sol.success), reason, sol.message, sol.x, slices, norm, sol.nfev)
