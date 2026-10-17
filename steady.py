"""
steady.py -- full-element steady state, docs/model.md sections 3-4 and
Amendment A1 (3a continuation, 3b element inputs, 3c pressure reference,
3d power multipliers).

Only channel=False is implemented so far: every slice sees the same inlet and
outlet pressure, so all slices share one pressure drop.

Unknowns (scaled):  x_i = m_i / m_avg (i = 0..N-1),  x_N = dp_element / P_SCALE
                    outlet reference: p_in = p_ref + dp_element, p_out = p_ref
                    inlet reference:  p_in = p_ref,              p_out = p_ref - dp_element
Residuals:          R_i = (dp_element - dp_slice_i) / P_SCALE
                    R_N = (sum m_i - m_total) / m_total
Solved with scipy.optimize.root(method="hybr").

A slice that returns out_of_range or pressure_collapse, or a nonpositive flow,
at any point the solver visits, ends that attempt. Whether the case has "no
solution" is decided by the continuation rule (3a), never by one attempt.
"""
import time

import numpy as np
from scipy.optimize import root

import params as P
from slice import slice_dp

P_SCALE = 5.0e5        # Pa, unknown and residual scale (docs/model.md section 3)
N_SHAPE_AVG = 50       # points per slice for the slice-averaged power shape (section 3)
FD_REL_STEP = 1.0e-6   # relative finite-difference step for the Jacobian
LAMBDA_STEP0 = 0.25    # continuation: first homotopy step (3a)
LAMBDA_STEP_MIN = 1.0 / 256.0
STOP_FLAGS = ("out_of_range", "pressure_collapse")
FLAG_KEYS = ("re_m_low", "re_ht_low", "out_of_range", "pressure_collapse", "fuel_over_limit")


class _SliceFailure(Exception):
    def __init__(self, reason, slice_index, m_dot, p_in):
        super().__init__(reason)
        self.reason, self.slice_index, self.m_dot, self.p_in = reason, slice_index, m_dot, p_in


# ----------------------------------------------------------------------------
# Element inputs (docs/model.md 3b)
# ----------------------------------------------------------------------------
def _element(geom):
    """Length, full power, inlet temperature, reference pressure, published frits."""
    if geom is None:
        return dict(L=P.L_ELEMENT, power=P.total_power(1.0), T_in=P.T_INLET, p_ref=P.P_OUTLET,
                    cold=(P.COLD_FRIT_PORE, P.COLD_FRIT_EPS), hot=(P.HOT_FRIT_PORE, P.HOT_FRIT_EPS))
    needed = ("L", "power_density", "T_inlet", "P", "r_hot_chan", "t_hot_frit", "t_bed",
              "cold_frit_pore", "cold_frit_eps", "hot_frit_pore", "hot_frit_eps")
    missing = [k for k in needed if k not in geom]
    if missing:
        raise NotImplementedError(f"geom is missing {missing}; only element dicts like "
                                  "params.T19_ELEMENT are decided in docs/model.md (3b)")
    r_hf = geom["r_hot_chan"] + geom["t_hot_frit"]
    r_bed = r_hf + geom["t_bed"]
    power = geom["power_density"] * np.pi * (r_bed**2 - r_hf**2) * geom["L"]
    return dict(L=geom["L"], power=power, T_in=geom["T_inlet"], p_ref=geom["P"],
                cold=(geom["cold_frit_pore"], geom["cold_frit_eps"]),
                hot=(geom["hot_frit_pore"], geom["hot_frit_eps"]))


def _shape(power_shape, N):
    """Slice-averaged shape, 50 sub-interval midpoints per slice, renormalized to mean 1."""
    sub = (np.arange(N)[:, None] + (np.arange(N_SHAPE_AVG) + 0.5)[None, :] / N_SHAPE_AVG) / N
    F = P.power_shape(power_shape, sub).mean(axis=1)
    return F / F.mean()


def _slice_frits(el, cold_profile, hot, N):
    s = (np.arange(N) + 0.5) / N                     # slice centres, z_i / L
    if cold_profile is None:
        colds = [el["cold"]] * N                     # this element's published cold frit
    elif isinstance(cold_profile, str) and cold_profile == "none":
        colds = [None] * N
    elif callable(cold_profile):
        colds = [tuple(cold_profile(si)) for si in s]
    else:
        raise ValueError('cold_profile must be None, "none", or a callable s -> (pore, porosity)')
    hot = el["hot"] if hot is None else tuple(hot)
    return s, colds, hot


# ----------------------------------------------------------------------------
# One solve at a fixed homotopy level lambda
# ----------------------------------------------------------------------------
def _solve_once(lam, x0, power_frac, m_total, el, F, mult, colds, hot, s, geom, N, n_r,
                pressure_ref, inputs, t0):
    dz = el["L"] / N
    m_avg = m_total / N
    F_lam = 1.0 + lam * (F - 1.0)
    mult_lam = 1.0 + lam * (mult - 1.0)
    q = el["power"] * power_frac * F_lam * mult_lam / N
    outlet_ref = pressure_ref == "outlet"
    n_slice_calls = 0
    last_x = None

    def one(i, m_i, p_in):
        nonlocal n_slice_calls
        if not m_i > 0.0:
            raise _SliceFailure("nonpositive_flow", i, m_i, p_in)
        n_slice_calls += 1
        r = slice_dp(m_i, q[i], p_in, el["T_in"], colds[i], hot, dz, n_r, geom=geom)
        for flag in STOP_FLAGS:
            if r[flag]:
                raise _SliceFailure(flag, i, m_i, p_in)
        return r

    def unpack(x):
        nonlocal last_x
        last_x = np.array(x, dtype=float)
        dp_el = x[N] * P_SCALE
        p_in = el["p_ref"] + dp_el if outlet_ref else el["p_ref"]
        return x[:N] * m_avg, p_in, dp_el

    def residual(x):
        m, p_in, dp_el = unpack(x)
        dp = np.array([one(i, m[i], p_in)["dp"] for i in range(N)])
        R = np.empty(N + 1)
        R[:N] = (dp_el - dp) / P_SCALE
        R[N] = (m.sum() - m_total) / m_total
        return R

    def jacobian(x):
        # R_i depends only on x_i and x_N; R_N only on the x_i.
        m, p_in, _ = unpack(x)
        dp0 = np.array([one(i, m[i], p_in)["dp"] for i in range(N)])
        hx = FD_REL_STEP * np.maximum(np.abs(x[:N]), 1.0)
        dp_m = np.array([one(i, m[i] + hx[i] * m_avg, p_in)["dp"] for i in range(N)])
        J = np.zeros((N + 1, N + 1))
        J[np.arange(N), np.arange(N)] = -(dp_m - dp0) / hx / P_SCALE
        if outlet_ref:   # slice inlet pressure moves with x_N
            hp = FD_REL_STEP * max(abs(x[N]), 1.0)
            dp_p = np.array([one(i, m[i], p_in + hp * P_SCALE)["dp"] for i in range(N)])
            J[:N, N] = 1.0 - (dp_p - dp0) / hp / P_SCALE
        else:            # slice inlet pressure fixed at p_ref
            J[:N, N] = 1.0
        J[N, :N] = m_avg / m_total
        return J

    def result(converged, reason, message, x, slices=None, residual_norm=np.nan, nfev=0, J=None):
        m, p_in, dp_el = unpack(x)
        def col(key):
            return np.array([r[key] for r in slices]) if slices else np.full(N, np.nan)
        ddp_dm = (-np.diag(J)[:N] * P_SCALE / m_avg) if J is not None else np.full(N, np.nan)
        return dict(
            converged=converged, reason=reason, message=message,
            m=m, p_in=p_in, p_out=p_in - dp_el, dp_element=dp_el,
            q=q, s=s, dp=col("dp"), T_gas_out=col("T_gas_out"), T_fuel_max=col("T_fuel_max"),
            dp_cold_frit=col("dp_cold_frit"), dp_bed=col("dp_bed"), dp_hot_frit=col("dp_hot_frit"),
            flags={k: (np.array([r[k] for r in slices]) if slices else np.zeros(N, bool))
                   for k in FLAG_KEYS},
            jacobian=J, ddp_dm=ddp_dm,
            slice_inputs=dict(q=q, colds=colds, hot=hot, dz=dz, n_r=n_r, T_in=el["T_in"],
                              p_in=p_in, geom=geom, m_avg=m_avg),
            residual_norm=residual_norm, nfev=nfev, n_slice_calls=n_slice_calls,
            wall_time=time.perf_counter() - t0, x=x, lam=lam, inputs=inputs,
        )

    try:
        if x0 is None:
            x0 = np.ones(N + 1)
            x0[N] = np.mean([one(i, m_avg, el["p_ref"])["dp"] for i in range(N)]) / P_SCALE
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (N + 1,):
            raise ValueError(f"x0 must have length N + 1 = {N + 1}")
        sol = root(residual, x0, jac=jacobian, method="hybr")
        m, p_in, _ = unpack(sol.x)
        slices = [one(i, m[i], p_in) for i in range(N)]
        R = residual(sol.x)
        J = jacobian(sol.x) if sol.success else None
    except _SliceFailure as f:
        msg = (f"slice {f.slice_index} returned {f.reason} at m = {f.m_dot:.6g} kg/s, "
               f"p_in = {f.p_in:.6g} Pa")
        x_fail = last_x if last_x is not None else np.r_[np.ones(N), 0.0]
        return result(False, f.reason, msg, x_fail)

    return result(bool(sol.success), None if sol.success else "solver", sol.message, sol.x,
                  slices, float(np.linalg.norm(R)), sol.nfev, J)


# ----------------------------------------------------------------------------
# Public solver with the continuation rule (docs/model.md 3a)
# ----------------------------------------------------------------------------
def solve_element(power_frac, m_total, geom=None, power_shape="uniform", cold_profile=None,
                  hot=None, N=20, n_r=20, channel=False, pattern="Z", pressure_ref="outlet",
                  x0=None, power_multipliers=None):
    """Returns a result dict. converged=False means no solution after the full
    continuation rule; "path" says whether the direct solve or continuation was used."""
    if channel:
        raise NotImplementedError("channel=True (docs/model.md section 2) is not implemented yet")
    if pressure_ref not in ("outlet", "inlet"):
        raise ValueError(f'pressure_ref must be "outlet" or "inlet" (got {pressure_ref!r})')
    if pattern not in ("Z", "U"):
        raise ValueError(f'pattern must be "Z" or "U" (got {pattern!r})')

    t0 = time.perf_counter()
    el = _element(geom)
    F = _shape(power_shape, N)
    mult = np.ones(N) if power_multipliers is None else np.asarray(power_multipliers, dtype=float)
    if mult.shape != (N,):
        raise ValueError(f"power_multipliers must have length N = {N}")
    s, colds, hot = _slice_frits(el, cold_profile, hot, N)
    inputs = dict(power_frac=power_frac, m_total=m_total, power_shape=power_shape,
                  cold_profile=cold_profile, hot=hot, N=N, n_r=n_r, channel=channel,
                  pattern=pattern, pressure_ref=pressure_ref,
                  power_multipliers=None if power_multipliers is None else mult.tolist())
    common = (power_frac, m_total, el, F, mult, colds, hot, s, geom, N, n_r, pressure_ref, inputs, t0)

    # Step 1: direct solve at the target (lambda = 1)
    res = _solve_once(1.0, x0, *common)
    if res["converged"]:
        res.update(path="direct", final_step=None)
        return res

    # Step 2: continuation from equal slice power (lambda = 0), halving on failure
    last = _solve_once(0.0, None, *common)
    if not last["converged"]:
        last.update(path="continuation", final_step=None,
                    message="no solution found (continuation could not start at lambda = 0): "
                            + str(last["message"]))
        return last
    lam, step, failed = 0.0, LAMBDA_STEP0, None
    while lam < 1.0:
        trial = min(1.0, lam + step)
        r = _solve_once(trial, last["x"], *common)
        if r["converged"]:
            lam, last = trial, r
        else:
            failed = r
            step /= 2.0
            if step < LAMBDA_STEP_MIN:
                out = dict(failed)
                out.update(converged=False, path="continuation", final_step=step,
                           message=(f"no solution found (continuation exhausted at lambda = "
                                    f"{lam:.4f}): {failed['message']}"),
                           last_converged=dict(lam=lam, m=last["m"],
                                               T_fuel_max=last["T_fuel_max"],
                                               dp_element=last["dp_element"]))
                return out
    last.update(path="continuation", final_step=step)
    return last