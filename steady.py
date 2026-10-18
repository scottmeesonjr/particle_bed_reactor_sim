"""
steady.py -- full-element steady state, docs/model.md sections 2-4, Amendment A1
(3a continuation, 3b element inputs, 3c pressure reference, 3d power multipliers)
and Amendment A2 (channel and plenum details).

Unknowns (scaled):  x_i = m_i / m_avg (i = 0..N-1),  x_N = dp_element / P_SCALE
    dp_element = p0 - p_exit, with p0 the plenum inlet pressure (z = 0) and p_exit the
    channel exit pressure (z = L).
    outlet reference: p_exit = p_ref, p0 = p_ref + dp_element
    inlet reference:  p0 = p_ref,     p_exit = p_ref - dp_element
Residuals:  R_i = (p_pl(z_i) - p_ch(z_i) - dp_slice_i) / P_SCALE
            R_N = (sum m_i - m_total) / m_total
channel=False: p_pl = p0 and p_ch = p_exit at every slice (all slices share one drop).
channel=True (Z pattern): plenum fed at z = 0, channel exits at z = L, both flowing upward.

A slice that returns out_of_range or pressure_collapse, a manifold state outside the
property range, or a nonpositive flow, at any point the solver visits, ends that attempt.
Whether the case has "no solution" is decided by the continuation rule (3a).
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
FD_REL_STEP = 1.0e-6   # relative finite-difference step for the Jacobian
LAMBDA_STEP0 = 0.25    # continuation: first homotopy step (3a)
LAMBDA_STEP_MIN = 1.0 / 256.0
RE_PETUKHOV_MIN = P.RE_MIN_PETUKHOV   # A2: Petukhov used at max(Re, 3000), floored by laminar 64/Re
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
    """Length, power, inlet temperature, reference pressure, frits, manifold radii."""
    if geom is None:
        return dict(L=P.L_ELEMENT, power=P.total_power(1.0), T_in=P.T_INLET, p_ref=P.P_OUTLET,
                    cold=(P.COLD_FRIT_PORE, P.COLD_FRIT_EPS), hot=(P.HOT_FRIT_PORE, P.HOT_FRIT_EPS),
                    r_ch=P.R_HOT_CHAN, r_cf=P.R_COLD_FRIT_OUT, r_el=P.R_ELEMENT)
    needed = ("L", "power_density", "T_inlet", "P", "r_hot_chan", "t_hot_frit", "t_bed",
              "t_cold_frit", "t_plenum", "cold_frit_pore", "cold_frit_eps", "hot_frit_pore",
              "hot_frit_eps")
    missing = [k for k in needed if k not in geom]
    if missing:
        raise NotImplementedError(f"geom is missing {missing}; only element dicts like "
                                  "params.T19_ELEMENT are decided in docs/model.md (3b)")
    r_hf = geom["r_hot_chan"] + geom["t_hot_frit"]
    r_bed = r_hf + geom["t_bed"]
    r_cf = r_bed + geom["t_cold_frit"]
    power = geom["power_density"] * np.pi * (r_bed**2 - r_hf**2) * geom["L"]
    return dict(L=geom["L"], power=power, T_in=geom["T_inlet"], p_ref=geom["P"],
                cold=(geom["cold_frit_pore"], geom["cold_frit_eps"]),
                hot=(geom["hot_frit_pore"], geom["hot_frit_eps"]),
                r_ch=geom["r_hot_chan"], r_cf=r_cf, r_el=r_cf + geom["t_plenum"])


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


def manifold_setup(el, N, channel, pressure_ref):
    """Pure-data description of the plenum and channel (picklable)."""
    return dict(
        N=N, dz=el["L"] / N, T_in=el["T_in"], channel=bool(channel), pressure_ref=pressure_ref,
        p_ref=el["p_ref"],
        A_ch=math.pi * el["r_ch"] ** 2, Dh_ch=2.0 * el["r_ch"],
        A_pl=math.pi * (el["r_el"] ** 2 - el["r_cf"] ** 2), Dh_pl=2.0 * (el["r_el"] - el["r_cf"]),
    )


# ----------------------------------------------------------------------------
# Plenum and channel (docs/model.md section 2, Amendment A2)
# ----------------------------------------------------------------------------
def darcy_friction(Re):
    """A2: Petukhov at max(Re, 3000), never below laminar 64/Re."""
    pet = (0.790 * math.log(max(Re, RE_PETUKHOV_MIN)) - 1.64) ** -2
    return max(64.0 / Re, pet)


def _friction_dp(W, rho, mu, A, Dh, length):
    if W <= 0.0 or length <= 0.0:
        return 0.0
    Re = W * Dh / (A * mu)
    return darcy_friction(Re) * W * W / (2.0 * rho * A * A * Dh) * length


def plenum_pressures(m, p0, st, friction=True):
    """Plenum pressure at each slice centre. Fed at z = 0 with sum(m); flow at a slice
    centre is taken halfway through that slice's draw (A2)."""
    m = np.asarray(m, dtype=float)
    N = len(m)
    if not st["channel"]:
        return np.full(N, float(p0))
    A, Dh, dz, T = st["A_pl"], st["Dh_pl"], st["dz"], st["T_in"]
    mu = float(pr.mu(T))
    W_tot = float(m.sum())
    W = W_tot - (np.cumsum(m) - 0.5 * m)
    p = np.empty(N)
    p_prev, W_prev = float(p0), W_tot
    rho_prev = float(pr.rho(T, p_prev))
    for i in range(N):
        seg = 0.5 * dz if i == 0 else dz
        rho_i = float(pr.rho(T, p_prev))
        mom = (W[i] ** 2 / rho_i - W_prev ** 2 / rho_prev) / A ** 2
        fr = _friction_dp(0.5 * (W_prev + W[i]), 0.5 * (rho_prev + rho_i), mu, A, Dh, seg) if friction else 0.0
        p[i] = p_prev - mom - fr
        p_prev, W_prev, rho_prev = p[i], W[i], rho_i
    return p


def channel_pressures(m, T_out, p_exit, st, friction=True):
    """Channel pressure at each slice centre, marched down from p_exit at z = L.
    Channel gas is the enthalpy-weighted mix of slice outflows below; flow at a slice
    centre is taken halfway through that slice's injection (A2). Entering gas carries
    no axial momentum."""
    m = np.asarray(m, dtype=float)
    N = len(m)
    if not st["channel"]:
        return np.full(N, float(p_exit))
    A, Dh, dz = st["A_ch"], st["Dh_ch"], st["dz"]
    hs = np.asarray(pr.h(np.asarray(T_out, dtype=float)), dtype=float)
    W = np.cumsum(m) - 0.5 * m
    T_c = np.asarray(pr.T_from_h((np.cumsum(m * hs) - 0.5 * m * hs) / W), dtype=float)
    W_up = float(m.sum())
    T_up = float(pr.T_from_h(np.sum(m * hs) / W_up))
    p_up = float(p_exit)
    rho_up = float(pr.rho(T_up, p_up))
    p = np.empty(N)
    for i in range(N - 1, -1, -1):
        seg = 0.5 * dz if i == N - 1 else dz
        rho_i = float(pr.rho(T_c[i], p_up))
        mom = (W_up ** 2 / rho_up - W[i] ** 2 / rho_i) / A ** 2
        mu = float(pr.mu(0.5 * (T_up + T_c[i])))
        fr = _friction_dp(0.5 * (W_up + W[i]), 0.5 * (rho_up + rho_i), mu, A, Dh, seg) if friction else 0.0
        p[i] = p_up + mom + fr                      # flow-direction: p_up = p_i - mom - fr
        p_up, W_up, rho_up, T_up = p[i], W[i], rho_i, T_c[i]
    return p


def end_pressures(x_N, st):
    """(p0, p_exit) from the scaled unknown x_N (3c)."""
    dp_el = x_N * P_SCALE
    if st["pressure_ref"] == "outlet":
        return st["p_ref"] + dp_el, st["p_ref"]
    return st["p_ref"], st["p_ref"] - dp_el


def manifold_derivatives(m, T_out, x_N, st, m_avg, rel=FD_REL_STEP):
    """Finite-difference derivatives of the manifold pressures (cheap, no slice calls).
    Returns p_pl, p_ch, dPpl (N, N+1) and dPch_x (N, N+1) w.r.t. the scaled unknowns
    at fixed slice outlet temperatures, and dPch_T (N, N) w.r.t. slice outlet temperatures."""
    m = np.asarray(m, dtype=float); T_out = np.asarray(T_out, dtype=float)
    N = len(m)
    p0, p_exit = end_pressures(x_N, st)
    p_pl = plenum_pressures(m, p0, st)
    p_ch = channel_pressures(m, T_out, p_exit, st)
    dPpl = np.zeros((N, N + 1)); dPch_x = np.zeros((N, N + 1)); dPch_T = np.zeros((N, N))
    if st["channel"]:
        for j in range(N):
            hx = rel * max(abs(m[j] / m_avg), 1.0)
            m2 = m.copy(); m2[j] += hx * m_avg
            dPpl[:, j] = (plenum_pressures(m2, p0, st) - p_pl) / hx
            dPch_x[:, j] = (channel_pressures(m2, T_out, p_exit, st) - p_ch) / hx
            hT = rel * T_out[j]
            T2 = T_out.copy(); T2[j] += hT
            dPch_T[:, j] = (channel_pressures(m, T2, p_exit, st) - p_ch) / hT
    hN = rel * max(abs(x_N), 1.0)
    p0b, p_exitb = end_pressures(x_N + hN, st)
    dPpl[:, N] = (plenum_pressures(m, p0b, st) - p_pl) / hN
    dPch_x[:, N] = (channel_pressures(m, T_out, p_exitb, st) - p_ch) / hN
    return p_pl, p_ch, dPpl, dPch_x, dPch_T


# ----------------------------------------------------------------------------
# One solve at a fixed homotopy level lambda
# ----------------------------------------------------------------------------
def _solve_once(lam, x0, power_frac, m_total, el, F, mult, colds, hot, s, geom, N, n_r,
                st, inputs, t0):
    dz = st["dz"]
    m_avg = m_total / N
    F_lam = 1.0 + lam * (F - 1.0)
    mult_lam = 1.0 + lam * (mult - 1.0)
    q = el["power"] * power_frac * F_lam * mult_lam / N
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

    def manifold(fn, *args):
        try:
            return fn(*args)
        except pr.PropertyRangeError:
            raise _SliceFailure("out_of_range", -1, float("nan"), float("nan"))

    def unpack(x):
        nonlocal last_x
        last_x = np.array(x, dtype=float)
        p0, p_exit = end_pressures(x[N], st)
        return x[:N] * m_avg, p0, p_exit

    def evaluate(x):
        m, p0, p_exit = unpack(x)
        p_pl = manifold(plenum_pressures, m, p0, st)
        slices = [one(i, m[i], p_pl[i]) for i in range(N)]
        T_out = np.array([r["T_gas_out"] for r in slices])
        p_ch = manifold(channel_pressures, m, T_out, p_exit, st)
        dp = np.array([r["dp"] for r in slices])
        R = np.empty(N + 1)
        R[:N] = (p_pl - p_ch - dp) / P_SCALE
        R[N] = (m.sum() - m_total) / m_total
        return R, slices, p_pl, p_ch

    def residual(x):
        return evaluate(x)[0]

    def jacobian(x, return_parts=False):
        # Chain rule: slice sensitivities (3 slice calls each) + cheap manifold derivatives.
        m, p0, p_exit = unpack(x)
        p_pl = manifold(plenum_pressures, m, p0, st)
        base = [one(i, m[i], p_pl[i]) for i in range(N)]
        D0 = np.array([r["dp"] for r in base]); T0 = np.array([r["T_gas_out"] for r in base])
        hx = FD_REL_STEP * np.maximum(np.abs(x[:N]), 1.0)
        hp = FD_REL_STEP * max(abs(x[N]), 1.0) * P_SCALE
        pm = [one(i, m[i] + hx[i] * m_avg, p_pl[i]) for i in range(N)]
        pp = [one(i, m[i], p_pl[i] + hp) for i in range(N)]
        D_x = (np.array([r["dp"] for r in pm]) - D0) / hx           # per unit x_i
        T_x = (np.array([r["T_gas_out"] for r in pm]) - T0) / hx
        D_p = (np.array([r["dp"] for r in pp]) - D0) / hp           # per Pa of inlet pressure
        T_p = (np.array([r["T_gas_out"] for r in pp]) - T0) / hp
        _, _, dPpl, dPch_x, dPch_T = manifold(manifold_derivatives, m, T0, x[N], st, m_avg)
        dD = np.zeros((N, N + 1)); dD[np.arange(N), np.arange(N)] = D_x
        dD += D_p[:, None] * dPpl
        dT = np.zeros((N, N + 1)); dT[np.arange(N), np.arange(N)] = T_x
        dT += T_p[:, None] * dPpl
        J = np.zeros((N + 1, N + 1))
        J[:N] = (dPpl - dPch_x - dPch_T @ dT - dD) / P_SCALE
        J[N, :N] = m_avg / m_total
        if return_parts:
            return J, D_x / m_avg
        return J

    def result(converged, reason, message, x, slices=None, p_pl=None, p_ch=None,
               residual_norm=np.nan, nfev=0, J=None, ddp_dm=None):
        m, p0, p_exit = unpack(x)
        def col(key):
            return np.array([r[key] for r in slices]) if slices else np.full(N, np.nan)
        return dict(
            converged=converged, reason=reason, message=message,
            m=m, p_in=p0, p_out=p_exit, dp_element=p0 - p_exit,
            p_plenum=p_pl if p_pl is not None else np.full(N, np.nan),
            p_channel=p_ch if p_ch is not None else np.full(N, np.nan),
            q=q, s=s, dp=col("dp"), T_gas_out=col("T_gas_out"), T_fuel_max=col("T_fuel_max"),
            dp_cold_frit=col("dp_cold_frit"), dp_bed=col("dp_bed"), dp_hot_frit=col("dp_hot_frit"),
            flags={k: (np.array([r[k] for r in slices]) if slices else np.zeros(N, bool))
                   for k in FLAG_KEYS},
            jacobian=J, ddp_dm=ddp_dm if ddp_dm is not None else np.full(N, np.nan),
            slice_inputs=dict(q=q, colds=colds, hot=hot, dz=dz, n_r=n_r, T_in=el["T_in"],
                              p_in=p0, p_slices=p_pl, p0=p0, p_exit=p_exit, x_N=x[N],
                              geom=geom, m_avg=m_avg, m_total=m_total, setup=st),
            residual_norm=residual_norm, nfev=nfev, n_slice_calls=n_slice_calls,
            wall_time=time.perf_counter() - t0, x=x, lam=lam, inputs=inputs,
        )

    try:
        if x0 is None:
            x0 = np.ones(N + 1)
            x0[N] = np.mean([one(i, m_avg, st["p_ref"])["dp"] for i in range(N)]) / P_SCALE
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (N + 1,):
            raise ValueError(f"x0 must have length N + 1 = {N + 1}")
        sol = root(residual, x0, jac=jacobian, method="hybr")
        R, slices, p_pl, p_ch = evaluate(sol.x)
        J = ddp_dm = None
        if sol.success:
            J, ddp_dm = jacobian(sol.x, return_parts=True)
    except _SliceFailure as f:
        where = "a manifold state" if f.slice_index < 0 else f"slice {f.slice_index}"
        msg = (f"{where} returned {f.reason}" if f.slice_index < 0 else
               f"{where} returned {f.reason} at m = {f.m_dot:.6g} kg/s, p_in = {f.p_in:.6g} Pa")
        x_fail = last_x if last_x is not None else np.r_[np.ones(N), 0.0]
        return result(False, f.reason, msg, x_fail)

    return result(bool(sol.success), None if sol.success else "solver", sol.message, sol.x,
                  slices, p_pl, p_ch, float(np.linalg.norm(R)), sol.nfev, J, ddp_dm)


# ----------------------------------------------------------------------------
# Public solver with the continuation rule (docs/model.md 3a)
# ----------------------------------------------------------------------------
def solve_element(power_frac, m_total, geom=None, power_shape="uniform", cold_profile=None,
                  hot=None, N=20, n_r=20, channel=False, pattern="Z", pressure_ref="outlet",
                  x0=None, power_multipliers=None):
    """Returns a result dict. converged=False means no solution after the full
    continuation rule; "path" says whether the direct solve or continuation was used."""
    if pressure_ref not in ("outlet", "inlet"):
        raise ValueError(f'pressure_ref must be "outlet" or "inlet" (got {pressure_ref!r})')
    if pattern not in ("Z", "U"):
        raise ValueError(f'pattern must be "Z" or "U" (got {pattern!r})')
    if channel and pattern == "U":
        raise NotImplementedError("The U pattern is Tier 2 and not built (docs/model.md A2)")

    t0 = time.perf_counter()
    el = _element(geom)
    F = _shape(power_shape, N)
    mult = np.ones(N) if power_multipliers is None else np.asarray(power_multipliers, dtype=float)
    if mult.shape != (N,):
        raise ValueError(f"power_multipliers must have length N = {N}")
    s, colds, hot = _slice_frits(el, cold_profile, hot, N)
    st = manifold_setup(el, N, channel, pressure_ref)
    inputs = dict(power_frac=power_frac, m_total=m_total, power_shape=power_shape,
                  cold_profile=cold_profile, hot=hot, N=N, n_r=n_r, channel=channel,
                  pattern=pattern, pressure_ref=pressure_ref,
                  power_multipliers=None if power_multipliers is None else mult.tolist())
    common = (power_frac, m_total, el, F, mult, colds, hot, s, geom, N, n_r, st, inputs, t0)

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
