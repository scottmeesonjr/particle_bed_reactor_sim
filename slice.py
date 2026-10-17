"""
slice.py -- one axial slice of the fuel element: radial march from the cold
frit outer face to the hot channel. Implements docs/model.md section 1 exactly.

Layers, marched inward with s = distance from the cold frit outer face:
  1. cold frit  R_COLD_FRIT_OUT -> R_BED_OUT       full Ergun, adiabatic
  2. bed        R_BED_OUT       -> R_HOT_FRIT_OUT  KTA 3102.3 dp, heat release,
                                                   KTA 3102.2 fuel temperature
  3. hot frit   R_HOT_FRIT_OUT  -> R_HOT_CHAN      full Ergun, adiabatic
State = (p, h). Each layer uses n_r uniform midpoint (RK2) steps in r, with
properties evaluated at the midpoint.

Geometry: geom=None is the nominal element; otherwise a geometry dict from
params.py (params.T19_ELEMENT, params.T19_LARGE_REACTOR), read by its keys.

Limitations (from the spec): particle internal conduction neglected; frits
adiabatic.
"""
import math

import params as P
import properties as pr
from properties import PropertyRangeError


class _PressureCollapse(Exception):
    """Pressure reached 0 or below during the march."""


# ----------------------------------------------------------------------------
# Geometry (values only from params.py)
# ----------------------------------------------------------------------------
_GEOM_KEYS = ("r_hot_chan", "t_hot_frit", "t_bed", "t_cold_frit", "d_particle", "eps_bed")


def _geometry(geom):
    """Radii (channel, hot frit outer, bed outer, cold frit outer), d_p, alpha."""
    if geom is None:
        return (P.R_HOT_CHAN, P.R_HOT_FRIT_OUT, P.R_BED_OUT, P.R_COLD_FRIT_OUT,
                P.D_PARTICLE, P.EPS_BED)
    missing = [k for k in _GEOM_KEYS if k not in geom]
    if missing:
        raise ValueError(f"geom is missing keys {missing}; pass a params.py geometry dict")
    r_ch = geom["r_hot_chan"]
    r_hf = r_ch + geom["t_hot_frit"]
    r_bed = r_hf + geom["t_bed"]
    r_cf = r_bed + geom["t_cold_frit"]
    return r_ch, r_hf, r_bed, r_cf, geom["d_particle"], geom["eps_bed"]


# ----------------------------------------------------------------------------
# Correlations (pressure drops are positive = pressure falls along s)
# ----------------------------------------------------------------------------
def _ergun_dpds(G, rho, mu, d, eps):
    """Full Ergun, both terms, d = pore size, eps = frit porosity. Pa/m."""
    u = G / rho
    return (150.0 * mu * (1.0 - eps)**2 * u / (eps**3 * d**2)
            + 1.75 * rho * (1.0 - eps) * u**2 / (eps**3 * d))


def _kta_dpds(G, rho, mu, d_p, alpha):
    """KTA 3102.3 bed pressure gradient. Returns (dp/ds in Pa/m, Re)."""
    Re = G * d_p / mu
    Re_m = Re / (1.0 - alpha)
    f = 320.0 / Re_m + 6.0 / Re_m**0.1
    return f * (1.0 - alpha) / alpha**3 * G**2 / (2.0 * rho * d_p), Re


def _kta_nu(Re, Pr, alpha):
    """KTA 3102.2 particle-to-gas Nusselt number."""
    return (1.27 * Pr**(1.0 / 3.0) * Re**0.36 / alpha**1.18
            + 0.033 * Pr**0.5 * Re**0.86 / alpha**1.07)


# ----------------------------------------------------------------------------
# Midpoint (RK2) march through one layer
# ----------------------------------------------------------------------------
def _march(r_start, r_end, n_r, p0, h0, rhs, observe=None):
    """March (p, h) inward from r_start to r_end in n_r uniform steps.
    rhs(r, p, h) -> (dp/ds, dh/ds), with dp/ds the positive pressure-drop rate.
    observe(r, p, h), if given, is called at every node (n_r + 1 points).
    Raises _PressureCollapse if p <= 0 at a midpoint or node."""
    def check(p):
        if not p > 0.0:
            raise _PressureCollapse
        return p

    ds = (r_start - r_end) / n_r
    p, h = check(p0), h0
    if observe:
        observe(r_start, p, h)
    for j in range(n_r):
        r = r_start - j * ds
        dp1, dh1 = rhs(r, p, h)
        dp2, dh2 = rhs(r - 0.5 * ds, check(p - 0.5 * ds * dp1), h + 0.5 * ds * dh1)
        p = check(p - ds * dp2)
        h += ds * dh2
        if observe:
            observe(r - ds, p, h)
    return p, h


# ----------------------------------------------------------------------------
# One slice
# ----------------------------------------------------------------------------
def slice_dp(m_dot, q_slice, p_in, T_in, cold, hot, dz, n_r, geom=None):
    """docs/model.md section 1.

    m_dot   slice mass flow, kg/s (> 0)
    q_slice slice power, W
    p_in    pressure at the cold frit outer face, Pa
    T_in    inlet gas temperature, K
    cold    None (no cold frit) or (pore_size_m, porosity)
    hot     (pore_size_m, porosity)
    dz      slice height, m
    n_r     midpoint steps per layer
    geom    None (nominal element) or a params.py geometry dict

    Returns a dict: dp, T_gas_out, T_fuel_max, p_out, flags re_m_low,
    re_ht_low, out_of_range, pressure_collapse, fuel_over_limit, plus the
    per-layer drops dp_cold_frit, dp_bed, dp_hot_frit (Pa).
    """
    if m_dot <= 0.0:
        raise ValueError(f"slice_dp needs m_dot > 0 (got {m_dot})")
    if n_r < 1:
        raise ValueError(f"slice_dp needs n_r >= 1 (got {n_r})")

    r_ch, r_hf, r_bed, r_cf, d_p, alpha = _geometry(geom)
    v_bed = math.pi * (r_bed**2 - r_hf**2) * dz
    q3 = q_slice / v_bed                                   # W/m^3 of bed

    def G_at(r):
        return m_dot / (2.0 * math.pi * r * dz)

    flags = dict(re_m_low=False, re_ht_low=False, out_of_range=False,
                 pressure_collapse=False, fuel_over_limit=False)
    T_fuel_max = -math.inf

    # --- Layer 1: cold frit (adiabatic, T = T_in) ---------------------------
    def frit_rhs(T, d, eps):
        mu_T = float(pr.mu(T))
        def rhs(r, p, h):
            return _ergun_dpds(G_at(r), float(pr.rho(T, p)), mu_T, d, eps), 0.0
        return rhs

    # --- Layer 2: bed ---------------------------------------------------------
    def bed_rhs(r, p, h):
        T = float(pr.T_from_h(h))
        dpds, _ = _kta_dpds(G_at(r), float(pr.rho(T, p)), float(pr.mu(T)), d_p, alpha)
        return dpds, q3 * 2.0 * math.pi * r * dz / m_dot

    def bed_observe(r, p, h):
        nonlocal T_fuel_max
        T_g = float(pr.T_from_h(h))
        mu_T, k_T, cp_T = float(pr.mu(T_g)), float(pr.k(T_g)), float(pr.cp(T_g))
        Re = G_at(r) * d_p / mu_T
        if Re / (1.0 - alpha) < P.RE_M_MIN_KTA_DP:
            flags["re_m_low"] = True
        if Re < P.RE_MIN_KTA_HT:
            flags["re_ht_low"] = True
        h_c = _kta_nu(Re, mu_T * cp_T / k_T, alpha) * k_T / d_p
        a = 6.0 * (1.0 - alpha) / d_p
        T_fuel_max = max(T_fuel_max, T_g + q3 / (h_c * a))

    try:
        h_in = float(pr.h(T_in))
        p1 = p_in
        if cold is not None:
            p1, _ = _march(r_cf, r_bed, n_r, p_in, h_in,
                           frit_rhs(T_in, *cold))
        p2, h2 = _march(r_bed, r_hf, n_r, p1, h_in,
                        bed_rhs, observe=bed_observe)
        T_out = float(pr.T_from_h(h2))
        # --- Layer 3: hot frit (adiabatic, T = bed outlet temperature) ------
        p3, _ = _march(r_hf, r_ch, n_r, p2, h2,
                       frit_rhs(T_out, *hot))
    except (PropertyRangeError, _PressureCollapse) as err:
        key = "out_of_range" if isinstance(err, PropertyRangeError) else "pressure_collapse"
        flags[key] = True
        nan = math.nan
        return dict(dp=nan, T_gas_out=nan, T_fuel_max=nan, p_out=nan,
                    dp_cold_frit=nan, dp_bed=nan, dp_hot_frit=nan, **flags)

    flags["fuel_over_limit"] = T_fuel_max > P.T_LIMIT
    return dict(dp=p_in - p3, T_gas_out=T_out, T_fuel_max=T_fuel_max, p_out=p3,
                dp_cold_frit=p_in - p1, dp_bed=p1 - p2, dp_hot_frit=p2 - p3,
                **flags)
