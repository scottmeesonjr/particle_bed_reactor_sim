"""
properties.py -- hydrogen gas and fuel (ZrC proxy) properties, following the
v3 freeze document (decision D3, Section 5.3) plus decision D13.

Hydrogen, by temperature:
  below 950 K   : CoolProp, normal hydrogen, real gas at P_OP
  950-1050 K    : smooth blend (weights continuous in value AND slope)
  above 1050 K  : density  -> ideal gas
                  cp       -> NIST-JANAF H2 table (H-050), PCHIP interpolation
                  mu, k    -> Svehla (NASA TM-4647) curve fits, 1000-5000 K range
  enthalpy      : integral of the blended cp, anchored to CoolProp at T_MIN,
                  so h and cp are always consistent (energy balance closes)

Property limit (decision D13): T_MAX = 5000 K, the top of the Svehla fit range.
Above it this module raises PropertyRangeError. The solver treats that as
"no steady solution within the property range" (see validation criterion V6).
Real hydrogen begins to dissociate in the upper part of this range; the model
ignores dissociation. State this as a limitation.

Pressure: tables are built at P_OP. Density is scaled by P / P_OP. Viscosity,
conductivity and cp are taken at P_OP. Pressure drop is under ~1 MPa of 7 MPa.

Fuel: ZrC proxy. cp from the NIST-JANAF ZrC table (C-110); density from params.

Data files (in data/): janaf_H2.csv, janaf_ZrC.csv.
All public functions accept numpy arrays.
"""
from pathlib import Path

import numpy as np
import CoolProp.CoolProp as CP
from scipy.interpolate import PchipInterpolator

from params import P_OP, FUEL_DENSITY

FLUID = "Hydrogen"
M_H2 = 2.01588e-3            # kg/mol
M_ZRC = 103.235e-3           # kg/mol (Zr 91.224 + C 12.011)
R_U = 8.314462618            # J/(mol K)

T_MIN = 150.0                # K, lowest table temperature
T_MAX = 5000.0               # K, property limit (D13): top of Svehla fit range
BLEND_LO, BLEND_HI = 950.0, 1050.0
DT_TABLE = 1.0               # K, table spacing

DATA_DIR = Path(__file__).resolve().parent / "data"


class PropertyRangeError(ValueError):
    """Raised when a temperature leaves the property range (T_MIN to T_MAX)."""


# ----------------------------------------------------------------------------
# Svehla (1995), NASA TM-4647, hydrogen H2 curve fits:
#   ln(eta) = A ln T + B/T + C/T^2 + D,  eta in micropoise
#   ln(lam) = A ln T + B/T + C/T^2 + D,  lam in microwatt/(cm K)
# Ranges 200-1000 K and 1000-5000 K.
# CHECK EVERY NUMBER against the PDF in data/ before freezing.
# Spot check: viscosity at 1000 K = 2.056e-5 Pa s (both ranges agree there).
# ----------------------------------------------------------------------------
_SVEHLA_VISC = [
    (200.0, 1000.0, (0.74553182, 43.555109, -3257.9340, 0.13556243)),
    (1000.0, 5000.0, (0.96730605, 679.31897, -210251.79, -1.8251697)),
]
_SVEHLA_COND = [
    (200.0, 1000.0, (1.0240124, 297.09752, -31396.363, 1.0560824)),
    (1000.0, 5000.0, (1.0611992, 258.85783, 6316.3191, 0.79973205)),
]


def _svehla(T, ranges):
    T = np.asarray(T, dtype=float)
    out = np.full_like(T, np.nan)
    for lo, hi, (A, B, C, D) in ranges:
        m = (T >= lo) & (T <= hi)
        out[m] = np.exp(A * np.log(T[m]) + B / T[m] + C / T[m] ** 2 + D)
    return out


def svehla_viscosity(T):
    """Pa s."""
    return _svehla(T, _SVEHLA_VISC) * 1e-7          # micropoise -> Pa s


def svehla_conductivity(T):
    """W/(m K)."""
    return _svehla(T, _SVEHLA_COND) * 1e-4          # uW/(cm K) -> W/(m K)


# ----------------------------------------------------------------------------
# JANAF tables
# ----------------------------------------------------------------------------
def _load_janaf(name, molar_mass):
    lines = [ln for ln in (DATA_DIR / name).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    d = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])  # skip header row
    T = d[:, 0]
    cp = d[:, 1] / molar_mass                 # J/(kg K)
    h = d[:, 2] * 1e3 / molar_mass            # J/kg
    return T, cp, h


_H2_T, _H2_CP, _H2_H = _load_janaf("janaf_H2.csv", M_H2)
_ZRC_T, _ZRC_CP, _ZRC_H = _load_janaf("janaf_ZrC.csv", M_ZRC)
_janaf_h2_cp = PchipInterpolator(_H2_T, _H2_CP)
_janaf_zrc_cp = PchipInterpolator(_ZRC_T, _ZRC_CP)


# ----------------------------------------------------------------------------
# Blend weight: 0 below BLEND_LO, 1 above BLEND_HI, smoothstep between.
# Smoothstep (3x^2 - 2x^3) has zero slope at both ends, so the blended
# property is continuous in value and slope.
# ----------------------------------------------------------------------------
def _blend_weight(T):
    x = np.clip((np.asarray(T, dtype=float) - BLEND_LO) / (BLEND_HI - BLEND_LO), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _build_tables():
    T = np.arange(T_MIN, T_MAX + DT_TABLE / 2, DT_TABLE)
    low = T <= BLEND_HI
    high = T >= BLEND_LO
    w = _blend_weight(T)

    def coolprop(prop):
        out = np.full_like(T, np.nan)
        out[low] = CP.PropsSI(prop, "T", T[low], "P", P_OP, FLUID)
        return out

    def high_side(fn):
        out = np.full_like(T, np.nan)
        out[high] = fn(T[high])
        return out

    rho_lo, mu_lo, k_lo, cp_lo = (coolprop(p) for p in ("D", "V", "L", "C"))
    rho_hi = high_side(lambda t: P_OP * M_H2 / (R_U * t))
    mu_hi = high_side(svehla_viscosity)
    k_hi = high_side(svehla_conductivity)
    cp_hi = high_side(_janaf_h2_cp)

    def blend(lo, hi):
        return np.where(T < BLEND_LO, lo, np.where(T > BLEND_HI, hi, (1 - w) * lo + w * hi))

    rho, mu, k, cp = blend(rho_lo, rho_hi), blend(mu_lo, mu_hi), blend(k_lo, k_hi), blend(cp_lo, cp_hi)

    # Enthalpy = integral of cp, anchored to CoolProp at T_MIN
    h0 = CP.PropsSI("H", "T", T_MIN, "P", P_OP, FLUID)
    h = h0 + np.concatenate([[0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(T))])
    return T, rho, mu, k, cp, h


_T, _RHO, _MU, _K, _CP, _H = _build_tables()


def _check(T, lo=T_MIN, hi=T_MAX):
    T = np.asarray(T, dtype=float)
    if np.any(T < lo) or np.any(T > hi) or np.any(~np.isfinite(T)):
        raise PropertyRangeError(
            f"Temperature outside the property range {lo:.0f}-{hi:.0f} K "
            f"(got {np.nanmin(T):.0f}-{np.nanmax(T):.0f} K)")
    return T


# ----------------------------------------------------------------------------
# Public hydrogen functions
# ----------------------------------------------------------------------------
def rho(T, P=P_OP):
    """Density, kg/m^3 (scaled by P / P_OP)."""
    return np.interp(_check(T), _T, _RHO) * (np.asarray(P, dtype=float) / P_OP)


def mu(T):
    """Dynamic viscosity, Pa s."""
    return np.interp(_check(T), _T, _MU)


def k(T):
    """Thermal conductivity, W/(m K)."""
    return np.interp(_check(T), _T, _K)


def cp(T):
    """Isobaric specific heat, J/(kg K)."""
    return np.interp(_check(T), _T, _CP)


def h(T):
    """Specific enthalpy, J/kg (consistent with cp)."""
    return np.interp(_check(T), _T, _H)


def T_from_h(h_value):
    """Temperature from enthalpy, K (inverse of h). Raises PropertyRangeError outside the table."""
    h_value = np.asarray(h_value, dtype=float)
    if np.any(h_value < _H[0]) or np.any(h_value > _H[-1]):
        raise PropertyRangeError("Enthalpy outside the property range")
    return np.interp(h_value, _H, _T)


def enthalpy_rise(T1, T2):
    """h(T2) - h(T1), J/kg."""
    return float(h(T2) - h(T1))


# ----------------------------------------------------------------------------
# Fuel (ZrC proxy)
# ----------------------------------------------------------------------------
def fuel_cp(T):
    """ZrC specific heat, J/(kg K), NIST-JANAF C-110."""
    return _janaf_zrc_cp(_check(T, lo=_ZRC_T[0], hi=_ZRC_T[-1]))


def fuel_rho_cp(T):
    """ZrC volumetric heat capacity of the solid, J/(m^3 K)."""
    return FUEL_DENSITY * fuel_cp(T)


# ----------------------------------------------------------------------------
# Reports required by v3 Section 7
# ----------------------------------------------------------------------------
def ideal_gas_error_at_1000K():
    """Relative density difference, ideal gas vs CoolProp, at 1000 K and P_OP."""
    real = CP.PropsSI("D", "T", 1000.0, "P", P_OP, FLUID)
    ideal = P_OP * M_H2 / (R_U * 1000.0)
    return (ideal - real) / real


def continuity_report(T_lo=900.0, T_hi=1100.0):
    """Largest relative step in value and in slope across the blend region."""
    m = (_T >= T_lo) & (_T <= T_hi)
    out = {}
    for name, arr in (("rho", _RHO), ("mu", _MU), ("k", _K), ("cp", _CP), ("h", _H)):
        y = arr[m]
        dy = np.diff(y)
        slope = dy / DT_TABLE
        out[name] = dict(
            max_value_step=float(np.max(np.abs(dy) / np.abs(y[:-1]))),
            max_slope_change=float(np.max(np.abs(np.diff(slope))) / (np.max(np.abs(slope)) + 1e-300)),
        )
    return out


if __name__ == "__main__":
    import params as p

    print(f"{'T (K)':>7} {'rho':>9} {'mu (uPa s)':>11} {'k (W/mK)':>9} {'cp (kJ/kgK)':>12}")
    for T in (200, 500, 900, 1000, 1100, 1500, 2000, 3000, 4000, 5000):
        print(f"{T:7.0f} {rho(T):9.3f} {mu(T)*1e6:11.2f} {k(T):9.3f} {cp(T)/1e3:12.2f}")

    print(f"\nSvehla viscosity at 1000 K: {svehla_viscosity(np.array([1000.0]))[0]:.4e} Pa s "
          f"(plan's spot value 2.056e-5)")
    print(f"Ideal-gas density error at 1000 K, {P_OP/1e6:.1f} MPa: {ideal_gas_error_at_1000K()*100:+.2f}%")

    print("\nContinuity over 900-1100 K (largest step per 1 K; largest slope change, relative):")
    for name, r in continuity_report().items():
        print(f"  {name:>4}: value {r['max_value_step']:.2e}   slope {r['max_slope_change']:.2e}")

    print(f"\nEnthalpy rise 200 -> 3000 K: {enthalpy_rise(200, 3000)/1e6:.2f} MJ/kg")
    print(f"Mass flow by rule D2: {p.nominal_mass_flow():.4f} kg/s (frozen value is whatever prints here)")
    print("ZrC volumetric heat capacity (MJ/m^3 K; v3 says ~3.5, 3.8, 4.0):",
          ", ".join(f"{fuel_rho_cp(T)/1e6:.2f} at {T} K" for T in (1000, 2000, 3000)))

    try:
        mu(5200.0)
    except PropertyRangeError as e:
        print(f"\nAbove the limit -> PropertyRangeError: {e}")