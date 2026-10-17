"""
properties.py -- hydrogen properties from 150 K to 4000 K.

CoolProp's normal-hydrogen equation of state is only valid up to 1000 K,
but outlet temperatures reach ~3000 K. So:
  T <= 1000 K : CoolProp (real gas)
  T >  1000 K : extended so every property is continuous at 1000 K
    density   : ideal-gas scaling, with the compressibility Z frozen at 1000 K
    viscosity : power law mu ~ T^n fitted to CoolProp over 600-1000 K
    conduct.  : power law k ~ T^m fitted the same way
    cp        : NIST WebBook Shomate equation for H2 gas, shifted to match
                CoolProp at 1000 K
State this extension in the project's limitations.

Pressure dependence: tables are built at P_OP. Density is scaled by P/P_OP
(ideal-gas behaviour). Viscosity, conductivity and cp are taken at P_OP.
This is reasonable because the pressure drop is ~1 MPa out of 8 MPa.

All functions accept numpy arrays.
"""
import numpy as np
import CoolProp.CoolProp as CP

from params import P_OP

FLUID = "Hydrogen"
M_H2 = 2.01588e-3          # kg/mol
R_U = 8.314462618          # J/mol-K
T_SWITCH = 1000.0          # K, upper limit of the CoolProp model
T_MIN, T_MAX = 150.0, 4000.0

# NIST WebBook Shomate coefficients for H2 gas: cp = A + B t + C t^2 + D t^3 + E/t^2
# (J/mol-K, t = T/1000). VERIFY against webbook.nist.gov before freezing.
_SHOMATE = [
    (298.0, 1000.0, (33.066178, -11.363417, 11.432816, -2.772874, -0.158558)),
    (1000.0, 2500.0, (18.563083, 12.257357, -2.859786, 0.268238, 1.977990)),
    (2500.0, 6000.0, (43.413560, -4.293079, 1.272428, -0.096876, -20.533862)),
]


def _cp_shomate(T):
    """Ideal-gas cp of H2, J/kg-K."""
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    for lo, hi, (A, B, C, D, E) in _SHOMATE:
        m = (T >= lo) & (T <= hi)
        t = T[m] / 1000.0
        out[m] = (A + B*t + C*t**2 + D*t**3 + E/t**2) / M_H2
    return out


def _coolprop(prop, T, P=P_OP):
    return np.array([CP.PropsSI(prop, "T", float(t), "P", P, FLUID) for t in np.atleast_1d(T)])


def _build_tables(n=800):
    T = np.linspace(T_MIN, T_MAX, n)
    lo = T <= T_SWITCH
    hi = ~lo

    # Low-temperature part straight from CoolProp
    rho = np.empty(n); mu = np.empty(n); k = np.empty(n); cp = np.empty(n)
    rho[lo] = _coolprop("D", T[lo])
    mu[lo] = _coolprop("V", T[lo])
    k[lo] = _coolprop("L", T[lo])
    cp[lo] = _coolprop("C", T[lo])

    # Values at the switch point for continuity
    rho_s, mu_s, k_s, cp_s = (_coolprop(p, [T_SWITCH])[0] for p in ("D", "V", "L", "C"))
    Z_s = P_OP * M_H2 / (rho_s * R_U * T_SWITCH)

    # Power-law exponents fitted over 600-1000 K
    Tfit = np.linspace(600.0, T_SWITCH, 20)
    n_mu = np.polyfit(np.log(Tfit), np.log(_coolprop("V", Tfit)), 1)[0]
    n_k = np.polyfit(np.log(Tfit), np.log(_coolprop("L", Tfit)), 1)[0]

    Th = T[hi]
    rho[hi] = P_OP * M_H2 / (Z_s * R_U * Th)
    mu[hi] = mu_s * (Th / T_SWITCH) ** n_mu
    k[hi] = k_s * (Th / T_SWITCH) ** n_k
    cp[hi] = _cp_shomate(Th) + (cp_s - _cp_shomate(np.array([T_SWITCH]))[0])

    return T, rho, mu, k, cp, n_mu, n_k


_T, _RHO, _MU, _K, _CP, N_MU, N_K = _build_tables()


def _check(T):
    T = np.asarray(T, dtype=float)
    if np.any(T < T_MIN) or np.any(T > T_MAX):
        raise ValueError(f"Temperature outside {T_MIN}-{T_MAX} K: {T.min():.0f}-{T.max():.0f} K")
    return T


def rho(T, P=P_OP):
    """Density, kg/m^3."""
    return np.interp(_check(T), _T, _RHO) * (np.asarray(P) / P_OP)


def mu(T):
    """Dynamic viscosity, Pa-s."""
    return np.interp(_check(T), _T, _MU)


def k(T):
    """Thermal conductivity, W/m-K."""
    return np.interp(_check(T), _T, _K)


def cp(T):
    """Isobaric specific heat, J/kg-K."""
    return np.interp(_check(T), _T, _CP)


def enthalpy_rise(T1, T2, n=2000):
    """Integral of cp dT from T1 to T2, J/kg."""
    Ts = np.linspace(T1, T2, n)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(cp(Ts), Ts)


if __name__ == "__main__":
    import params as p

    print(f"Fitted exponents above 1000 K: mu ~ T^{N_MU:.3f}, k ~ T^{N_K:.3f}")
    print(f"{'T (K)':>7} {'rho':>9} {'mu (uPa s)':>11} {'k':>7} {'cp (kJ/kgK)':>12}")
    for T in (200, 500, 1000, 1500, 2000, 2500, 3000):
        print(f"{T:7.0f} {rho(T):9.3f} {mu(T)*1e6:11.2f} {k(T):7.3f} {cp(T)/1e3:12.2f}")

    # Sanity check 1: energy balance
    q_from_flow = p.M_DOT_TOTAL * enthalpy_rise(p.T_INLET, p.T_OUT_TARGET)
    print(f"\nPower from flow x enthalpy rise (200->3000 K): {q_from_flow/1e6:.2f} MW")
    print(f"Power from power density x bed volume:         {p.total_power()/1e6:.2f} MW")

    # Sanity check 2: viscous Ergun pressure drop across the cold frit at inlet conditions
    A = 2 * np.pi * p.R_COLD_FRIT_OUT * p.L_ELEMENT
    u = p.M_DOT_TOTAL / (rho(p.T_INLET) * A)
    e, d = p.COLD_FRIT_EPS, p.COLD_FRIT_PORE
    dP = 150 * mu(p.T_INLET) * (1 - e)**2 / (e**3 * d**2) * u * p.T_COLD_FRIT
    print(f"\nCold frit viscous pressure drop (estimate): {dP/1e3:.0f} kPa")
    print(f"Total element pressure drop from [2] CFD:   {p.VALIDATION_TARGETS['cold_frit_dominated_dP_total_kPa']} kPa")
