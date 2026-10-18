"""
params.py -- every model input, with its source, following the
"Step 1 (v3): Final Parameters" freeze document plus decision D13.

FREEZE RULE: commit this file together with properties.py, rules.py, data/
and validation.md BEFORE any validation run. After that, never change a value.
A real bug found later becomes a new PARAMS_VERSION; rerun everything and
report both versions.

Status key (from v3): S = sourced, A = assumption, R = rule computed in code,
D = design choice.

Sources:
  [J19b]  Ji, Zhang, Sun, Shi. Thermal performance optimization of a fuel element
          in particle bed reactors for nuclear thermal propulsion.
          Nucl. Eng. Des. 355:110316, 2019. doi:10.1016/j.nucengdes.2019.110316
  [T19]   Ji, Liu, Sun, Shi. Analysis of the flow distribution in a particle bed
          reactor for nuclear thermal propulsion. Energies 12(19):3590, 2019.
  [K93]   Kerrebrock. Flow instability in particle-bed nuclear reactors.
          NASA Lewis, Nuclear Propulsion Technical Interchange Meeting, 1993.
  [SNTP]  Haslett. Space Nuclear Thermal Propulsion Program Final Report, 1995.
  [HM]    Heeger Materials ZrC powder data sheet (density 6.78 g/cm^3).
  [JANAF] NIST-JANAF tables H-050 (H2) and C-110 (ZrC), files in data/.
  [SV95]  Svehla. Transport Coefficients for the NASA Lewis Chemical
          Equilibrium Program. NASA TM-4647, 1995.
  [R22]   Reger et al., "Toward Development of an Improved Friction Correlation
          for the Near-Wall Region of Pebble Bed Systems," arXiv:2203.05041 --
          states the KTA pressure-drop range 10 < Re_m < 1e5, 0.36 < eps < 0.42.
"""
import math
import numpy as np

PARAMS_VERSION = "v3"

# ============================================================================
# 1. NOMINAL FUEL ELEMENT -- [J19b] preliminary design (Table 1 and Table 2)
# ============================================================================
R_HOT_CHAN      = 0.025    # m  hot gas channel radius                 S [J19b] T2
R_HOT_FRIT_OUT  = 0.027    # m  outer radius of hot frit               S [J19b] T2
R_BED_OUT       = 0.037    # m  outer radius of particle bed           S [J19b] T2
R_COLD_FRIT_OUT = 0.039    # m  outer radius of cold frit              S [J19b] T2
R_ELEMENT       = 0.041    # m  element (plenum outer) radius          S [J19b] T2
L_ELEMENT       = 0.65     # m  length                                 S [J19b] T2

D_PARTICLE      = 0.0005   # m  fuel particle diameter                 S [J19b] T2
EPS_BED         = 0.38     # -  bed porosity                           S [J19b] T2
COLD_FRIT_PORE  = 2.0e-6   # m                                         S [J19b] T2
COLD_FRIT_EPS   = 0.5      # -                                         S [J19b] T2
HOT_FRIT_PORE   = 1.0e-4   # m                                         S [J19b] T2
HOT_FRIT_EPS    = 0.5      # -                                         S [J19b] T2

R_BED_IN    = R_HOT_FRIT_OUT
T_HOT_FRIT  = R_HOT_FRIT_OUT - R_HOT_CHAN        # 2 mm
T_BED       = R_BED_OUT - R_BED_IN               # 10 mm (20 particle diameters)
T_COLD_FRIT = R_COLD_FRIT_OUT - R_BED_OUT        # 2 mm
T_PLENUM    = R_ELEMENT - R_COLD_FRIT_OUT        # 2 mm

POWER_DENSITY   = 20.0e9   # W/m^3 (20.0 MW/L of bed volume)           S [J19b] T1
T_INLET         = 200.0    # K                                         S [J19b] T1
P_OUTLET        = 7.0e6    # Pa, chamber (element outlet) pressure     S [J19b] T1
T_OUT_TARGET    = 3000.0   # K                                         S [J19b] T1
M_DOT_PUBLISHED = 0.505    # kg/s -- ONLY the validation runner uses it S [J19b] T1
FLOW_PATTERN    = "Z"      # baseline; "U" is Tier 2                   S [J19b]

P_OP = P_OUTLET            # reference pressure for property tables

T_LIMIT  = 3800.0          # K, (U,Zr,Nb)C melting point               S [J19b]
DP_LIMIT = 600.0e3         # Pa, about 1/3 of NERVA core drop          S [J19b]
APP_WARNING_TEMPS_K = (2800.0, 3100.0, 3500.0, 3800.0)  #              S [SNTP], [J19b]

# ============================================================================
# 2. MASS FLOW -- rule D2 (energy balance with our NIST-based enthalpy)
# ============================================================================
# m_dot = Q / (h(T_OUT_TARGET) - h(T_INLET)). The published 0.505 kg/s is not
# copied: the Tsinghua property model implies ~13-15% more enthalpy rise than
# NIST-JANAF, and matching it would be tuning to the target.         R (D2)
# Throttling: mass flow proportional to power.                        D


def bed_volume(r_in=R_BED_IN, r_out=R_BED_OUT, length=L_ELEMENT):
    """Annular bed volume, m^3."""
    return math.pi * (r_out**2 - r_in**2) * length


def total_power(power_fraction=1.0):
    """Element thermal power, W."""
    return POWER_DENSITY * bed_volume() * power_fraction


def nominal_mass_flow(power_fraction=1.0):
    """Rule D2 with throttling. Frozen value = what this prints after the freeze commit."""
    from properties import enthalpy_rise   # local import avoids a circular import
    return total_power(power_fraction) / enthalpy_rise(T_INLET, T_OUT_TARGET)


# ============================================================================
# 3. AXIAL POWER SHAPES -- rule D12, [J19b] Eqs. 14-16; s = x/L
#    Renormalized to a mean of exactly 1. Tier 1: uniform, cosine.
# ============================================================================
_REFLECTOR_COEFFS = (1.094, -4.002, 31.445, -101.605, 166.788, -138.205, 45.517)


def _raw_shape(name, s):
    s = np.asarray(s, dtype=float)
    if name == "uniform":
        return np.ones_like(s)
    if name == "cosine":
        return 1.25 * np.cos(np.pi * (s - 0.5) / 1.4)
    if name == "asymmetric":
        return 0.398 + 1.9752 * s + 0.5265 * s**2 - 2.244 * s**3
    if name == "reflectors":
        return sum(a * s**i for i, a in enumerate(_REFLECTOR_COEFFS))
    raise ValueError(f"Unknown power shape: {name}")


def _mean_over_unit(f):
    s = np.linspace(0.0, 1.0, 4001)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(f(s), s)


def power_shape(name, s):
    """Axial power shape at s (0-1), renormalized to mean exactly 1."""
    return _raw_shape(name, s) / _mean_over_unit(lambda x: _raw_shape(name, x))


POWER_SHAPES_TIER1 = ("uniform", "cosine")
POWER_SHAPES_TIER3 = ("asymmetric", "reflectors")

# ============================================================================
# 4. COLD FRIT POROSITY SHAPE FUNCTION -- [J19b] Eqs. 1-6; s = x/L
#    eps(s) = eps0 + eps1 + eps2
#    eps0 = (a (s - b)^2 + c) * alpha0, with a = 3(1-c)/(b^3 + (1-b)^3)
#           (the mean relative base porosity eps0/alpha0 is exactly 1)
#    eps1 = alpha1 * exp(-lam1 * s)          top reflector
#    eps2 = alpha2 * exp(-lam2 * (1 - s))    bottom reflector
# ============================================================================


def cold_frit_porosity(s, b, c, alpha0, alpha1=0.0, lam1=0.0, alpha2=0.0, lam2=0.0):
    """Absolute cold frit porosity along the element, [J19b] shape function."""
    s = np.asarray(s, dtype=float)
    a = 3.0 * (1.0 - c) / (b**3 + (1.0 - b)**3)
    eps0 = (a * (s - b)**2 + c) * alpha0
    eps1 = alpha1 * np.exp(-lam1 * s) if alpha1 else 0.0
    eps2 = alpha2 * np.exp(-lam2 * (1.0 - s)) if alpha2 else 0.0
    return eps0 + eps1 + eps2


# Published optimal designs (Pareto points chosen by [J19b], Sec. 4.2)
POINT_A = dict(b=0.434, c=1.070, alpha0=0.533)          # cosine power      S [J19b]
POINT_B = dict(b=0.444, c=1.052, alpha0=0.582)          # asymmetric power  S [J19b]
POINT_C = dict(b=0.418, c=1.036, alpha0=0.478,          # reflectors        S [J19b]
               alpha1=0.164, lam1=28.470, alpha2=0.162, lam2=32.830)

# Shape-function bounds used by [J19b] (Tables 3 and 4, Sec. 4.2.1.2)
SHAPE_BOUNDS = {
    "bare_symmetric":  dict(b=(0.3, 0.5), c=(1.0, 1.2), alpha0=(0.4, 0.6)),
    "bare_asymmetric": dict(b=(0.4, 0.7), c=(1.0, 1.2), alpha0=(0.4, 0.6)),
    "reflectors":      dict(b=(0.4, 0.5), c=(1.0, 1.2), alpha0=(0.4, 0.6),
                            alpha1=(0.1, 0.4), lam1=(10.0, 50.0),
                            alpha2=(0.1, 0.4), lam2=(10.0, 50.0)),
}


def free_zone_bounds(n_zones=8, case="bare_symmetric", n_grid=21):
    """Tier 3 rule: each zone's porosity range = what the [J19b] shape function
    can reach within its published bounds (zone-averaged). Returns (n_zones, 2)."""
    bnd = SHAPE_BOUNDS[case]
    edges = np.linspace(0.0, 1.0, n_zones + 1)
    s = np.linspace(0.0, 1.0, 801)
    zone = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_zones - 1)
    lo, hi = np.full(n_zones, np.inf), np.full(n_zones, -np.inf)
    for b in np.linspace(*bnd["b"], n_grid):
        for c in np.linspace(*bnd["c"], n_grid):
            for a0 in np.linspace(*bnd["alpha0"], n_grid):
                eps = cold_frit_porosity(s, b, c, a0)
                z = np.bincount(zone, weights=eps) / np.bincount(zone)
                lo, hi = np.minimum(lo, z), np.maximum(hi, z)
    return np.column_stack([lo, hi])


# ============================================================================
# 5. FUEL -- rule D7 (needed only by the transient model)
# ============================================================================
FUEL_DENSITY = 6780.0        # kg/m^3, ZrC proxy                       S [HM]
# Fuel cp: properties.fuel_cp(T), NIST-JANAF C-110 table in data/.     S [JANAF]
FUEL_PROPERTY_SENSITIVITY = 0.5   # +/-50% case covers the proxy uncertainty

# ============================================================================
# 6. MODEL RESOLUTION AND CORRELATION RANGES
# ============================================================================
N_SLICES = 20     # start; double until peak T and dp change < 1%      R
N_RADIAL = 20     # start; same rule                                   R
GRID_TOLERANCE = 0.01

RE_M_MIN_KTA_DP = 10.0       # KTA bed pressure drop valid for 10 < Re_m < 1e5   S [R22]
RE_M_MAX_KTA_DP = 1.0e5
EPS_RANGE_KTA_DP = (0.36, 0.42)  #                                    S [R22]
RE_MIN_KTA_HT = 100.0        # KTA heat transfer lower limit (v2 audit, D4)
RE_MIN_PETUKHOV = 3000.0   # below this, manifold friction uses max(64/Re, Petukhov(3000)); model.md Amendment A2
# Re_m = Re / (1 - eps). Runs below these are FLAGGED, never stopped.

# ============================================================================
# 7. STUDY SETTINGS
# ============================================================================
POWER_SWEEP = (0.05, 1.00)            # fraction of nominal = 1-20 MW/L
OPERATIONAL_RANGE = (0.20, 1.00)      # 20% power = 4 MW/L
Q1_POWER_DENSITY_RANGE = (1.0e9, 30.0e9)   # W/m^3 = 1-30 MW/L, both elements (Q1)
NASA_THRESHOLD_MW_PER_L = 4.0         # [K93] confirmed, Kerrebrock pp. 498

# ============================================================================
# 8. VALIDATION ELEMENTS -- [T19] (validation runs only)
# ============================================================================
T19_ELEMENT = dict(        # V5, V6
    L=0.6392, r_hot_chan=0.02129, t_hot_frit=0.00300, t_bed=0.00615,
    t_cold_frit=0.00200, t_plenum=0.00300, d_particle=0.0005, eps_bed=0.38,
    cold_frit_pore=2.0e-6, cold_frit_eps=0.4, hot_frit_pore=1.0e-4, hot_frit_eps=0.5,
    power_density=30.0e9, m_dot=0.396, T_inlet=200.0,
    P=8.0e6,               # A: run BOTH outlet-referenced and inlet-referenced
)
# Hot-frit-dominated variant (V6): rules.hot_frit_dominated() -- rule D8.

T19_LARGE_REACTOR = dict(  # V1-V4 (Tier 2)
    R=0.507, L_options=(1.2, 1.0, 0.8), m_dot_options=(21.0, 17.5, 14.0),
    r_hot_chan=0.100, t_hot_frit=0.003, t_bed=0.281, t_cold_frit=0.002,
    t_plenum=0.010, d_particle=0.006, frit_pore=0.004,
    eps_bed=0.38, cold_frit_eps=0.1, hot_frit_eps=0.2,
    power=1000e6, T_inlet=200.0, P=8.0e6,
)

# ============================================================================
# 9. PRE-REGISTERED VALIDATION TARGETS -- v3 Section 6, tiers from the plan.
#    Comparisons only, never inputs. Temperatures compare trends, not values.
# ============================================================================
VALIDATION = {
    "V5":  dict(tier=1, target="T19 element, cold-frit-dominated: dp = 1189.2 kPa",
                pass_rule="dp within +/-25%; stable", needs_channel=True),
    "V6":  dict(tier=1, target="T19 element, hot-frit-dominated (rule D8)",
                pass_rule="instability, or no steady solution with all temperatures "
                          "below the property limit (5000 K, decision D13)",
                needs_channel=False),
    "V7":  dict(tier=1, target="No cold frit at design exit temperature [K93]",
                pass_rule="static instability below a power density within a factor "
                          "of 3 of 4 MW/L", needs_channel=False),
    "V8":  dict(tier=1, target="Strong cold frit [K93]",
                pass_rule="V7 instability disappears", needs_channel=False),
    "V9":  dict(tier=1, target="Ledinegg check",
                pass_rule="eigenvalue and pressure-drop-slope onsets agree within one "
                          "sweep step", needs_channel=False),
    "V10": dict(tier=1, target="J19b design, uniform power, 0.505 kg/s: dp = 529.8 kPa",
                pass_rule="dp within +/-25%", needs_channel=True),
    "V11": dict(tier=1, target="Cosine vs. uniform power (3348.4 -> 3905.5 K)",
                pass_rule="peak temperature rises; dp changes < 2%", needs_channel=True),
    "V12": dict(tier=1, target="J19b point A, cosine power (3147.6 K, 414.4 kPa)",
                pass_rule="peak T and dp both fall; dp within +/-25% of 414.4 kPa",
                needs_channel=True),
    "V1":  dict(tier=2, target="T19 large reactor, Z: dp = 872.4 kPa",
                pass_rule="within +/-25%", needs_channel=True),
    "V2":  dict(tier=2, target="Z vs. U",
                pass_rule="U has lower peak temperature; dp changes < 2%", needs_channel=True),
    "V3":  dict(tier=2, target="Core heights 80/100/120 cm",
                pass_rule="dp within +/-25% each; beta decreases with shorter core",
                needs_channel=True),
    "V4":  dict(tier=2, target="Frit-dominance variants",
                pass_rule="both lower peak temperature vs. baseline", needs_channel=True),
    "V13": dict(tier=3, target="Reproduction-mode optimizer, cosine power",
                pass_rule="porosity peak upstream of power peak (b < 0.5)", needs_channel=True),
}
# Fallback rule (fixed now): if the channel model is not passing tests at the
# 7:30 checkpoint, every target with needs_channel=True is marked "not run".


if __name__ == "__main__":
    print(f"Thicknesses (mm): hot frit {T_HOT_FRIT*1e3:.1f}, bed {T_BED*1e3:.1f}, "
          f"cold frit {T_COLD_FRIT*1e3:.1f}, plenum {T_PLENUM*1e3:.1f}")
    print(f"Bed volume: {bed_volume()*1e3:.3f} L   (v3: 1.307 L)")
    print(f"Element power: {total_power()/1e6:.2f} MW   (v3: 26.14 MW)")
    print(f"Mass flow by rule D2: {nominal_mass_flow():.4f} kg/s   (v3 estimate ~0.576)")
    s = np.linspace(0, 1, 4001)
    for name in POWER_SHAPES_TIER1 + POWER_SHAPES_TIER3:
        F = power_shape(name, s)
        print(f"Power shape {name:<11} mean {F.mean():.4f}  peak {F.max():.3f} at s = {s[F.argmax()]:.2f}")
    epsA = cold_frit_porosity(s, **POINT_A)
    print(f"Point A: mean relative porosity {epsA.mean()/POINT_A['alpha0']:.4f}, "
          f"peak at x = {s[epsA.argmax()]*L_ELEMENT:.3f} m (paper Fig. 7: ~0.28 m), "
          f"range {epsA.min():.3f}-{epsA.max():.3f}")
    fz = free_zone_bounds()
    print("Free-zone porosity bounds (8 zones):", ", ".join(f"{a:.3f}-{b:.3f}" for a, b in fz))