"""
params.py - every model input, with its source, following the
"Step 1 (v3): Final Parameters" freeze document.

FREEZE RULE: commit this file to Git BEFORE any validation run. After that,
never change a value. If a change is unavoidable, add a new PARAMS_VERSION
and explain why in the commit message.

Status key (from the v3 document):
  S = sourced, A = assumption, R = rule computed in code, D = design choice

Sources:
  [J19b] Ji, Zhang, Sun, Shi. Thermal performance optimization of a fuel element
         in particle bed reactors for nuclear thermal propulsion.
         Nucl. Eng. Des. 355:110316, 2019. doi:10.1016/j.nucengdes.2019.110316
  [T19]  Ji, Liu, Sun, Shi. Analysis of the flow distribution in a particle bed
         reactor for nuclear thermal propulsion. Energies 12(19):3590, 2019.
  [K93]  Kerrebrock. Flow instability in particle-bed nuclear reactors.
         NASA Lewis, Nuclear Propulsion Technical Interchange Meeting, 1993.
  [SNTP] Haslett. Space Nuclear Thermal Propulsion Program Final Report, 1995.
  [HM]   Heeger Materials ZrC powder data sheet (density 6.78 g/cm^3).
  [v2]   Step 1 (v2) document -- defines rule D8.
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

# Derived thicknesses (inner edge of bed = outer edge of hot frit)
R_BED_IN    = R_HOT_FRIT_OUT
T_HOT_FRIT  = R_HOT_FRIT_OUT - R_HOT_CHAN        # 2 mm
T_BED       = R_BED_OUT - R_BED_IN               # 10 mm (20 particle diameters)
T_COLD_FRIT = R_COLD_FRIT_OUT - R_BED_OUT        # 2 mm
T_PLENUM    = R_ELEMENT - R_COLD_FRIT_OUT        # 2 mm

POWER_DENSITY  = 20.0e9    # W/m^3 (20.0 MW/L, per bed volume)         S [J19b] T1
T_INLET        = 200.0     # K                                         S [J19b] T1
P_OUTLET       = 7.0e6     # Pa, chamber (element outlet) pressure     S [J19b] T1
T_OUT_TARGET   = 3000.0    # K                                         S [J19b] T1
M_DOT_PUBLISHED = 0.505    # kg/s -- used ONLY for validation runs
                           #         and as a sensitivity case         S [J19b] T1
FLOW_PATTERN   = "Z"       # baseline; "U" is the alternative          S [J19b]

# Reference pressure for the property tables. properties.py imports P_OP.
P_OP = P_OUTLET

# Design limits -> optimizer constraints and app warnings
T_LIMIT        = 3800.0    # K, (U,Zr,Nb)C melting point               S [J19b]
DP_LIMIT       = 600.0e3   # Pa, about 1/3 of NERVA core drop          S [J19b]
APP_WARNING_TEMPS_K = (2800.0, 3100.0, 3500.0, 3800.0)  #              S [SNTP], [J19b]

# ============================================================================
# 2. MASS FLOW -- rule D2 (energy balance with OUR property data)
# ============================================================================
# Rule: choose the mass flow so total power raises hydrogen from T_INLET to
# T_OUT_TARGET using NIST-based enthalpy. We do NOT copy the published
# 0.505 kg/s, because the Tsinghua property model gives an enthalpy rise
# ~13-15% higher than NIST-JANAF; matching it would be tuning to the target.
# Hand estimate in the v3 document: about 0.576 kg/s.                    R (D2)
# Throttling rule: mass flow is proportional to power.                   D


def bed_volume(r_in=R_BED_IN, r_out=R_BED_OUT, length=L_ELEMENT):
    """Annular bed volume, m^3."""
    return math.pi * (r_out**2 - r_in**2) * length


def total_power(power_fraction=1.0):
    """Element thermal power, W."""
    return POWER_DENSITY * bed_volume() * power_fraction


def nominal_mass_flow(power_fraction=1.0):
    """Rule D2: m_dot = Q / (h(T_out) - h(T_in)), scaled with power (throttling)."""
    from properties import enthalpy_rise   # imported here to avoid a circular import
    return total_power(power_fraction) / enthalpy_rise(T_INLET, T_OUT_TARGET)


# ============================================================================
# 3. AXIAL POWER SHAPES -- rule D12, formulas from [J19b]
#    s = x / L from 0 to 1. Each shape is renormalized to a mean of exactly 1
#    so total power never changes.
#    Tier 1: uniform, cosine.  Tier 2: asymmetric, reflectors.
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


def power_shape(name, s):
    """Axial power shape at positions s (0-1), renormalized to mean exactly 1."""
    s_fine = np.linspace(0.0, 1.0, 2001)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    mean = integrate(_raw_shape(name, s_fine), s_fine)
    return _raw_shape(name, s) / mean


POWER_SHAPES_TIER1 = ("uniform", "cosine")
POWER_SHAPES_TIER2 = ("asymmetric", "reflectors")

# ============================================================================
# 4. FUEL (needed only for the transient model) -- rule D7
# ============================================================================
FUEL_DENSITY = 6780.0      # kg/m^3, ZrC proxy                         S [HM]
# FUEL_CP: from the NIST-JANAF ZrC table (C-110), computed in code.     TO DO
FUEL_PROPERTY_SENSITIVITY = 0.5  # +/-50% case covers the proxy uncertainty

# ============================================================================
# 5. MODEL RESOLUTION
# ============================================================================
N_SLICES = 20     # start; double until peak T and dp change < 1%
N_RADIAL = 20     # start; same rule

# ============================================================================
# 6. STUDY SETTINGS
# ============================================================================
POWER_SWEEP = (0.05, 1.00)          # fraction of nominal = 1-20 MW/L
OPERATIONAL_RANGE = (0.20, 1.00)    # 20% power = 4 MW/L
NASA_THRESHOLD_MW_PER_L = 4.0       # [K93] -- VERIFY in the PDF before freezing

# ============================================================================
# 7. VALIDATION ELEMENTS -- [T19] (inputs for validation runs only)
# ============================================================================
T19_ELEMENT = dict(        # used for V5, V6
    L=0.6392, r_hot_chan=0.02129, t_hot_frit=0.00300, t_bed=0.00615,
    t_cold_frit=0.00200, t_plenum=0.00300, d_particle=0.0005, eps_bed=0.38,
    cold_frit_pore=2.0e-6, cold_frit_eps=0.4, hot_frit_pore=1.0e-4, hot_frit_eps=0.5,
    power_density=30.0e9, m_dot=0.396,
    T_inlet=200.0, P=8.0e6,   # ASSUMPTIONS (A): run both outlet- and inlet-referenced
)
# Hot-frit-dominated T19 variant (V6): built by rule D8 ("swap shares").
# D8 is defined in the v2 document. TO DO: copy the rule here before freezing.

T19_LARGE_REACTOR = dict(  # used for V1-V4 (Tier 2: need channel + plenum model)
    R=0.507, L_options=(1.2, 1.0, 0.8), m_dot_options=(21.0, 17.5, 14.0),
    r_hot_chan=0.100, t_hot_frit=0.003, t_bed=0.281, t_cold_frit=0.002,
    t_plenum=0.010, d_particle=0.006, frit_pore=0.004,
    eps_bed=0.38, cold_frit_eps=0.1, hot_frit_eps=0.2,
    power=1000e6, T_inlet=200.0, P=8.0e6,
)

# ============================================================================
# 8. PRE-REGISTERED VALIDATION TARGETS -- v3 Section 6
#    These are comparisons, never model inputs.
#    Temperature criteria compare directions and changes, not absolute values.
# ============================================================================
VALIDATION = {
    # --- Tier 1: work with the uniform-pressure (no manifold) model ---
    "V5":  dict(tier=1, target="T19 element, cold-frit-dominated: dp = 1189.2 kPa",
                pass_rule="dp within +/-25%; stable"),
    "V6":  dict(tier=1, target="T19 element, hot-frit-dominated (rule D8)",
                pass_rule="instability, or no steady solution below 8000 K"),
    "V7":  dict(tier=1, target="No cold frit at design exit temperature [K93]",
                pass_rule="static instability below a power density within a factor of 3 of 4 MW/L"),
    "V8":  dict(tier=1, target="Strong cold frit [K93]",
                pass_rule="V7 instability disappears"),
    "V9":  dict(tier=1, target="Ledinegg check",
                pass_rule="eigenvalue and pressure-drop-slope onsets agree within one sweep step"),
    "V10": dict(tier=1, target="J19b design, uniform power, 0.505 kg/s: dp = 529.8 kPa",
                pass_rule="dp within +/-25% (Tier 1 compares radial part only; say so)"),
    # --- Tier 2: need the hot channel + plenum (manifold) model ---
    "V1":  dict(tier=2, target="T19 large reactor, Z: dp = 872.4 kPa",
                pass_rule="within +/-25%"),
    "V2":  dict(tier=2, target="Z vs. U",
                pass_rule="U has lower peak temperature; dp changes < 2%"),
    "V3":  dict(tier=2, target="Core heights 80/100/120 cm",
                pass_rule="dp within +/-25% each; beta decreases with shorter core"),
    "V4":  dict(tier=2, target="Frit-dominance variants",
                pass_rule="both lower peak temperature vs. baseline"),
    "V11": dict(tier=2, target="Cosine vs. uniform power (3348.4 -> 3905.5 K)",
                pass_rule="peak temperature rises; dp changes < 2%"),
    "V12": dict(tier=2, target="J19b optimal point A, cosine power (3147.6 K, 414.4 kPa)",
                pass_rule="peak T and dp both fall; dp within +/-25% of 414.4 kPa"),
    # --- Cut for the hackathon ---
    "V13": dict(tier=3, target="Reproduction-mode optimizer, cosine power",
                pass_rule="porosity peak upstream of power peak (b < 0.5)"),
}


if __name__ == "__main__":
    print(f"Thicknesses (mm): hot frit {T_HOT_FRIT*1e3:.1f}, bed {T_BED*1e3:.1f}, "
          f"cold frit {T_COLD_FRIT*1e3:.1f}, plenum {T_PLENUM*1e3:.1f}")
    print(f"Bed volume: {bed_volume()*1e3:.3f} L   (v3 says 1.307 L)")
    print(f"Element power: {total_power()/1e6:.2f} MW   (v3 says 26.14 MW)")
    print(f"Mass flow by rule D2: {nominal_mass_flow():.3f} kg/s   "
          f"(v3 hand estimate ~0.576; published {M_DOT_PUBLISHED})")
    s = np.linspace(0, 1, 2001)
    for name in POWER_SHAPES_TIER1 + POWER_SHAPES_TIER2:
        F = power_shape(name, s)
        print(f"Shape {name:<11} mean {F.mean():.4f}  peak {F.max():.3f} at s = {s[F.argmax()]:.2f}")
