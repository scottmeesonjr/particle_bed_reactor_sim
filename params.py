"""
params.py - every model input, with its source.

FREEZE RULE: commit this file to Git BEFORE running any validation case.
Do not change a value after validation starts. If you must, add a new
version below and say why in the commit message.

Main source:
  [2] Ji, Liu, Sun, Shi. "Analysis of the Flow Distribution in a Particle Bed
      Reactor for Nuclear Thermal Propulsion." Energies 12(19):3590, 2019.
      https://doi.org/10.3390/en12193590
      Section 3.3.3, second example (high-power-density fuel element,
      similar to the Ludewig et al. Brookhaven design), Table 5.
"""
import math

PARAMS_VERSION = "v1"

# ----------------------------------------------------------------------------
# Geometry  -- [2] Table 5
# ----------------------------------------------------------------------------
L_ELEMENT   = 0.6392     # m, fuel element length                      [2] T5
R_HOT_CHAN  = 0.02129    # m, hot gas channel radius                   [2] T5
T_HOT_FRIT  = 0.00300    # m, hot frit thickness                       [2] T5
T_BED       = 0.00615    # m, fuel bed thickness                       [2] T5
T_COLD_FRIT = 0.00200    # m, cold frit thickness                      [2] T5
T_PLENUM    = 0.00300    # m, inlet plenum thickness                   [2] T5
D_PARTICLE  = 0.0005     # m, fuel particle diameter                   [2] T5
EPS_BED     = 0.38       # -, fuel bed porosity                        [2] T5

# Derived radii (inside -> outside). Flow goes outside -> inside.
R_HOT_FRIT_IN   = R_HOT_CHAN
R_BED_IN        = R_HOT_FRIT_IN + T_HOT_FRIT
R_BED_OUT       = R_BED_IN + T_BED
R_COLD_FRIT_OUT = R_BED_OUT + T_COLD_FRIT

# ----------------------------------------------------------------------------
# Frits -- [2] Sec. 3.3.3, cold-frit-dominated element.
# [2] models frit pressure drop with the Ergun equation, using pore size
# in place of particle diameter and the frit's own porosity.
# ----------------------------------------------------------------------------
COLD_FRIT_PORE = 2.0e-6  # m  (0.002 mm)                               [2] 3.3.3
COLD_FRIT_EPS  = 0.40    # -                                           [2] 3.3.3
HOT_FRIT_PORE  = 1.0e-4  # m  (0.1 mm)                                 [2] 3.3.3
HOT_FRIT_EPS   = 0.50    # -                                           [2] 3.3.3

# ----------------------------------------------------------------------------
# Operating conditions
# ----------------------------------------------------------------------------
POWER_DENSITY = 30e6 * 1e3   # W/m^3  (30 MW/L)                        [2] 3.3.3
M_DOT_TOTAL   = 0.396        # kg/s, per element                       [2] 3.3.3
T_OUT_TARGET  = 3000.0       # K, "approaching 3000 K"                 [2] 3.3.3

# ASSUMPTION: [2] states 200 K inlet and 8.0 MPa operating pressure for its
# Example A (the large single core, Sec. 3.3.1), not explicitly for this
# fuel element. We assume the same values here. State this in limitations.
T_INLET = 200.0      # K                                     [2] 3.3.1 (assumed)
P_OP    = 8.0e6      # Pa                                    [2] 3.3.1 (assumed)

# ----------------------------------------------------------------------------
# Correlations used by [2] -- use the same ones so the comparison is fair
# ----------------------------------------------------------------------------
# Bed pressure loss: KTA 3102.3 friction factor
#   f = 320/(Re/(1-eps)) + 6/(Re/(1-eps))**0.1
# Bed heat transfer: KTA 3102.2
#   Nu = 1.27*Pr**(1/3)/eps**1.18 * Re**0.36 + 0.033*Pr**(1/2)/eps**1.07 * Re**0.86
# Frits: Ergun equation with pore size and frit porosity.
# Interfacial area density: A_sf = 6*(1-eps)/d_p

# ----------------------------------------------------------------------------
# Model resolution
# ----------------------------------------------------------------------------
N_SLICES = 20        # axial slices (parallel paths)
N_RADIAL = 50        # radial integration steps across the bed

# ----------------------------------------------------------------------------
# Validation targets from [2] -- these are COMPARISONS, never model inputs
# ----------------------------------------------------------------------------
VALIDATION_TARGETS = {
    "cold_frit_dominated_stable": True,          # [2] Fig. 13: converged
    "cold_frit_dominated_T_max_K": 3163.8,       # [2] 3.3.3
    "cold_frit_dominated_dP_total_kPa": 1189.2,  # [2] 3.3.3; includes plenum and
                                                 # hot channel losses, which a
                                                 # uniform-pressure model omits
    "hot_frit_dominated_stable": False,          # [2] 3.3.3: CFD diverged,
                                                 # T_max > 8000 K limit
}

# ----------------------------------------------------------------------------
# TO DO before freezing (needed only for the transient model)
# ----------------------------------------------------------------------------
FUEL_DENSITY = None   # kg/m^3  TO DO: source (fuel kernel/coating material)
FUEL_CP      = None   # J/kg-K  TO DO: source


def bed_volume():
    """Annular fuel bed volume, m^3."""
    return math.pi * (R_BED_OUT**2 - R_BED_IN**2) * L_ELEMENT


def total_power():
    """Element thermal power, W, from power density x bed volume."""
    return POWER_DENSITY * bed_volume()


if __name__ == "__main__":
    print(f"Radii (mm): hot channel {R_HOT_CHAN*1e3:.2f}, bed {R_BED_IN*1e3:.2f}"
          f"-{R_BED_OUT*1e3:.2f}, cold frit outer {R_COLD_FRIT_OUT*1e3:.2f}")
    print(f"Bed volume: {bed_volume()*1e3:.3f} L")
    print(f"Element power: {total_power()/1e6:.2f} MW")
