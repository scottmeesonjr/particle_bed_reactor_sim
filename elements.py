"""
elements.py -- element descriptions built by rule from params.py (docs/model.md A3).

large_reactor(L, variant): the Tsinghua large single-core reactor (Energies 2019,
Example A/B/C), as an element dict that steady.py and slice.py accept.
  - Geometry from params.T19_LARGE_REACTOR. The 111 mm radial reflector lies outside
    the plenum and is not modeled (507 mm core radius = 396 mm plenum outer + 111 mm).
  - Power density held at the L = 1.2 m value, 1000 MW / bed volume = 1.938 MW/L
    (the paper keeps power density unchanged when shortening the core).
  - Mass flow paired with length as published: 1.2 m -> 21 kg/s, 1.0 -> 17.5, 0.8 -> 14.
  - Variants (Example C): "baseline" = 4 mm pores, cold porosity 0.1, hot 0.2;
    "cold_dominated" = cold frit 10 um pores at porosity 0.2;
    "hot_dominated" = hot frit 400 um pores (porosity unchanged).
"""
import math

import params as P

R = P.T19_LARGE_REACTOR
VARIANTS = ("baseline", "cold_dominated", "hot_dominated")


def _bed_volume(L):
    r_hf = R["r_hot_chan"] + R["t_hot_frit"]
    return math.pi * ((r_hf + R["t_bed"]) ** 2 - r_hf ** 2) * L


POWER_DENSITY_LARGE = R["power"] / _bed_volume(1.2)        # W/m^3 (1.938 MW/L)


def large_reactor(L=1.2, variant="baseline"):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    flows = dict(zip(R["L_options"], R["m_dot_options"]))
    if L not in flows:
        raise ValueError(f"L must be one of {sorted(flows)} m (published cases)")
    cold = (R["frit_pore"], R["cold_frit_eps"])
    hot = (R["frit_pore"], R["hot_frit_eps"])
    if variant == "cold_dominated":
        cold = (10e-6, 0.2)
    elif variant == "hot_dominated":
        hot = (400e-6, R["hot_frit_eps"])
    return dict(L=L, r_hot_chan=R["r_hot_chan"], t_hot_frit=R["t_hot_frit"], t_bed=R["t_bed"],
                t_cold_frit=R["t_cold_frit"], t_plenum=R["t_plenum"], d_particle=R["d_particle"],
                eps_bed=R["eps_bed"], cold_frit_pore=cold[0], cold_frit_eps=cold[1],
                hot_frit_pore=hot[0], hot_frit_eps=hot[1], power_density=POWER_DENSITY_LARGE,
                m_dot=flows[L], T_inlet=R["T_inlet"], P=R["P"], name=f"large reactor L={L} m ({variant})")
