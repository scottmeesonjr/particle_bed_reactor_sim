"""
rules.py -- frozen rules that need the slice model to compute their numbers.

The CODE here is frozen with params.py. The NUMBERS it produces are computed
later, once slice.py exists, and saved to results/ with the commit hash.

Rule D8 (hot-frit-dominated variant, used by validation target V6):
  1. Run the element's published design at nominal steady state
     (uniform power, its own mass flow).
  2. Record the cold and hot frit pressure drops, dp_c and dp_h.
  3. Swap them: find new pore sizes so the cold frit gives dp_h and the
     hot frit gives dp_c, each at its own nominal conditions.
  4. Porosities, thicknesses, and everything else stay unchanged.

Contract with slice.py: slice_fn(element) must return a dict containing at least
  "dp_cold_frit" and "dp_hot_frit" (Pa) for that element at nominal steady state.
`element` is a dict like params.T19_ELEMENT.
"""
import numpy as np
from scipy.optimize import brentq


def _solve_pore(element, slice_fn, key_pore, key_dp, target_dp):
    """Pore size (m) that gives target_dp for one frit, searched in log space."""
    def resid(log_d):
        trial = dict(element, **{key_pore: float(np.exp(log_d))})
        return np.log(slice_fn(trial)[key_dp]) - np.log(target_dp)

    lo, hi = np.log(1e-8), np.log(1e-2)        # 10 nm to 1 cm
    if resid(lo) * resid(hi) > 0:
        raise RuntimeError(f"D8: no pore size between 10 nm and 1 cm gives {key_dp} = {target_dp:.3g} Pa")
    return float(np.exp(brentq(resid, lo, hi, xtol=1e-10)))


def hot_frit_dominated(element, slice_fn, sweeps=3):
    """Rule D8. Returns (new_element, report). Two frits interact weakly through
    the gas temperature, so the swap is repeated `sweeps` times."""
    base = slice_fn(element)
    dp_c0, dp_h0 = base["dp_cold_frit"], base["dp_hot_frit"]
    new = dict(element)
    for _ in range(sweeps):
        new["cold_frit_pore"] = _solve_pore(new, slice_fn, "cold_frit_pore", "dp_cold_frit", dp_h0)
        new["hot_frit_pore"] = _solve_pore(new, slice_fn, "hot_frit_pore", "dp_hot_frit", dp_c0)
    after = slice_fn(new)
    report = dict(
        original=dict(dp_cold_frit=dp_c0, dp_hot_frit=dp_h0,
                      cold_share=dp_c0 / (dp_c0 + dp_h0)),
        swapped=dict(dp_cold_frit=after["dp_cold_frit"], dp_hot_frit=after["dp_hot_frit"],
                     cold_share=after["dp_cold_frit"] / (after["dp_cold_frit"] + after["dp_hot_frit"])),
        cold_frit_pore=new["cold_frit_pore"], hot_frit_pore=new["hot_frit_pore"],
    )
    return new, report


if __name__ == "__main__":
    # Self-test with a STAND-IN slice function (viscous Ergun at fixed gas
    # temperatures). This only checks that the rule's logic works; the real
    # numbers come from slice.py.
    import params as p
    import properties as pr

    def stand_in_slice(el):
        out = {}
        area_c = 2 * np.pi * (el["r_hot_chan"] + el["t_hot_frit"] + el["t_bed"] + el["t_cold_frit"]) * el["L"]
        area_h = 2 * np.pi * (el["r_hot_chan"] + el["t_hot_frit"]) * el["L"]
        for side, T, area in (("cold", el["T_inlet"], area_c), ("hot", 3000.0, area_h)):
            e, d, t = el[f"{side}_frit_eps"], el[f"{side}_frit_pore"], el[f"t_{side}_frit"]
            u = el["m_dot"] / (pr.rho(T, el["P"]) * area)
            out[f"dp_{side}_frit"] = 150 * pr.mu(T) * (1 - e)**2 / (e**3 * d**2) * u * t
        return out

    new, rep = hot_frit_dominated(p.T19_ELEMENT, stand_in_slice)
    print("STAND-IN self-test (not real results):")
    print(f"  original cold share {rep['original']['cold_share']:.3f} -> swapped {rep['swapped']['cold_share']:.3f}")
    print(f"  pores: cold {p.T19_ELEMENT['cold_frit_pore']:.2e} -> {rep['cold_frit_pore']:.2e} m, "
          f"hot {p.T19_ELEMENT['hot_frit_pore']:.2e} -> {rep['hot_frit_pore']:.2e} m")