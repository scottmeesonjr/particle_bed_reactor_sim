"""
model.py -- adapter between the solver (slice.py, steady.py, stability.py) and
the code that uses it (runner.py, app.py). Owner: Scott.

This file contains NO physics. It only translates names and calls the solver.
Where the solver does not support something yet, it raises a clear error
instead of guessing (ground rule: never run a reduced form silently).

Provides:
  run_steady(case), find_onset(case, power_fractions, method), nominal_slice(element)
      -> the contract documented at the top of runner.py
  solve(...)
      -> the app contract below (for app.py)

APP CONTRACT -- solve() arguments and returned keys (used by app.py)

    solve(power_fraction, power_shape, cold_frit_on, cold_frit_porosity,
          use_point_a, hot_frit_pore, disturbance, disturbed_slice,
          channel_on, n_slices=20, n_radial=20) -> dict

Arguments
  power_fraction      fraction of nominal power, 0.05-1.00
  power_shape         "uniform" or "cosine" (params.power_shape names)
  cold_frit_on        False removes the cold frit entirely
  cold_frit_porosity  uniform cold frit porosity (ignored if use_point_a)
  use_point_a         True uses the Ji et al. point A porosity profile
  hot_frit_pore       hot frit pore size, m
  disturbance         extra power in one slice, fraction (0-0.30)
  disturbed_slice     index of that slice, 0-based
  channel_on          include hot-channel and plenum pressure effects
  n_slices, n_radial  grid

Returned dict
  "fake"              always False here
  "status"            "stable", "runaway", "out_of_range", "no_solution", or
                      "unknown" (steady solution found, stability not tested)
  "status_reason"     one plain sentence explaining the status
  "x"                 (n_slices,) slice centres along the element, m
  "r"                 (n_radial + 1,) bed node radii, outer to inner, m, or None
                      when radial profiles are unavailable (then gas_T, fuel_T are None)
  "m_dot"             (n_slices,) mass flow per slice, kg/s
  "gas_T", "fuel_T"   (n_slices, n_radial + 1) bed node temperatures, K, or None
  "gas_outlet_T"      (n_slices,) gas temperature leaving each slice, K
  "T_fuel_max_slice"  (n_slices,) peak fuel temperature in each slice, K
  "peak_fuel_T"       float, K
  "mixed_outlet_T"    float, flow-weighted outlet temperature, K
  "dp_total"          float, Pa
  "dp_cold_frit"      float, mean over slices, Pa
  "dp_hot_frit"       float, mean over slices, Pa
  "m_dot_total"       float, kg/s
  "flags"             list of plain-language range warnings (may be empty)
  "solve_time_s"      float, seconds
"""
import inspect
import time

import numpy as np

import params as P
import properties as pr
import slice as sl
import steady

try:
    import stability
except ImportError:            # not written yet
    stability = None

IS_FAKE = False

# steady.py supports channel=True (Z pattern; docs/model.md section 2, Amendment A2).
CHANNEL_AVAILABLE = True

N = P.N_SLICES
N_R = P.N_RADIAL

_NOMINAL_KEYS = dict(L=P.L_ELEMENT, r_hot_chan=P.R_HOT_CHAN, t_hot_frit=P.T_HOT_FRIT,
                     t_bed=P.T_BED, t_cold_frit=P.T_COLD_FRIT, d_particle=P.D_PARTICLE,
                     eps_bed=P.EPS_BED)

FLAG_TEXT = {
    "re_m_low": "Bed pressure-drop correlation is below its valid Reynolds range in some slices.",
    "re_ht_low": "Heat transfer correlation is below its valid Reynolds range at the hot face.",
    "fuel_over_limit": f"Fuel is above the {P.T_LIMIT:,.0f} K limit in some slices.",
}


def _supports(func, name):
    return name in inspect.signature(func).parameters


# ---------------------------------------------------------------------------
# Translating runner cases into steady.solve_element arguments
# ---------------------------------------------------------------------------
def _is_nominal(element):
    return all(np.isclose(element.get(k, np.nan), v) for k, v in _NOMINAL_KEYS.items())


def _geom(element):
    return None if _is_nominal(element) else element


def _cold_profile(cold_frit):
    if cold_frit == "published":
        return None                          # the element's own published cold frit
    if cold_frit == "none":
        return "none"
    if isinstance(cold_frit, dict) and "profile" in cold_frit:
        shape = cold_frit["profile"]
        return lambda s: (P.COLD_FRIT_PORE, float(P.cold_frit_porosity(s, **shape)))
    raise ValueError(f"Unknown cold_frit setting: {cold_frit!r}")


def _mass_flow(case, power_frac):
    m = case["m_dot"]
    if m == "rule_D2":
        if _geom(case["element"]) is not None:
            raise NotImplementedError("rule D2 mass flow is defined only for the nominal element")
        return P.nominal_mass_flow(power_frac)
    return float(m) * power_frac             # throttling: flow proportional to power


def _solve_case(case, power_frac=1.0, x0=None):
    return steady.solve_element(
        power_frac, _mass_flow(case, power_frac), geom=_geom(case["element"]),
        power_shape=case["power_shape"], cold_profile=_cold_profile(case["cold_frit"]),
        channel=case["channel_on"], pressure_ref=case["pressure_ref"],
        pattern=case.get("pattern", "Z"), x0=x0)


def _flag_list(res):
    return [FLAG_TEXT[k] for k in FLAG_TEXT if np.any(res["flags"].get(k, False))]


def _require_stability():
    if stability is None:
        raise NotImplementedError("stability.py is not written yet; stable/runaway cannot be decided")


def _point_stability(res):
    """Stability at one converged point (docs/model.md section 5a)."""
    _require_stability()
    static = stability.static_check(res)
    eigen = stability.eigen_check(res)
    return dict(ledinegg=bool(static["ledinegg"]), det_sign=int(static["det_sign"]),
                max_real=float(eigen["max_real"]), oscillatory=bool(eigen["oscillatory"]))


def _status_if_failed(res):
    return "out_of_range" if res["reason"] == "out_of_range" else "no_solution"


# ---------------------------------------------------------------------------
# runner.py contract
# ---------------------------------------------------------------------------
def run_steady(case):
    res = _solve_case(case)
    if not res["converged"]:
        return dict(status=_status_if_failed(res), dp_total=np.nan, peak_fuel_T=np.nan,
                    flags=_flag_list(res), message=res["message"])
    st = _point_stability(res)
    unstable = st["ledinegg"] or st["max_real"] > 0.0
    return dict(status="runaway" if unstable else "stable", dp_total=float(res["dp_element"]),
                peak_fuel_T=float(np.nanmax(res["T_fuel_max"])), flags=_flag_list(res),
                stability=st,
                beta=float(np.max(res["dp_bed"]) / np.min(res["dp_bed"])),       # A3: Energies Eq. 11
                flow_cov=float(np.std(res["m"]) / np.mean(res["m"])),
                flow_max_over_min=float(np.max(res["m"]) / np.min(res["m"])))


def find_onset(case, power_fractions, method):
    """Sweep power from high to low, each point starting from the previous one.
    method "slope": Ledinegg or determinant sign change since the first converged point.
    method "eigen": an eigenvalue with positive real part.
    (validation.md, amendment A1 item 4)"""
    if method not in ("slope", "eigen"):
        raise ValueError(f"method must be 'slope' or 'eigen' (got {method!r})")
    _require_stability()
    unstable, no_solution, x0, first_sign = [], [], None, None
    prev_f = None
    for f in sorted(power_fractions, reverse=True):
        res = _solve_case(case, power_frac=f, x0=x0)
        if not res["converged"] and prev_f is not None:
            res = _halve_toward(case, prev_f, x0, f)     # docs/model.md 3a, power sweeps
        if not res["converged"]:
            no_solution.append(float(f))
            continue
        prev_f, x0 = f, res["x"]
        st = _point_stability(res)
        if first_sign is None:
            first_sign = st["det_sign"]
        if method == "slope":
            bad = st["ledinegg"] or st["det_sign"] != first_sign
        else:
            bad = st["max_real"] > 0.0
        if bad:
            unstable.append(float(f))
    return dict(unstable_fractions=sorted(unstable), no_solution_fractions=sorted(no_solution),
                onset_fraction=max(unstable) if unstable else None)


def _halve_toward(case, f_from, x_from, f_target):
    """Reach f_target from the last converged sweep point by intermediate powers,
    halving the step on failure; give up below 1/256 of the original gap."""
    gap = f_from - f_target
    step, f_cur, x_cur, res = gap / 2.0, f_from, x_from, None
    while f_cur - f_target > 1e-12:
        f_try = max(f_target, f_cur - step)
        res = _solve_case(case, power_frac=f_try, x0=x_cur)
        if res["converged"]:
            f_cur, x_cur = f_try, res["x"]
        else:
            step /= 2.0
            if step < gap / 256.0:
                return res
    return res


def nominal_slice(element):
    """One slice of an element at its published nominal conditions (rule D8 input)."""
    g = element
    r_hf = g["r_hot_chan"] + g["t_hot_frit"]
    r_bed = r_hf + g["t_bed"]
    power = g["power_density"] * np.pi * (r_bed**2 - r_hf**2) * g["L"]
    res = sl.slice_dp(g["m_dot"] / N, power / N, g["P"], g["T_inlet"],
                      (g["cold_frit_pore"], g["cold_frit_eps"]),
                      (g["hot_frit_pore"], g["hot_frit_eps"]), g["L"] / N, N_R, geom=g)
    return res          # flags included; rules.py decides how to treat them


# ---------------------------------------------------------------------------
# app.py contract (see the module docstring)
# ---------------------------------------------------------------------------
def solve(power_fraction, power_shape, cold_frit_on, cold_frit_porosity,
          use_point_a, hot_frit_pore, disturbance, disturbed_slice,
          channel_on, n_slices=N, n_radial=N_R):
    t0 = time.perf_counter()
    if channel_on and not CHANNEL_AVAILABLE:
        raise NotImplementedError("The hot-channel and plenum model isn't built yet.")

    if not cold_frit_on:
        cold = "none"
    elif use_point_a:
        cold = lambda s: (P.COLD_FRIT_PORE, float(P.cold_frit_porosity(s, **P.POINT_A)))
    else:
        cold = lambda s: (P.COLD_FRIT_PORE, float(cold_frit_porosity))

    kwargs = {}
    if disturbance > 0:
        if not _supports(steady.solve_element, "power_multipliers"):
            raise NotImplementedError("The one-slice disturbance needs steady.py to accept "
                                      "power_multipliers (docs/model.md section 3d).")
        mult = np.ones(n_slices)
        mult[disturbed_slice] += disturbance
        kwargs["power_multipliers"] = mult

    m_total = P.nominal_mass_flow(power_fraction)
    res = steady.solve_element(power_fraction, m_total, power_shape=power_shape,
                               cold_profile=cold, hot=(hot_frit_pore, P.HOT_FRIT_EPS),
                               N=n_slices, n_r=n_radial, channel=channel_on, **kwargs)

    x = res["s"] * P.L_ELEMENT
    base = dict(fake=False, x=x, r=None, gas_T=None, fuel_T=None, m_dot_total=m_total,
                solve_time_s=0.0)

    if not res["converged"]:
        status = _status_if_failed(res)
        reason = ("Temperatures left the property range (above 5000 K)." if status == "out_of_range"
                  else "The solver found no steady solution for these settings.")
        nan = np.full(n_slices, np.nan)
        return dict(base, status=status, status_reason=reason, m_dot=res["m"], gas_outlet_T=nan,
                    T_fuel_max_slice=nan, peak_fuel_T=np.nan, mixed_outlet_T=np.nan,
                    dp_total=np.nan, dp_cold_frit=np.nan, dp_hot_frit=np.nan,
                    flags=[], solve_time_s=time.perf_counter() - t0)

    if stability is None:
        status, reason = "unknown", "Steady solution only. The stability test isn't built yet."
    else:
        st = _point_stability(res)
        if st["ledinegg"] or st["max_real"] > 0.0:
            status, reason = "runaway", "A small loss of flow in one slice grows instead of recovering."
        else:
            status, reason = "stable", "A small loss of flow in any slice dies out."

    # Radial profiles for the bed map, if slice.py exposes them (docs/model.md section 1a).
    gas_T = fuel_T = r = None
    if _supports(sl.slice_dp, "profile"):
        si = res["slice_inputs"]
        prof = [sl.slice_dp(res["m"][i], si["q"][i], si["p_slices"][i], si["T_in"], si["colds"][i], si["hot"],
                            si["dz"], si["n_r"], geom=si["geom"], profile=True) for i in range(n_slices)]
        r = np.asarray(prof[0]["r_nodes"])
        gas_T = np.array([p_["T_gas_nodes"] for p_ in prof])
        fuel_T = np.array([p_["T_fuel_nodes"] for p_ in prof])

    h_mix = np.sum(res["m"] * pr.h(res["T_gas_out"])) / np.sum(res["m"])
    return dict(base, status=status, status_reason=reason, r=r, gas_T=gas_T, fuel_T=fuel_T,
                m_dot=res["m"], gas_outlet_T=res["T_gas_out"], T_fuel_max_slice=res["T_fuel_max"],
                peak_fuel_T=float(np.nanmax(res["T_fuel_max"])),
                mixed_outlet_T=float(pr.T_from_h(h_mix)), dp_total=float(res["dp_element"]),
                dp_cold_frit=float(np.mean(res["dp_cold_frit"])),
                dp_hot_frit=float(np.mean(res["dp_hot_frit"])),
                flags=_flag_list(res), solve_time_s=time.perf_counter() - t0)