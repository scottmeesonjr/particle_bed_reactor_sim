"""
fake.py -- FAKE solver output for building the app before the real solver exists.

Ground rule 7: fake data lives only in this file, and this file is deleted
once the real solver works. Nothing here is physics. The numbers only move
with the sliders so the app can be built and tested.

THE INTERFACE BELOW IS THE CONTRACT. The real solver must provide a function
with the same name, arguments, and returned keys, so swapping it in is a
one-line change in app.py:

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
  "fake"            True here; the real solver returns False
  "status"          "stable", "runaway", or "out_of_range"
  "status_reason"   one plain sentence explaining the status
  "x"               (n_slices,) slice centers along the element, m
  "r"               (n_radial,) radial positions across the bed, outer to inner, m
  "m_dot"           (n_slices,) mass flow per slice, kg/s
  "gas_T"           (n_slices, n_radial) gas temperature, K
  "fuel_T"          (n_slices, n_radial) fuel temperature, K
  "gas_outlet_T"    (n_slices,) gas temperature leaving each slice, K
  "peak_fuel_T"     float, K
  "mixed_outlet_T"  float, flow-weighted outlet temperature, K
  "dp_total"        float, Pa
  "dp_cold_frit"    float, Pa
  "dp_hot_frit"     float, Pa
  "m_dot_total"     float, kg/s
  "flags"           list of plain-language range warnings (may be empty)
  "solve_time_s"    float, seconds
"""
import time
import numpy as np

import params as p

IS_FAKE = True


def solve(power_fraction, power_shape, cold_frit_on, cold_frit_porosity,
          use_point_a, hot_frit_pore, disturbance, disturbed_slice,
          channel_on, n_slices=p.N_SLICES, n_radial=p.N_RADIAL):
    t0 = time.perf_counter()

    s = (np.arange(n_slices) + 0.5) / n_slices
    x = s * p.L_ELEMENT
    r = np.linspace(p.R_BED_OUT, p.R_BED_IN, n_radial)

    # FAKE: power per slice from the real shape function, plus the disturbance
    q = p.power_shape(power_shape, s)
    q[disturbed_slice] *= 1.0 + disturbance

    # FAKE: flow split that leans away from hot slices more when the cold frit is weak
    if use_point_a:
        frit_strength = 1.0
    elif cold_frit_on:
        frit_strength = np.clip(cold_frit_porosity / 0.5, 0.2, 2.0) ** -1
    else:
        frit_strength = 0.05
    lean = 0.6 / (1.0 + 4.0 * frit_strength)
    weights = q ** (1.0 - lean)
    if channel_on:
        weights *= 1.0 + 0.08 * (s - 0.5)          # FAKE flow-shift tilt
    m_total = 0.575 * power_fraction
    m_dot = m_total * weights / weights.sum()

    # FAKE: temperatures rise with power per unit flow
    heat_ratio = (q / q.mean()) / (m_dot / m_dot.mean())
    gas_out = 200.0 + 2800.0 * heat_ratio
    depth = np.linspace(0.0, 1.0, n_radial)
    gas_T = 200.0 + (gas_out[:, None] - 200.0) * depth[None, :] ** 1.3
    fuel_T = gas_T + 120.0 * heat_ratio[:, None] * (0.6 + 0.4 * depth[None, :])

    peak = float(fuel_T.max())
    mixed = float(np.sum(m_dot * gas_out) / m_total)

    # FAKE: a status rule that only exists so the label can be tested
    weak_frit = (not cold_frit_on) or (not use_point_a and cold_frit_porosity > 0.62)
    if peak > 5000.0:
        status, reason = "out_of_range", "Temperatures left the property range (above 5000 K)."
    elif weak_frit and power_fraction < 0.3:
        status, reason = "runaway", "Fake rule: weak cold frit at low power."
    else:
        status, reason = "stable", "Fake rule: no runaway condition met."

    dp_cold = (0.0 if not cold_frit_on else 380e3 * (0.5 / max(cold_frit_porosity, 0.05)) ** 3) * power_fraction
    dp_hot = 45e3 * (1e-4 / hot_frit_pore) ** 2 * power_fraction
    dp_bed = 90e3 * power_fraction ** 1.5
    dp_total = dp_cold + dp_hot + dp_bed + (25e3 * power_fraction ** 2 if channel_on else 0.0)

    flags = []
    if power_fraction < 0.15:
        flags.append("Fake flag: bed pressure-drop correlation outside its range at the hot face.")

    return {
        "fake": True, "status": status, "status_reason": reason,
        "x": x, "r": r, "m_dot": m_dot, "gas_T": gas_T, "fuel_T": fuel_T,
        "gas_outlet_T": gas_out, "peak_fuel_T": peak, "mixed_outlet_T": mixed,
        "dp_total": dp_total, "dp_cold_frit": dp_cold, "dp_hot_frit": dp_hot,
        "m_dot_total": m_total, "flags": flags,
        "solve_time_s": time.perf_counter() - t0,
    }
