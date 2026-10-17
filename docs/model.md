# Model specification (implement exactly; frozen with v3 decisions)

## 0. Conventions
- SI units. Pressure drop positive = pressure falls along the flow.
- Slice i = 0 at the bottom (z = 0, where the Z-pattern plenum is fed); channel exits at z = L.
- Geometry, frit values, limits, and rules come from params.py (read the actual names there).
  Nominal element (Ji et al. 2019): hot channel radius 25 mm; hot frit 25-27 mm; bed 27-37 mm;
  cold frit 37-39 mm; plenum 39-41 mm; L = 0.65 m; d_p = 0.5 mm; bed porosity alpha = 0.38.
- Properties come from properties.py as frozen: h(T), T_from_h(h), cp(T), mu(T), k(T), rho(T, P).
  Only rho takes pressure. PropertyRangeError is raised outside the data range: catch it, flag it,
  never extrapolate.

## 1. One slice: slice_dp(m_dot, q_slice, p_in, T_in, cold, hot, dz, n_r, geom=None)
Inputs: m_dot = slice mass flow (kg/s); q_slice = slice power (W); p_in = pressure at the cold frit
outer face (Pa); T_in (K); cold = None (no cold frit) or (pore_size_m, porosity);
hot = (pore_size_m, porosity); dz = slice height (m); n_r = steps per layer;
geom = None (nominal element from params.py) or a geometry dict defined in params.py
(e.g. params.T19_ELEMENT). Geometry values are never defined outside params.py.

Definitions:
- q''' = q_slice / V_bed_slice, with V_bed_slice = pi * (R_bed_outer^2 - R_hotfrit_outer^2) * dz
  (power per BED volume, consistent with the MW/L power density).
- Superficial mass flux G(r) = m_dot / (2 * pi * r * dz); superficial velocity u = G / rho.
- s = distance marched inward. State = (p, h). March each layer with midpoint (RK2) steps,
  n_r uniform steps in r, evaluating properties at the midpoint.

Layer 1, cold frit (R_coldfrit_outer -> R_bed_outer). Adiabatic: T = T_in throughout.
- If cold is None: skip (zero pressure drop).
- Full Ergun with d = pore size, eps = frit porosity:
  dp/ds = 150 * mu * (1-eps)^2 * u / (eps^3 * d^2) + 1.75 * rho * (1-eps) * u^2 / (eps^3 * d)

Layer 2, bed (R_bed_outer -> R_hotfrit_outer). Heat released only here.
- Pressure (KTA 3102.3): dp/ds = f * (1-alpha)/alpha^3 * G^2 / (2 * rho * d_p)
  with f = 320 / (Re/(1-alpha)) + 6 / (Re/(1-alpha))^0.1 and Re = G * d_p / mu.
- Energy: dh/ds = q''' * 2 * pi * r * dz / m_dot; gas T = T_from_h(h).
- Fuel (particle surface) temperature at each of the n_r + 1 bed nodes (not the midpoints): T_f = T_g + q''' / (h_c * a)
  with a = 6 * (1-alpha) / d_p, h_c = Nu * k / d_p, Pr = mu * cp / k, and (KTA 3102.2)
  Nu = 1.27 * Pr^(1/3) * Re^0.36 / alpha^1.18 + 0.033 * Pr^(1/2) * Re^0.86 / alpha^1.07.
  Particle internal conduction is neglected (stated limitation).

Layer 3, hot frit (R_hotfrit_outer -> R_channel). Adiabatic: T = bed outlet temperature.
- Full Ergun as in layer 1 with the hot frit's pore size and porosity.

Returns: dp (= p_in - p at channel face), T_gas_out, T_fuel_max, p_out, and flags:
- re_m_low: any bed node with Re/(1-alpha) < 10 (KTA pressure-drop range)
- re_ht_low: any bed node with Re < 100 (KTA heat-transfer range)
- out_of_range: PropertyRangeError caught; stop the march, return NaN values with this flag
- pressure_collapse: pressure reaches 0 or below anywhere in the march; stop, return NaN
  values with this flag
- fuel_over_limit: T_fuel_max > params.T_LIMIT
Also returned: dp_cold_frit, dp_bed, dp_hot_frit (per-layer drops, Pa; 0 for no cold frit).
Sensitivity cases (not default): Ergun for the bed; Wakao-Kaguei Nu = 2 + 1.1 Pr^(1/3) Re^0.6.

## 2. Plenum and channel
- Z pattern: plenum fed at z = 0, channel exits at z = L, both flow upward.
- Plenum flow W_pl(z) = m_total - (slice flows already drawn); channel flow W_ch(z) = sum of slice
  flows below z.
- Between slice centers, in the flow direction:
  dp = -Delta(W^2 / (rho * A^2)) - f_D * W^2 / (2 * rho * A^2 * D_h) * dz
  Channel: A = pi * R_ch^2, D_h = 2 * R_ch; entering gas carries no axial momentum.
  Plenum: A = pi * (R_el^2 - R_cf^2), D_h = 2 * (R_el - R_cf); leaving gas takes its momentum
  (full pressure recovery; stated assumption).
  Petukhov: f_D = (0.790 * ln(Re) - 1.64)^-2, Re = W * D_h / (A * mu).
- Channel temperature: enthalpy-weighted mix of slice outflows.
- Channel marched down from p(L) = P_OUTLET; plenum marched up from unknown p_in.

## 3. Full element: N + 1 system
- Unknowns (scaled): x_i = m_i / m_avg (i = 1..N); x_N+1 = (p_in - P_OUTLET) / 5e5.
- Residuals: R_i = [p_pl(z_i) - p_ch(z_i) - dp_slice_i] / 5e5;
  R_N+1 = (sum(m_i) - m_total) / m_total.
- channel=False: p_ch = P_OUTLET and p_pl = p_in everywhere (all slices share one pressure drop).
- Solve with scipy.optimize.root (hybr); continuation in power from the previous converged point.
- Slice power: q_i = element power * power_frac * F_i / N, where F_i is params.power_shape
  averaged over slice i at 50 evenly spaced points inside it (midpoints of 50 equal
  sub-intervals), and the N values F_i are then renormalized to mean exactly 1.
- Cold frit per slice: cold_profile = None (published constant cold frit), "none" (no cold
  frit), or a callable cold_profile(s_i) -> (pore, porosity), with s_i = slice-centre z_i / L.
  Ji et al. profile: pore = params.COLD_FRIT_PORE, porosity = params.cold_frit_porosity(s_i, ...).
- Stated limitation: the model cannot represent reverse flow in a slice. A trial or solution
  with m_i <= 0 stops the solve with reason "nonpositive_flow" and is reported as
  "no solution found".

## 4. Mass flow
- Nominal: params.nominal_mass_flow() (rule D2). Throttled: m(P) = m_nom * P / P_nom.
- Validation runs use the published flows only.

## 5. Stability
- Static: at each converged point, finite-difference Jacobian of the residuals;
  numpy.linalg.slogdet sign stored; a sign change between power steps = fold.
  Also flag any slice with d(dp_slice)/d(m) < 0 (Ledinegg).
- Dynamic (transient.py): one state per slice, the outlet temperature T_out,i:
  C_i * dT_out,i/dt = Q_i - m_i * (h(T_out,i) - h_in),
  C_i = rho_fuel * fuel_cp * (1-alpha) * V_bed_slice * 0.5 (linear-profile assumption).
  Fuel cp proxy (ZrC) is invalid above 3805 K (melting); flag there.
- Eigenvalues of the finite-difference Jacobian of dT/dt; positive real part = unstable.
  Report static and oscillatory results separately.

## 6. Ji et al. cold frit porosity profile
eps(x) = (a * (x - b*L)^2 + c) * alpha0, a = 3 * (1-c) / ((b^3 + (1-b)^3) * L^2);
reflector terms zero for the bare element. Point A: b = 0.434, c = 1.070, alpha0 = 0.533.

## 7. Frit-share sweep (Q1)
Vary pore size; report the resulting frit share
(frit pressure drop / total element pressure drop at nominal steady state).
"No cold frit" = cold = None.

## Amendment A1 (pre-run decisions; implement before any validation run)

### 3a. Continuation and failure rule (item 2)
A target solve must not report "no solution" because one solver trial point failed.
- Homotopy parameter lambda from 0 to 1. At lambda, slice power uses the shape
  F_lambda = 1 + lambda * (F_target - 1) and multipliers 1 + lambda * (mult - 1). At lambda = 0 every
  slice has the same power, so with channel=False the equal-flow solution exists for uniform frits.
- Step 1: solve the target directly (lambda = 1) from the equal-flow guess. If it converges, done.
- Step 2: otherwise solve lambda = 0, then advance lambda with step 0.25, each solve starting from the
  last converged x. On any failure, halve the step and retry from the last converged point.
- If the step falls below 1/256 without reaching lambda = 1, stop: "no solution found
  (continuation exhausted at lambda = ...)", with the reason of the last failed attempt and the last
  converged state (lambda, flows, peak temperatures).
- Power sweeps use the same halving rule with power fraction as the parameter: each sweep point
  starts from the previous converged point; on failure, insert intermediate powers by halving.
- Results record which path was used (direct or continuation) and the final step size.

### 3b. Element-specific inputs (item 3)
For geom = a params.py element dict (params.T19_ELEMENT, or the rules.py D8 variant of it):
- L = geom["L"], dz = L / N. Radii from geom as in section 1.
- Element power = geom["power_density"] * pi * (r_bed_out^2 - r_hotfrit_out^2) * L.
- T_in = geom["T_inlet"]; reference pressure = geom["P"].
- cold_profile=None means THIS element's published cold frit: (geom["cold_frit_pore"],
  geom["cold_frit_eps"]). Default hot frit: (geom["hot_frit_pore"], geom["hot_frit_eps"]).
- geom=None: all of the above from params.py (unchanged).
- T19_LARGE_REACTOR is Tier 2 and is not decided here.

### 3c. Pressure reference (item 3)
- pressure_ref="outlet" (default): channel exit pressure = reference pressure (as now).
- pressure_ref="inlet": plenum inlet pressure at z = 0 = reference pressure. The unknown becomes the
  outlet: x_N = (p_ref - p_out) / P_SCALE. With channel=False every slice has p_in = p_ref and
  p_out = p_ref - x_N * P_SCALE.

### 3d. Per-slice power multipliers (app disturbance)
- Optional power_multipliers (length N, default all ones), applied after the shape is renormalized:
  q_i = element power * power_frac * F_i * mult_i / N. A disturbance adds power; total power rises.

### 1a. Radial profile output (app only)
- slice_dp(..., profile=False). If True, also return r_nodes, T_gas_nodes, T_fuel_nodes at the
  n_r + 1 bed nodes, ordered outer to inner. Values are unchanged; this only exposes them.

### 5a. stability.py interface (used by model.py)
Both take the dict returned by steady.solve_element at a converged point.
- static_check(result) -> {"ledinegg": bool, "det_sign": +1 or -1}
  ledinegg = any slice with d(dp_slice_i)/d(m_i) < 0; det_sign = sign of slogdet of the residual
  Jacobian.
- C_i in section 5 uses fuel_cp at T_mean,i = (T_in + T_out,i) / 2, the mean of the linear profile.
- eigen_check(result) -> {"max_real": float, "oscillatory": bool}
  max_real = largest real part of the transient-model Jacobian eigenvalues (section 5);
  oscillatory = that eigenvalue has a nonzero imaginary part.