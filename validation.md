# Validation Criteria — Particle Bed Stability Simulator
Committed before any validation run. Every result file records the commit hash.
No input or model choice may be changed to improve agreement with these targets.

## Definitions (fixed now)
- "No cold frit": cold frit layer removed (zero pressure drop).
- "Strong cold frit": the element's published cold frit (J19b: 2 µm, porosity 0.5; T19: 2 µm, porosity 0.4).
- Frit share: pressure drop across that frit / total element pressure drop, at nominal steady state.
- Power sweep: 5%–100% of nominal in steps of 2.5%. Mass flow scaled with power (3000 K outlet rule).
- Unstable: any eigenvalue with positive real part, OR a sign change of the Jacobian determinant
  along the continuation path. "No converged solution" is reported separately.
- Static and oscillatory instabilities are reported separately.

## Targets
| ID | Tier | Target | Pass criterion |
|----|------|--------|----------------|
| V1 | 2 | T19 larger reactor, Z: Δp = 872.4 kPa | Within ±25% |
| V2 | 2 | Z vs U: peak temperature falls, Δp nearly unchanged | U lower peak temperature; Δp changes < 2% |
| V3 | 2 | Core heights 80/100/120 cm | Δp within ±25% each; β decreases with shorter core |
| V4 | 2 | T19 frit-dominance variants | Both lower peak temperature vs baseline |
| V5 | 1 | T19 element, cold-frit-dominated: Δp = 1189.2 kPa | Within ±25%; stable |
| V6 | 1 | T19 element, hot-frit-dominated (D8 rule) | Instability, or no steady solution below 8000 K |
| V7 | 1 | No cold frit at design exit temperature (Kerrebrock) | Static instability below a power density within a factor of 3 of 4 MW/L |
| V8 | 1 | Strong cold frit (Kerrebrock) | V7 instability disappears |
| V9 | 1 | Ledinegg check | Eigenvalue onset and pressure-drop-slope onset agree within one sweep step |
| V10 | 1 | J19b preliminary design, uniform power, 0.505 kg/s: Δp = 529.8 kPa | Within ±25% |
| V11 | 1 | Cosine vs uniform: peak temp rises (3348.4 → 3905.5 K), Δp essentially unchanged | Peak temperature rises; Δp changes < 2% |
| V12 | 1 | J19b point A, cosine power: peak temp and Δp fall (to 3147.6 K, 414.4 kPa) | Both fall vs our cosine baseline; Δp within ±25% of 414.4 kPa |
| V13 | 3 | Reproduction-mode NSGA-II, cosine power | Chosen design has porosity peak upstream of power peak (b < 0.5) |

## Input rules for validation runs
- Published inputs exactly (J19b: 0.505 kg/s, 20 MW/L, 200 K inlet, 7.0 MPa outlet, Z pattern).
- T19 cases: run twice, with 8.0 MPa at the outlet and 8.0 MPa at the inlet; report both.
- Temperature criteria compare directions and changes only (systematic enthalpy difference, v3 D2).

## Fallback (fixed now)
If the channel and plenum model is not passing its tests by hour 7:30, V5, V10, V11, and V12
are marked NOT RUN. V6–V9 run with channel effects off, and the app states this.

## Hypotheses (recorded either way)
- H1: No cold frit → slice Δp-vs-flow curve has a minimum; published cold frit → rises steadily. (K93, T19)
- H2: No cold frit at 3000 K outlet → static instability below about 4 MW/L. (K93)
- H3: A sufficient cold frit removes the H2 instability. (K93)
- H4: At 20–30 MW/L with 0.5 mm particles, hot-frit-dominated is unstable; cold-frit-dominated is stable. (T19)
- H5: Cosine power raises peak temperature with almost no Δp change. (J19b)
- H6: Point A lowers peak temperature and Δp under cosine power. (J19b)
- H7: U reduces nonuniformity vs Z. (T19, Tier 2)
- Q4: No hypothesis — whether point A stays stable over 20–100% throttle is open.

## Known expected disagreements (report, do not fix)
Absolute temperatures; 1D model vs 2D CFD; parallel-passage limitation at low Reynolds number (K93);
J19b CFD treats flow as incompressible.

## Pre-run amendment A1 (committed before any validation run)

These clarify how the criteria above are applied. No validation result existed when they were written.

**1. V6 and the property limit (decision D13).** The sourced property data end at 5000 K (Svehla 1995 transport fits; NIST-JANAF H2 to 6000 K), and hydrogen dissociation, which the model does not include, becomes important in that range. The model therefore cannot evaluate states above 5000 K. V6's "no steady solution below 8000 K" is applied as: instability, or no steady solution with all temperatures below 5000 K. A case that leaves the property range is reported as "no steady solution within the property range." The original 8000 K wording above is kept for the record.
**2. "No steady solution" must survive the continuation rule.** A solve counts as "no steady solution" only after the continuation rule in docs/model.md section 3a is exhausted. A failure at a single solver trial point is not a result.
**3. Sweep step.** Sweeps use the 2.5% step defined above. runner.py previously used 2%; that mismatch was fixed before any run.
**4. Static vs. eigenvalue onsets (V7, V9).** The static onset is the highest swept power at which any slice has d(Δp_slice)/d(m) < 0 (Ledinegg) or the residual Jacobian determinant has changed sign since the first converged sweep point. The eigenvalue onset is the highest swept power with an eigenvalue of positive real part. Points without a converged solution are listed separately and are not counted as unstable.