3d0ffee
# Validation criteria (pre-registered)

Committed with `params.py`, `properties.py`, `rules.py`, and `data/` in one freeze commit, before any validation run. Every result file records the commit hash. Results are never used to change anything in this file or the frozen code.

## Decision D13 (added before freeze): property limit and V6

The v3 text of V6 passes on "no steady solution below 8000 K." The sourced property data stop below that: the Svehla transport fits end at 5000 K, the JANAF H₂ table at 6000 K, and hydrogen dissociation (not modeled) becomes important in that range. Therefore:

- `properties.py` covers 150–5000 K and raises `PropertyRangeError` above 5000 K.
- The solver reports any case that leaves the property range as **"no steady solution within the property range."**
- V6 is amended to: *instability, or no steady solution with all temperatures below the property limit (5000 K).*

This was decided from the data's range, before any result existed.

## Criteria (v3 Section 6)

All runs use the source's stated inputs. Because of the systematic enthalpy difference (D2), temperature criteria compare directions and relative changes, not absolute values.

| ID | Tier | Target | Pass criterion |
| V5 | 1 | T19 element, cold-frit-dominated: Δp = 1189.2 kPa | Within ±25%; stable |
| V6 | 1 | T19 element, hot-frit-dominated (rule D8) | Instability, or no steady solution with all temperatures below 5000 K (D13) |
| V7 | 1 | No cold frit at design exit temperature [K93] | Static instability below a power density within a factor of 3 of 4 MW/L |
| V8 | 1 | Strong cold frit [K93] | V7 instability disappears |
| V9 | 1 | Ledinegg check | Eigenvalue and pressure-drop-slope onsets agree within one sweep step |
| V10 | 1 | J19b preliminary design, uniform power, 0.505 kg/s: Δp = 529.8 kPa | Within ±25% |
| V11 | 1 | Cosine vs. uniform power: peak temperature rises (3348.4 → 3905.5 K); Δp essentially unchanged | Peak temperature rises; Δp changes by less than 2% |
| V12 | 1 | J19b point A, cosine power: peak temperature and Δp both fall (to 3147.6 K and 414.4 kPa) | Both fall; Δp within ±25% of 414.4 kPa |
| V1 | 2 | T19 larger reactor, Z: Δp = 872.4 kPa | Within ±25% |
| V2 | 2 | Z vs. U | U lower peak temperature; Δp changes by less than 2% |
| V3 | 2 | Core heights 80/100/120 cm | Δp within ±25% each; β decreases with shorter core |
| V4 | 2 | Frit-dominance variants | Both lower peak temperature vs. baseline |
| V13 | 3 | Reproduction-mode optimizer, cosine power | Chosen design has porosity peak upstream of the power peak (b < 0.5) |

Expected disagreements to report, not fix: absolute temperatures (D2); 1D model vs. 2D CFD; parallel-passage limitation at low Reynolds number [K93]; J19b's CFD treats the flow as incompressible, while our model includes density change.

## Fallback rule

If the channel and plenum model is not passing its tests at the 7:30 checkpoint, V5, V10, V11, and V12 are marked **"not run."** They are never run in a reduced form. V6–V9, Q1, and Q2 continue with channel effects off, and the app says so.

## Pressure-reference rule

The Tsinghua (T19) cases state 8.0 MPa without saying where it applies. Each T19 case is run twice: once with 8.0 MPa at the element outlet and once at the inlet. Both results are reported.

## Correlation range flags

Runs are flagged, never stopped, when the bed's modified Reynolds number Re_m = Re/(1 − ε) falls below 10 (KTA pressure drop; range 10 < Re_m < 10⁵ per Reger et al., arXiv:2203.05041) or Re falls below 100 (KTA heat transfer).

## Hypotheses

| ID | Hypothesis | Source |
| --- | --- | --- |
| H1 | With no cold frit, a slice's pressure drop vs. flow curve has a minimum. With the published cold frit, it rises steadily. | Kerrebrock; Tsinghua |
| H2 | With no cold frit at a 3000 K outlet, static instability appears below about 4 MW/L. | Kerrebrock |
| H3 | A sufficient cold frit removes the H2 instability. | Kerrebrock |
| H4 | At 20–30 MW/L with 0.5 mm particles, a hot-frit-dominated element is unstable and a cold-frit-dominated one is stable. | Tsinghua |
| H5 | Cosine power raises peak temperature over uniform, with almost no change in pressure drop. | Ji et al. |
| H6 | Point A lowers both peak temperature and pressure drop under cosine power. | Ji et al. |
| H7 | U reduces flow nonuniformity vs. Z (Tier 2). | Tsinghua |
| Q4 | Does point A stay stable from 20% to 100% throttle? No prediction. | — |
