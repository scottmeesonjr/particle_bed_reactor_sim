# H1 outcome (slice curve, commit f0401da)
Hypothesis H1 (validation.md): with no cold frit, the slice dp-vs-flow curve has a
minimum; with the published cold frit, dp rises steadily.

Setup: one nominal slice, fixed power 1.307 MW, p_in 7.0 MPa, T_in 200 K, n_r 20;
60 flows from 0.05x to 3x nominal.

Result:
- No cold frit: interior minimum at 26.43 g/s (0.919x nominal), dp 50.4 kPa. Shallow:
  +7% at the lowest valid flow, +58% at 3x nominal.
- Published cold frit: dp rises at every valid point; nominal dp 561.0 kPa
  (cold frit 90% of it).
- At 0.529x nominal flow and below, the gas exceeds 5000 K
  (crossing between 0.529x and 0.567x). Behavior there is unknown.
- fuel_over_limit (> 3800 K) from 0.567x to 0.749x nominal; re_ht_low at 14 of 25 points.

Verdict: H1 holds within the property-data range.

Note: without a cold frit, the nominal point (1.0x) sits just right of the minimum,
near the zero-slope point. This is a single-slice, fixed-power result; whether the
element is unstable is decided by the full-element tests (V7-V9), not by this plot.
