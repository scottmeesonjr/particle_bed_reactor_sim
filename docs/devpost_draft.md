# Devpost draft: Can a Frit Stop a Runaway?

*Draft. [ANTHONIE: ...] marks text Anthonie fills in. Numbers come from the official Tier 1 run at commit 89ef317; don't change them by hand.*

## Inspiration
A particle bed nuclear rocket heats hydrogen by pushing it through a bed of tiny, very hot fuel particles. The design has a known failure mode: if one region runs a little hot, the hydrogen there gets harder to push, less of it flows, and that region gets hotter still until the fuel fails. A porous wall called a cold frit is meant to stop this. We wanted to know whether it actually does, across the whole throttle range, using only published numbers.

## What it does
The simulator splits one published fuel element into 20 parallel flow paths and solves for how the hydrogen divides between them, including how pressure changes along the inlet plenum and the hot exit channel. It then tests whether that flow split is stable, with two independent methods: a pressure-drop slope test (Ledinegg) and an eigenvalue test of the fuel-temperature dynamics. The app runs the solver live (about 0.3 s per solve) for any power level, frit setting, power shape, or one-slice disturbance you choose.

## How we built it
- Python solver (NumPy/SciPy): a radial march through each slice (cold frit, particle bed, hot frit), coupled across slices and the two manifolds, with a continuation rule when a direct solve fails.
- Hydrogen properties from CoolProp below 1000 K and NASA/NIST data (Svehla, NIST-JANAF) up to 5000 K, blended smoothly.
- Streamlit app for the live simulator and the validation table.
- 70 automated verification tests: mass and energy balance, exact momentum bookkeeping, Jacobians checked against brute force, a grid convergence study (20×20 is within 0.02% of 40×40).

## Validation: rules first, results second
Every pass/fail rule was written and committed to Git before the solver produced a single result, and failures are reported, not tuned away.
- **Passed:** V5 (Tsinghua element pressure drop, 1061 kPa against a published 1189 kPa, within the ±25% rule), V6, V8–V12. [SCOTT: confirm V5 inlet and V6 numbers from the JSON]
- **Failed: V7.** With no cold frit, the model's instability onset came out at 1.0 MW/L, outside our pre-set window of 1.3–12 MW/L around Kerrebrock's ~4 MW/L. [ANTHONIE: explanation, e.g. hot frit still present; whether 1.0 MW/L is the sweep edge]
- **Headline result (V12):** Ji et al.'s optimized cold frit ("point A") lowers the peak fuel temperature from 4376 K to 3731 K and the pressure drop from 512 to 395 kPa, but only once flow shift in the hot channel is modeled. Without the channel model, point A made things worse.

## Challenges we ran into
- Hydrogen data above 1000 K: no single source covers 150–5000 K, so we blended three and checked continuity of values and slopes.
- Keeping ourselves honest under time pressure: freezing parameters and pass rules before any run.
- [ANTHONIE: solver/physics challenge]

## Accomplishments we're proud of
- A validation table that includes a real failure.
- Two independent stability tests that agree (V9).
- [ANTHONIE]

## What we learned
[BOTH]

## Limitations
- The heat-transfer correlation is below its valid Reynolds range at the hot face in almost every case.
- The baseline elements run above the 3800 K fuel limit at full power (V10, V11); point A brings the peak back under it.
- Properties stop at 5000 K, and hydrogen dissociation is not modeled.
- Steady state plus linear stability only; no full transient simulation of a runaway.

## What's next
Tier 2 checks (V1–V4, U-flow pattern), stability maps across power and frit resistance, and a full transient model.

## Built with
Python, NumPy, SciPy, CoolProp, Streamlit, Matplotlib, pytest
