# FritCheck: Can a Frit Stop a Runaway?

**A validated stability simulator for one fuel element of a particle bed nuclear rocket.**
Team ATcOde: Scott Meeson, Anthonie Page.

## Inspiration
A particle bed nuclear rocket heats hydrogen by pushing it through a bed of tiny, very hot fuel particles. The design has a known failure mode: if one region runs a little hot, the hydrogen there gets harder to push, less of it flows, and that region gets hotter still until the fuel fails. A porous wall called a cold frit is meant to stop this. A NASA analysis (Kerrebrock, 1993) concluded that a bed without a cold frit would be unstable at low power density and that a cold frit could very likely cure it, but said firmer conclusions needed calculations for specific designs. We ran that calculation for specific published designs, across the whole throttle range, using only published numbers.

## What it does
FritCheck splits one published fuel element (Ji et al., 2019) into 20 parallel flow paths and solves for how the hydrogen divides between them, including how pressure changes along the inlet plenum and the hot exit channel. It then tests whether that flow split is stable, with two independent methods: a pressure-drop slope test (Ledinegg) and an eigenvalue test of the fuel-temperature dynamics. The app runs the solver live (about 0.3 s per solve) for any power level, frit setting, power shape, or one-slice disturbance you choose, and shows the validation table, the stability maps, and a runaway animation next to it.

## How we built it
- **Solver (Python, NumPy, SciPy):** a radial march through each slice (cold frit, particle bed, hot frit), coupled across slices through the plenum and channel, solved as one system with a continuation rule when a direct solve fails.
- **Hydrogen properties:** CoolProp below 1000 K and NASA/NIST data (Svehla's transport coefficients, NIST-JANAF tables) up to 5000 K, blended smoothly and checked for continuity of values and slopes.
- **Inputs:** every geometry, frit, and operating value comes from published sources (Ji et al. 2019 in *Nuclear Engineering and Design* and in *Energies*; Kerrebrock 1993; the SNTP program final report), and each one is recorded with its source.
- **App:** Streamlit, with every result read directly from committed output files, never recomputed for display.
- **Verification:** 79 automated tests, including mass and energy balance, momentum bookkeeping, Jacobians and eigenvalues checked against brute force, property continuity, and published reference values. Grid convergence: 20×20 is within 0.02% of 40×40.

## Validation: rules first, results second
Every pass/fail rule was written and committed to Git before the first validation run, and failures are reported, not tuned away. The first official run is at commit 89ef317; a clean rerun at commit 7659096, which only added reporting of which sweep points converged, gave identical numbers and verdicts.

| Check | What it tests | Result |
|---|---|---|
| V5 | Tsinghua element pressure drop vs. published 1189.2 kPa (±25%), stable | **Pass:** 1060.8 kPa (outlet-referenced) and 1225.3 kPa (inlet-referenced), stable in both |
| V6 | Same element made hot-frit-dominated (frit pressure drops swapped by a pre-set rule: cold-frit share 96% → 4%) | **Pass:** runaway with both pressure references, as the paper reports |
| V7 | No cold frit: instability onset within a factor of 3 of Kerrebrock's ~4 MW/L | **Fail:** see below |
| V8 | Published cold frit removes the instability | **Pass:** steady and stable at all 39 powers from 5% to 100% |
| V9 | Slope test and eigenvalue test agree on onset | **Pass:** both at 5% power (but see below) |
| V10 | Ji et al. element pressure drop vs. published 529.8 kPa (±25%) | **Pass:** 512.0 kPa |
| V11 | Cosine power raises peak temperature with almost no pressure-drop change | **Pass:** 4027 K → 4376 K; pressure drop changes 0.06% |
| V12 | Ji et al.'s optimized cold frit ("point A") lowers peak temperature and pressure drop | **Pass:** 4376 K → 3731 K; 511.7 → 394.6 kPa (published 414.4 kPa) |

**About the V7 failure.** Without a cold frit and with the plenum and channel modeled, the element has **no steady solution at 38 of the 39 powers** we swept (1.5 to 20 MW/L): flow shift starves the bottom slice until its gas would pass the 5000 K limit of our property data. The only power with a steady solution, 1.0 MW/L, is unstable. So the pre-set rule reads an onset of 1.0 MW/L, outside the 1.33–12 MW/L window, and V7 fails. The real finding is stronger than an onset number: in this model, without a cold frit there is no safe operating point at all. V9 also rests on that single converged point, so its agreement is weak evidence.

When we turn the plenum and channel off, which is Kerrebrock's own parallel-passage setup, the no-cold-frit element is unstable from 1.0 up to 16.5 MW/L. That reproduces the form of his result (unstable below a threshold) but at roughly four times his number, so our model is more pessimistic than his, not less. His estimate was for a different design point, and parallel-passage models leave out cross-flow between regions of the bed, which he noted limits them.

**About temperatures.** Our temperatures run higher than the papers'. Before writing any code, we checked the published power and flow against NIST-JANAF hydrogen data: they imply about 3310 K at the outlet, not the 3000 K the paper states, which points to a roughly 13–15% difference in the paper's hydrogen enthalpy. Our solver later reproduced that hand estimate within 3 K. We use NIST data and compare temperature trends, not absolute values.

## Results: the open questions
Each study's procedure was fixed in code before its first run, and every result file records its commit.

- **How much cold frit cures the instability? (Q2)** In Kerrebrock's setup (channel off), a cold frit carrying between 20% and 30% of the element's pressure drop cures it; the published frit carries 90%. With the plenum and channel on, the element needs a cold frit carrying about 75.5% of the pressure drop (the boundary lies between 75.4% and 75.7%) to have a stable steady solution at every power; below that, the element first loses its steady solution at full power.
- **Does power density decide whether cold- or hot-frit control works? (Q1)** Ji et al. suggest hot-frit control suits low power density and cold-frit control suits high. In our model, cold-frit-dominated designs were stable at every power density from 1 to 30 MW/L on both published elements, and hot-frit-dominated designs never were, with or without the channel modeled. So, for stability, the model supports their high-power half and contradicts the low-power half. Their preference may rest on other factors we didn't test, like temperature uniformity.
- **Does the optimized design stay stable when throttled? (Q4)** Ji et al. optimized their cold frit (point A) at one operating point and never checked throttling. We found point A **stays stable at every power from 5% to 100%**, but with **half to two-thirds of the published frit's stability margin** (disturbances die out more slowly), and its temperature advantage reverses at low power: below about 5.5 MW/L it runs hotter than the published frit, and at 20–22.5% power it passes the 3800 K fuel limit, inside the operating range.
- **What does a runaway look like? (demo)** At 8 MW/L with the channel off, we nudged one slice 20 K hotter. With the cold frit, the disturbance dies out (it shrinks by a factor of e every 0.39 s). Without it, the hot slice loses flow, passes the 3800 K fuel limit at 7.6 s, and leaves our property range at 9.1 s.

## Challenges we ran into
- **Hydrogen data above 1000 K:** no single source covers 200–5000 K, so we blended three and tested that values and slopes have no jumps at the seams.
- **Published numbers that don't balance:** the energy check above meant deciding, before any run, to keep authoritative property data and report the disagreement instead of matching the paper.
- **Solutions that don't converge from a naive start:** near the instability the solver's trial steps can leave the property data or produce reverse flow, so we wrote a continuation rule in advance. A case counts as "no solution" only after that rule is exhausted, never because one trial step failed.
- **Keeping ourselves honest under time pressure:** freezing parameters and pass rules before any run, recording every amendment with its reason, and committing code before generating results so every result traces to an exact commit.

## Accomplishments we're proud of
- A validation table that includes a real failure, explained rather than tuned away.
- A new result on a published design: point A stays stable when throttled, but with a smaller margin and a fuel-limit violation at low power.
- A hand calculation made before any code existed, later matched by the solver within 3 K.
- Taking one solve from about 15 s to 0.3 s, which made live exploration and full sweeps possible.

## What we learned
- Writing pass/fail rules before seeing results changes how you work: every result becomes a finding, including the ones that miss.
- Published papers can disagree with reference data, and the honest move is to show it.
- Code review catches physics errors that tests miss when the test encodes the same mistake (see Limitations).
- An optimization done at one operating point can trade away margin elsewhere; checking the whole operating range matters.

## Limitations
- **Known issue in the plenum model:** every result with the plenum and channel on (V5–V12, and the channel-on parts of Q1, Q2, and Q4) uses a plenum momentum term that recovers ρu² of pressure instead of ½ρu², which contradicts our own stated assumption and likely overstates flow shift. A correction was written before the first official run (commit 052b12c on branch anthonie-steady) but was not merged into the code that produced these results. Under our versioning rule, the fix becomes a new model version, everything is rerun, and both runs are reported; that rerun is pending. Results with the channel off (Kerrebrock's setup in Q1 and Q2, and the runaway demo) don't use the plenum and are unaffected.
- The heat-transfer correlation is below its valid Reynolds range at the hot face in almost every case; it affects fuel temperature, not how flow splits.
- The baseline elements run above the 3800 K fuel limit at full power in our model (V10, V11); point A brings the peak back under it.
- Properties stop at 5000 K, and hydrogen dissociation is not modeled. "No solution" means a slice would exceed this range, not necessarily that no physical state exists.
- Parallel-passage model: no cross-flow between slices.
- The runaway demo is a simplified transient (one temperature per slice, with flows readjusting instantly), not a full transient simulation.

## What's next
- Apply the plenum correction as version v3.1 and publish both versions of every channel-on result side by side.
- Tier 2 checks (V1–V4, U flow pattern) on the larger Tsinghua reactor.
- A fast surrogate model for design screening, which Ji et al. list as future work.
- A full transient model and cross-flow between slices.

## How we used AI
We used Claude (Anthropic) for literature search, parameter sourcing, planning, and drafting the model specification, and Claude Code as a coding assistant. We reviewed every change, wrote the pass/fail rules and froze the inputs before any validation run, and made every modeling decision ourselves, recording each with its reason in the repository.

## Built with
Python, NumPy, SciPy, pandas, CoolProp, Streamlit, Matplotlib, pytest