# Can a Frit Stop a Runaway?

**A validated stability simulator for one fuel element of a particle bed nuclear rocket.**
Team ATcOde: Scott Meeson, Anthonie Page.

## Inspiration
A particle bed nuclear rocket heats hydrogen by pushing it through a bed of tiny, very hot fuel particles. The design has a known failure mode: if one region runs a little hot, the hydrogen there gets harder to push, less of it flows, and that region gets hotter still until the fuel fails. A porous wall called a cold frit is meant to stop this. NASA analysis concluded that a bed without a cold frit would be unstable at low power density and that a cold frit could very likely cure it, but said firmer conclusions needed calculations for specific designs. We wanted to run that calculation for a specific published design, across the whole throttle range, using only published numbers.

## What it does
The simulator splits one published fuel element (Ji et al., 2019) into 20 parallel flow paths and solves for how the hydrogen divides between them, including how pressure changes along the inlet plenum and the hot exit channel. It then tests whether that flow split is stable, with two independent methods: a pressure-drop slope test (Ledinegg) and an eigenvalue test of the fuel-temperature dynamics. The app runs the solver live (about 0.3 s per solve) for any power level, frit setting, power shape, or one-slice disturbance you choose, and shows the official validation table next to it.

## How we built it
- **Solver (Python, NumPy, SciPy):** a radial march through each slice (cold frit, particle bed, hot frit), coupled across slices through the plenum and channel, solved as one system with a continuation rule when a direct solve fails.
- **Hydrogen properties:** CoolProp below 1000 K and NASA/NIST data (Svehla's transport coefficients, NIST-JANAF tables) up to 5000 K, blended smoothly and checked for continuity of values and slopes.
- **Inputs:** every geometry, frit, and operating value comes from published sources (Ji et al. 2019 in *Nuclear Engineering and Design* and in *Energies*; Kerrebrock 1993; the SNTP program final report), and each one is recorded with its source.
- **App:** Streamlit, with the live simulator and the validation table read directly from the runner's output files.
- **Verification:** 69 automated tests, including mass and energy balance, momentum bookkeeping, Jacobians checked against brute force, property continuity, and published reference values. Grid convergence: 20×20 is within 0.02% of 40×40.

## Validation: rules first, results second
Every pass/fail rule was written and committed to Git before the first validation run, and failures are reported, not tuned away. The official Tier 1 run is at commit 89ef317.

| Check | What it tests | Result |
|---|---|---|
| V5 | Tsinghua element pressure drop vs. published 1189.2 kPa (±25%), stable | **Pass:** 1060.8 kPa (outlet-referenced) and 1225.3 kPa (inlet-referenced), stable in both |
| V6 | Same element made hot-frit-dominated (frit pressure drops swapped by a pre-set rule: cold-frit share 96% → 4%) | **Pass:** runaway with both pressure references, as the paper reports |
| V7 | No cold frit: instability onset within a factor of 3 of Kerrebrock's ~4 MW/L | **Fail:** onset at 1.0 MW/L, outside the 1.33–12 MW/L window |
| V8 | Published cold frit removes that instability | **Pass:** stable at every power from 5% to 100% |
| V9 | Slope test and eigenvalue test agree on onset | **Pass:** both at 5% power |
| V10 | Ji et al. element pressure drop vs. published 529.8 kPa (±25%) | **Pass:** 512.0 kPa |
| V11 | Cosine power raises peak temperature with almost no pressure-drop change | **Pass:** 4027 K → 4376 K; pressure drop changes 0.06% |
| V12 | Ji et al.'s optimized cold frit ("point A") lowers peak temperature and pressure drop | **Pass:** 4376 K → 3731 K; 511.7 → 394.6 kPa (published 414.4 kPa) |

**About the V7 failure.** Without a cold frit, the model finds instability only at the bottom of the sweep: the only unstable point is the lowest power swept, 5% (1.0 MW/L). That is below Kerrebrock's ~4 MW/L by more than our pre-set factor of 3. We kept the failure. Kerrebrock himself noted that parallel-passage models like ours represent this instability poorly at low Reynolds number, and at 5% power the gas at the hot face has a Reynolds number of about 3, below the range of the bed pressure-drop correlation. His estimate was also for a different design (about 40 MW/L at full power, versus 20 MW/L for this element). Both stability methods agree on the onset (V9), so it is not an artifact of one test.

**About temperatures.** Our temperatures run higher than the papers'. Before writing any code, we checked the published power and flow against NIST-JANAF hydrogen data: they imply about 3310 K at the outlet, not the 3000 K the paper states, which points to a roughly 13–15% difference in the paper's hydrogen enthalpy. Our solver later reproduced that hand estimate within 3 K. We use NIST data and compare temperature trends, not absolute values.

## Challenges we ran into
- **Hydrogen data above 1000 K:** no single source covers 200–5000 K, so we blended three and tested that values and slopes have no jumps at the seams.
- **Published numbers that don't balance:** the energy check above meant deciding, before any run, to keep authoritative property data and report the disagreement instead of matching the paper.
- **Solutions that don't converge from a naive start:** near the instability the solver's trial steps can leave the property data or produce reverse flow, so we wrote a continuation rule in advance. A case counts as "no solution" only after that rule is exhausted, never because one trial step failed.
- **Keeping ourselves honest under time pressure:** freezing parameters and pass rules before any run, recording every amendment with its reason, and committing code before generating results so every result traces to an exact commit.

## Accomplishments we're proud of
- A validation table that includes a real failure, explained rather than tuned away.
- Two independent stability tests that agree (V9).
- A hand calculation made before any code existed, later matched by the solver within 3 K.
- Taking one solve from about 15 s to 0.3 s, which made live exploration and full sweeps possible.

## What we learned
- Writing pass/fail rules before seeing results changes how you work: every result becomes a finding, including the ones that miss.
- Published papers can disagree with reference data, and the honest move is to show it.
- Code review catches physics errors that tests miss when the test encodes the same mistake (see Limitations).
- The cold frit matters: with it, the element is stable at every power we tested; without it, the element becomes unstable at the lowest power we tested.

## Limitations
- **Known issue in the official run:** the plenum momentum term in the official run recovers ρu² of pressure instead of ½ρu², which contradicts our own stated assumption. A correction was written before the run (commit 052b12c on branch anthonie-steady) but was not merged into the code that produced the numbers above. Under our versioning rule the fix becomes a new model version, everything is rerun, and both runs are reported; that rerun is pending.
- The heat-transfer correlation is below its valid Reynolds range at the hot face in almost every case; it affects fuel temperature, not how flow splits.
- The baseline elements run above the 3800 K fuel limit at full power in our model (V10, V11); point A brings the peak back under it.
- Properties stop at 5000 K, and hydrogen dissociation is not modeled.
- Parallel-passage model: no cross-flow between slices.
- Steady state plus linear stability only; no full transient simulation of a runaway.

## What's next
- Apply the plenum correction as version v3.1 and publish both validation runs side by side.
- Stability maps across power and frit resistance, and whether Ji et al.'s optimized design stays stable when throttled.
- Tier 2 checks (V1–V4, U flow pattern) and a full transient model.
- A fast surrogate model for design screening, which Ji et al. list as future work.

## How we used AI
We used Claude (Anthropic) for literature search, parameter sourcing, and planning, and Claude Code as a coding assistant. We reviewed every change, wrote the pass/fail rules and froze the inputs before any validation run, and made every modeling decision ourselves, recording each with its reason in the repository.

## Built with
Python, NumPy, SciPy, CoolProp, Streamlit, Matplotlib, pytest
