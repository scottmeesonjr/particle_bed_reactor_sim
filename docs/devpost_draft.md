# Can a Frit Stop a Runaway?

**A validated stability simulator for one fuel element of a particle bed nuclear rocket.**
Team ATcOde: Scott Meeson, Anthonie Page.

## Inspiration
A particle bed nuclear rocket heats hydrogen by pushing it through a bed of tiny, very hot fuel particles. The design has a known failure mode: if one region runs a little hot, the hydrogen there gets harder to push, less of it flows, and that region gets hotter still until the fuel fails. A porous wall called a cold frit is meant to stop this. NASA analysis concluded that a bed without a cold frit would be unstable at low power density and that a cold frit could very likely cure it, but said firmer conclusions needed calculations for specific designs. We wanted to run that calculation for a specific published design, across the whole throttle range, using only published numbers.

## What it does
The simulator splits one published fuel element (Ji et al., 2019) into 20 parallel flow paths and solves for how the hydrogen divides between them, including how pressure changes along the inlet plenum and the hot exit channel. It then tests whether that flow split is stable with two independent methods: a pressure-drop slope test (Ledinegg) and an eigenvalue test of the fuel-temperature dynamics.

The app has four parts:
- **Live simulator.** Pick power, frit settings, power shape, or a one-slice disturbance; the solver runs in under a second and shows the bed temperature map, the flow per slice, and the verdict.
- **Stability map and runaway.** Where stability is lost across power and cold frit strength, and an animation of a runaway with and without a cold frit.
- **AI vs. physics.** A trained surrogate model and the physics solver answer the same design side by side, with the time each took.
- **Validation and open questions.** The official results table and the answers to five research questions, each read from the result files and tagged with the Git commit that produced it.

## How we built it
- **Solver (Python, NumPy, SciPy):** a radial march through each slice (cold frit, particle bed, hot frit), coupled across slices through the plenum and channel, in both Z and U flow patterns, solved as one system with a continuation rule when a direct solve fails.
- **Hydrogen properties:** CoolProp below 1000 K and NASA/NIST data (Svehla's transport coefficients, NIST-JANAF tables) up to 5000 K, blended smoothly and checked for continuity of values and slopes.
- **Inputs:** every geometry, frit, and operating value comes from published sources (Ji et al. 2019 in *Nuclear Engineering and Design* and in *Energies*; Kerrebrock 1993; the SNTP program final report), and each one is recorded with its source.
- **Surrogate (scikit-learn):** gradient-boosted models trained on 2000 solver runs, generated in parallel from a seeded Latin hypercube, and tested on cases they never saw.
- **App:** Streamlit, reading every result from the runner's output files rather than recomputing it.
- **Verification:** 101 automated tests, including mass and energy balance, momentum bookkeeping, Jacobians checked against brute force, property continuity, published reference values, and the rules that score each research question. Grid convergence: 20×20 is within 0.02% of 40×40.

## Why the results aren't built in
Every pass/fail rule and every research-question procedure was written and committed to Git before it was first run. Each later decision was added as a dated amendment, before the run it affected, with its reason. Failures are reported, not tuned away. Every result file carries the commit hash of the code that produced it.

## Validation (Tier 1)
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

**About the V7 failure.** Without a cold frit, the model has a steady solution at only one of the 39 swept powers, the lowest (5%, 1.0 MW/L), and that point is unstable. At every higher power, flow shift starves the bottom slice until it overheats past the 5000 K limit of our hydrogen data, so there is no steady state to test. The onset therefore lands at 1.0 MW/L, below our pre-set window. We kept the failure. Kerrebrock noted that parallel-passage models like ours represent this instability poorly at low Reynolds number, and his estimate was for a different design (about 40 MW/L at full power, versus 20 MW/L here). Both stability methods agree (V9), so it is not an artifact of one test.

**About temperatures.** Our temperatures run higher than the papers'. Before writing any code, we checked the published power and flow against NIST-JANAF hydrogen data: they imply about 3310 K at the outlet, not the 3000 K the paper states, which points to a roughly 13–15% difference in the paper's hydrogen enthalpy. Our solver later reproduced that hand estimate within 3 K. We use NIST data and compare temperature trends, not absolute values.

## Validation (Tier 2): the Tsinghua large reactor
V1–V4 test a 1000 MW single-core reactor from the *Energies* paper. Before running them we added a rule: a check needing a case with no steady solution inside our 5000 K property range is reported as **"not evaluable"** and never counted as a pass. All four came back not evaluable. The paper's own baseline peaks at 7121 K, so this was expected.

The cases that did solve are reported separately, labeled as observations and not counted. At the 80 cm and 100 cm core heights, our pressure drops come within 0.3% and 1.3% of the paper's (outlet-referenced; 7–8% high if the pressure is referenced at the inlet), from inputs fixed before any run. Both frit-heavy variants solve at about 3600–3700 K where the baseline cannot, at 2 to 4 times the pressure drop.

## What we found: five research questions
**Q1. Does power density decide whether cold-frit or hot-frit control works?** Ji et al. explain the cold- vs. hot-frit debate this way. In our model, **partly**: cold-frit control is stable at every power density from 1 to 30 MW/L on both elements, but hot-frit control fails everywhere, including at low power density. For stability, that contradicts the low-power half of the explanation. Only stability is tested; hot-frit control may have other advantages.

**Q2. Where does the instability start, and how much cold frit cures it?** With the channel and plenum included, the element without a cold frit has no steady state above 1 MW/L. It is cured once the cold frit carries about **75.5%** of the element's pressure drop; the published frit carries 90%. In Kerrebrock's simpler parallel-passage setup, our model reproduces his conclusion in form (unstable below a threshold, cured by a cold frit) but at about four times his threshold.

**Q3. How large is the flow shift, and does a U flow pattern reduce it?** Yes, by about five times. With the standard Z pattern, slice flows vary by 15% and the most-fed slice gets 1.6 times the flow of the least-fed. That shift alone adds about 600 K to peak fuel temperature (3064 K → 3671 K), eating most of the margin to the 3800 K limit. The U pattern cuts the variation to 3% and brings the peak back to 3131 K. On the large reactor the question is not answerable, since neither pattern solves within range; with channel effects off it solves at 3217 K, so flow shift is what pushes it out.

**Q4. Does Ji et al.'s optimized frit (point A) stay stable across the 20–100% throttle range?** Ji et al. optimized it at one operating point. **Yes,** it is stable at every power from 5% to 100%. But its stability margin is a half to two-thirds of the published frit's, and its temperature advantage reverses at low throttle: it runs hotter than the published frit below 5.5 MW/L and goes above the 3800 K fuel limit at 20–22.5% power.

**Q5. Can a fast trained model stand in for the solver?** Ji et al. name a surrogate as future work. By the rule we set before training (balanced accuracy at least 0.95 and peak-temperature error at most 50 K), **no.** It is about 180,000 times faster than the solver, and its temperature error of 41.5 K passes. But it misses 8 of the 48 held-out designs with no steady state, predicting one exists, so its balanced accuracy is 0.915. Plain accuracy is 0.978, which looks excellent only because 88% of designs work; the stricter rule is what caught it. Outside its training range it degrades sharply (accuracy 0.685).

## Challenges we ran into
- **Hydrogen data above 1000 K:** no single source covers 200–5000 K, so we blended three and tested that values and slopes have no jumps at the seams.
- **Published numbers that don't balance:** the energy check above meant deciding, before any run, to keep authoritative property data and report the disagreement instead of matching the paper.
- **Solutions that don't converge from a naive start:** near the instability the solver's trial steps can leave the property data or produce reverse flow, so we wrote a continuation rule in advance. A case counts as "no solution" only after that rule is exhausted, never because one trial step failed.
- **Results that can't be judged:** the large reactor runs hotter than our property data reaches. Deciding beforehand what "not evaluable" means kept us from quietly swapping in a case that would run.
- **Keeping ourselves honest under time pressure:** freezing parameters and rules before any run, recording every amendment with its reason, and committing code before generating results.

## Accomplishments we're proud of
- A validation table that includes a real failure, explained rather than tuned away.
- Two independent stability tests that agree (V9), and a time-stepped runaway animation whose growth rate matches the eigenvalue test.
- A hand calculation made before any code existed, later matched by the solver within 3 K.
- Pressure drops within about 1% of the Tsinghua large reactor's published values, from untuned inputs.
- A surrogate we were willing to fail: the pre-registered rule said no, and that's what we report.

## What we learned
- Writing pass/fail rules before seeing results changes how you work: every result becomes a finding, including the ones that miss.
- Published papers can disagree with reference data, and the honest move is to show it.
- Accuracy can mislead when most cases are easy; choosing the metric before training mattered.
- Flow shift along the element, not the bed itself, is what most often pushes these designs past their limits in our model.

## Limitations
- **Known issue.** The plenum pressure-recovery term uses ρu², but our stated assumption (gas leaving the plenum keeps its momentum) implies ½ρu². A correction was committed during development but was overwritten by a later rewrite of the solver, and every result reported here uses ρu². Restoring it means a new model version and rerunning all results, which is our first next step.- The 5000 K property limit applies to hydrogen. Fuel temperature is reported but not range-limited, and fuel above 3800 K is past its melting point; a "stable" label there means the flow is stable, not that the fuel survives. About half of the designs the surrogate labels OK are in this situation.
- The heat-transfer correlation is below its valid Reynolds range at the hot face in almost every case; it affects fuel temperature, not how flow splits.
- Properties stop at 5000 K, and hydrogen dissociation is not modeled.
- Parallel-passage model: no cross-flow or heat conduction between slices.

## What's next
- Cross-flow and conduction between slices, and comparison with CFD of the same geometry.
- A surrogate whose "OK" also requires fuel below 3800 K, tested on a fresh, pre-registered set.
- Reproducing Ji et al.'s frit optimizer (V13) across the full throttle range, not one operating point.
- Experimental validation with a cold-flow test rig.

## How we used AI
We used Claude (Anthropic) for literature search, parameter sourcing, and planning, and Claude Code as a coding assistant. We reviewed every change, wrote the pass/fail rules and froze the inputs before any validation run, and made every modeling decision ourselves, recording each with its reason in the repository.

## Built with
Python, NumPy, SciPy, scikit-learn, CoolProp, Streamlit, Matplotlib, pytest