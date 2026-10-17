# SwampHacks XII Game Plan: PBR Stability Simulator
## What we're building

We are building a website where judges can break a nuclear rocket engine and then fix it.

A particle bed nuclear rocket heats hydrogen by blowing it through a bed of tiny hot fuel beads. If one spot gets a little hotter, the hydrogen there gets thicker and harder to push, like syrup. Less hydrogen goes there, so it gets even hotter, until it melts. That is a runaway.
The proposed fix is a frit: a porous wall the hydrogen must squeeze through first. It adds resistance that does not change with temperature, which weakens the runaway loop.

Our program cuts one fuel element into 20 slices and uses real physics and real hydrogen data to compute how much hydrogen goes through each slice and how hot it gets. We check it against a published Tsinghua study (Ji et al. 2019), which found the cold-frit design stable and the hot-frit design unstable. An AI model trained on thousands of runs then predicts the answer instantly, and we show how often it is right.

Everything we show is a model prediction, not a measurement. We say that up front.

Scott: Inputs, properties, verification tests, validation, web app, Devpost, pitch
params.py, properties.py, validation.md, tests/, app.py, README.md

Anthonie: Physics solver, stability analysis, sweeps, AI model
slice.py, steady.py, stability.py, transient.py, surrogate.py

Game plan
Times are hours after we start. The goal is a complete, working app by hour 12, then the AI layer, then submission with at least one hour of buffer.
1. Set up (both, 0:00–0:30)
    2. Scott creates a GitHub repo and adds the teammate. Clone it on both laptops.
    3. Both: create a virtual environment and run: pip install numpy scipy pandas CoolProp streamlit matplotlib pytest scikit-learn.
    4. Scott adds params.py and properties.py from the chat. Both run python properties.py and confirm it prints the property table and the two sanity checks.
    5. Commit: "initial setup".
2. Freeze the inputs (Scott, 0:30–1:15)
    1. Open the Tsinghua paper. Check every number in params.py against Table 5 and Section 3.3.3.
    2. Decide the power mismatch: keep 30 MW/L and 0.396 kg/s as stated, and let the model compute the outlet temperature. Write that decision as a comment in params.py.
    3. Check the Shomate coefficients in properties.py against the NIST WebBook page for hydrogen gas.
    4. Commit: "freeze params v1". Do not change these values after this commit.
3. Write the slice equations on paper (teammate, 0:30–1:15)
    1. Write the radial march for one slice, from the cold frit outer radius inward: cold frit (Ergun), then bed (KTA friction factor plus energy gain), then hot frit (Ergun).
    2. Write the fuel temperature equation using the KTA heat transfer correlation in the params.py comments.
    3. Take a photo of the page. It goes in the Devpost.
4. Write the validation criteria (Scott, 1:15–1:45)
    1. Create validation.md with one line per check and its pass rule, written before any results exist:
        ▪ Paper's cold-frit-dominated frits: model is stable at nominal power.
        ▪ Hot-frit-dominated variant: model is unstable at nominal power. Define the variant's frit values here first.
        ▪ Cold frit carries most of the pressure drop; total radial drop is within 40% of 1189 kPa (the paper's number also includes channel losses).
        ▪ Ledinegg slope test and eigenvalue test agree on where instability starts.
        ▪ Energy balance closes within 1%.
    2. Commit: "validation criteria".
5. Build the single-slice model (teammate, 1:15–4:00)
    1. In slice.py, write slice_dp(m_dot, q_slice, cold_mult, hot_mult). It returns pressure drop, gas outlet temperature, and peak fuel temperature.
    2. Milestone: plot pressure drop versus flow for one slice at nominal power. With frits turned off, the curve should have a minimum. With the paper's cold frit, it should rise steadily.
    3. Commit the plot image and the code.
6. Build the app with fake data (Scott, 1:45–4:00)
    1. In app.py, add sliders for power (50–150% of nominal), cold frit resistance multiplier, hot frit resistance multiplier, and a "hot slice" disturbance (0–30%).
    2. Add plots: flow per slice (bar chart), gas and fuel temperature per slice (line chart), and a big STABLE / RUNAWAY label.
    3. Write fake_solve() that returns made-up arrays of the right shape, clearly named fake. Run streamlit run app.py and confirm the sliders update the plots.
7. Write the verification tests (Scott, 4:00–5:30)
    1. In tests/, write pytest tests for: energy balance, mass balance, the constant-property Ergun case against the hand formula, and symmetry (uniform power gives equal slice flows).
    2. Run pytest after every change from now on.
8. Build the 20-slice solver (teammate, 4:00–6:00)
    1. In steady.py: hold the total pressure drop fixed. For each slice, find the flow that matches that pressure drop with scipy.optimize.brentq.
    2. Choose the total pressure drop so that total flow is 0.396 kg/s at nominal power.
    3. Milestone: uniform power gives 20 equal flows that add to 0.396 kg/s. Scott's tests pass.
9. Add stability (teammate, 6:00–8:00)
    1. In stability.py: a slice is unstable where its pressure drop falls as flow rises (Ledinegg test). Add the eigenvalue check as the second, independent test.
    2. Write max_stable_power(cold_mult, hot_mult) and run the frit sweep to make the stability map.
    3. In transient.py: a simple fuel-temperature time march after a small disturbance, for the runaway animation.
10. Run validation (both, 8:00–9:00)
    1. Run each check in validation.md exactly as written. Record pass or fail and the numbers.
    2. If a check fails, do not tune anything. Write down why, and report it in the app and Devpost.
    3. Commit: "validation results".
11. Connect the real solver to the app (Scott, 8:00–11:00)
    1. Replace fake_solve() with the real solver. Cache results with @st.cache_data.
    2. Add tabs: Stability map, Validation results, What's in the model, and a runaway animation.
    3. Check that one solve takes under about 2 seconds. If not, drop to 10 slices for live mode and say so on screen.
12. Tier 1 checkpoint (both, 11:00–12:00)
    1. Record a 2-minute backup video of the app: a runaway, then the frit fix.
    2. Push everything. From here on, the project is complete even if nothing else works.
13. AI layer (teammate, 12:00–14:00)
    1. In surrogate.py: run about 2,000 solver cases across power and frit settings (Latin hypercube, saved with its random seed).
    2. Train a stable/unstable classifier and a peak-temperature model. Report accuracy on held-out cases and on an out-of-range set.
    3. Scott adds an "AI vs physics" panel to the app showing both answers and their run times.
14. Write-up (Scott, 12:00–14:00)
    1. Draft the Devpost: pitch, what's real vs. modeled, validation table, limitations, how AI was used.
    2. Write README.md with install and run instructions.
    3. Make figures: model diagram, stability map, validation chart.
15. Optional Uno control panel (Scott, only if everything above is done)
    1. Potentiometer as the frit knob, a button for power, buzzer and red LED for runaway, talking to the laptop over USB.
    2. Hard limit: one hour. If it isn't working by then, unplug it.
16. Feature freeze and submit (both, 14:00–16:00)
    1. No new features after hour 14.
    2. Record the final demo video (under 2 minutes).
    3. Rehearse the 3-minute pitch three times, timing it. Practice the hard questions: "isn't this just a simulation?" and "did you tune it to match?"
    4. Submit on Devpost at least one hour before the deadline.
