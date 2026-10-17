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