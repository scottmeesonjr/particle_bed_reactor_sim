# Can a Frit Stop a Runaway?

A stability simulator for one fuel element of a particle bed nuclear rocket, built at SwampHacks XII.

Team: ATcOde
Members: Scott Meeson, Anthonie Page

## What it does

A particle bed nuclear rocket heats hydrogen by blowing it through a bed of tiny hot fuel particles. If one region gets a little hotter, the hydrogen there gets thicker and harder to push. Less hydrogen flows through it, so it gets hotter still, until the fuel melts. A porous wall called a cold frit adds resistance that does not change with temperature, which is meant to stop this runaway.

The simulator splits one published fuel element into 20 parallel flow paths and answers three things for any settings you choose:

1. **How the hydrogen splits** between the paths, so that every path has the same pressure drop and the flows add up to the total.
2. **Whether that split is stable**: if one path loses a little flow, does it recover or run away?
3. **Where stability is lost**, by repeating 1 and 2 across power levels and frit resistances to map the boundary.

Everything it shows is a model prediction from physics equations, not a measurement.

## Why the results aren't built in

- Every input comes from published sources and was committed to Git before any validation run (`params.py`).
- The pass/fail rule for every comparison with published results was committed as code before the first run (`validation.md`, `runner.py`).
- Results that disagree with the papers are reported, not tuned away.
- The app runs the solver live on whatever settings you pick.

## Install

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas CoolProp streamlit matplotlib pytest scikit-learn
```

## Run

```bash
streamlit run app.py        # the interactive simulator
python runner.py            # validation checks; writes results/
python runner.py --channel-off   # fallback rule: channel-dependent checks marked not run
python questions.py q3      # Q3: flow shift, Z vs U; then python question_report.py
python surrogate.py         # Q5: train and test the surrogate (uses every CPU core)
pytest                      # verification tests
python properties.py        # hydrogen property tables and checks
python params.py            # input summary and checks
```

## Files

| File | What it holds |
| --- | --- |
| `params.py` | Every input with its source, the power shapes, and the cold frit porosity profile |
| `properties.py` | Hydrogen and fuel properties from 150 K to 5000 K |
| `rules.py` | Rules computed in code, such as the hot-frit-dominated test element |
| `validation.md` | Pre-registered checks, hypotheses, and decisions |
| `runner.py` | Runs the checks and writes result files tagged with the Git commit |
| `app.py` | The Streamlit app |
| `data/` | NIST-JANAF tables for hydrogen and zirconium carbide |
| `results/` | Validation output, one file per run |
| `slice.py` | One slice: radial march through cold frit, bed, and hot frit |
| `steady.py` | All 20 slices together: flow split, continuation rule |
| `stability.py` | Static (Ledinegg) and eigenvalue stability tests |
| `model.py` | Connects the solver to the app and the validation runner |
| `questions.py`, `question_report.py` | Research questions Q1–Q4 and their reports |
| `surrogate.py` | Q5: solver training data, the trained surrogate, and its scorecard |
| `docs/model.md` | The equations, written down before coding |
| `tests/`, `tests_anthonie/` | Verification tests (`pytest` runs both) |

## Sources

- Ji, Zhang, Sun, Shi (2019). Thermal performance optimization of a fuel element in particle bed reactors for nuclear thermal propulsion. *Nuclear Engineering and Design* 355:110316. The fuel element design and most validation targets.
- Ji, Liu, Sun, Shi (2019). Analysis of the flow distribution in a particle bed reactor for nuclear thermal propulsion. *Energies* 12(19):3590.
- Kerrebrock (1993). Flow instability in particle-bed nuclear reactors. NASA Lewis Research Center.
- NIST-JANAF Thermochemical Tables (H-050, C-110); Svehla (1995), NASA TM-4647; CoolProp.

## Limitations

- Flow paths do not exchange flow or heat sideways, as they do in a real bed.
- Frits are modeled as uniform resistances.
- Hydrogen dissociation above about 3500 K is not modeled; properties stop at 5000 K.
- Our gas temperatures run hotter than the Tsinghua papers' because NIST data gives about 13–15% less enthalpy rise than their property model. We compare trends, not absolute temperatures.

## How we used AI

- AI was utilized in writing code.
- AI was not utilized for theory or governing equations.