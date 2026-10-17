# Project rules for Claude Code
Particle bed reactor stability simulator (SwampHacks XII). Python.

## Read first
- docs/model.md: equations to implement
- validation.md: validation criteria, hypotheses, fallback rule (never edit)
- tests/reference_values.md: expected values for tests
- data/: sourced data files (never edit)

## Ownership
- Anthonie: slice.py, steady.py, stability.py, transient.py
- Scott: params.py, properties.py, app.py, runner.py, tests/
- Never modify the other person's files. If a change is needed there, say so and stop.

## Anti-hardcoding rules
- Never change a parameter, correlation, or model choice to match a validation target or hypothesis.
- If a result disagrees with a hypothesis in validation.md, report it; do not alter code to produce the expected shape.
- Derived values come from functions in params.py, never typed-in numbers.
- Do not extrapolate properties: stop and flag if T > 5000 K.
- Fake or temporary code must be clearly named (temp_*, fake_*) and deleted when replaced.

## Conventions
- SI units. Pressure drop positive = pressure falls along the flow.
- Slice i = 0 at the bottom (z = 0, plenum inlet for the Z pattern); the channel exits at the top.
- Agreed interfaces:
  properties.py: h(T,p), T_from_h(h,p), rho(T,p), cp(T,p), mu(T,p), k(T,p), fuel_cp(T)
  params.py: geometry/frit constants, power_shape(name, x), nominal_mass_flow(),
             jb19_porosity(x, b, c, alpha0), P_OUT, T_IN
- Run pytest before every commit. Save results to results/ with the git commit hash.
