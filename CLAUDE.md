# Project rules for Claude Code
Particle bed reactor stability simulator (SwampHacks XII). Python.

## Read first
- docs/model.md: equations to implement (the spec; follow it exactly)
- validation.md: criteria, hypotheses, fallback rule (never edit)
- tests/reference_values.md: expected values for tests
- data/: sourced data (never edit)

## Ownership
- Anthonie: slice.py, steady.py, stability.py, transient.py, make_*.py
- Scott: params.py, properties.py, rules.py, app.py, runner.py, tests/
- Never modify the other person's files. If a change is needed there, say so and stop.

## Interfaces (as frozen; read the files for exact names)
- properties.py: h(T), T_from_h(h), cp(T), mu(T), k(T), rho(T, P).
  Only rho takes pressure. Catch PropertyRangeError; never extrapolate.
- params.py: geometry, frit values, T_LIMIT (3800 K design limit), P_OUTLET, T_INLET,
  power_shape, nominal_mass_flow, porosity-profile function. Use these; never retype numbers.

## Anti-hardcoding rules
- Never change a parameter, correlation, or model choice to match a validation target or hypothesis.
- If a result disagrees with a hypothesis in validation.md, report it; don't change code to get the expected shape.
- If docs/model.md doesn't decide something, stop and ask. Don't choose silently.
- Temporary code must be named temp_* or fake_* and deleted when replaced.

## Conventions
- SI units. Pressure drop positive = pressure falls along the flow.
- Slice i = 0 at the bottom; channel exits at the top (Z pattern).
- Run pytest before every commit. Save results to results/ with the git commit hash.
