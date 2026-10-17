# Reference values for verification tests (computed before coding)

## Svehla H2 transport (from data/svehla_H2.txt)
| T (K) | viscosity (Pa*s) | conductivity (W/(m*K)) |
|-------|------------------|------------------------|
| 200   | 6.8167e-06 | 0.1316 |
| 300   | 8.9738e-06 | 0.1879 |
| 500   | 1.2683e-05 | 0.2667 |
| 999.99 | 2.0557e-05 | 0.4427 |
| 1000.01 | 2.0558e-05 | 0.4427 |
| 2000  | 3.3507e-05 | 0.8077 |
| 3000  | 4.5601e-05 | 1.1885 |
Tolerance: 1e-4 relative. Continuity: |value(999.99) - value(1000.01)| / value < 1e-4.

## JANAF H2 (ideal gas)
| T (K) | cp (J/(kg*K)) |
|-------|---------------|
| 200  | 13615.4 |
| 1000 | 14983.5 |
| 2000 | 17005.0 |
| 3000 | 18397.4 |
Enthalpy rise 200 -> 3000 K: 45.397 MJ/kg (ideal gas).
Interpolation must reproduce every table point exactly.

## JANAF ZrC
cp: 517.2 J/(kg*K) at 1000 K, 555.9 at 2000 K, 590.0 at 3000 K.

## Nominal element (Ji et al. 2019)
- Bed volume: 1.3069 L
- Power at 20 MW/L: 26.138 MW
- D2 mass flow (ideal-gas JANAF hand estimate): 0.5758 kg/s
  The code uses CoolProp below 1000 K, so its value may differ slightly; whatever
  the frozen code prints is the value used. Flag if it differs by more than 1%.

## Power shapes (means over 0 <= x <= L)
- Cosine (Fm 1.25, Le 1.4L): 1.0038 before renormalization; peak 1.25
- Asymmetric polynomial: 1.0001; peak 1.290 at x/L = 0.625
- Reflector polynomial: 0.9993; range 0.903 to 1.094
After renormalization, every mean = 1.000 (tolerance 1e-6).

## Ji et al. point A porosity profile
b = 0.434, c = 1.070, alpha0 = 0.533: mean relative porosity = 1.000 (tolerance 1e-6).
