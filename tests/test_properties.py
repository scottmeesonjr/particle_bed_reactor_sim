"""Verification: properties.py reproduces its sourced data (tests/reference_values.md)."""
import numpy as np
import pytest

import properties as pr

# Svehla (1995) Table III, evaluated by hand in tests/reference_values.md
SVEHLA = [  # T (K), viscosity (Pa s), conductivity (W/(m K))
    (200.0, 6.8167e-06, 0.1316), (300.0, 8.9738e-06, 0.1879), (500.0, 1.2683e-05, 0.2667),
    (999.99, 2.0557e-05, 0.4427), (1000.01, 2.0558e-05, 0.4427),
    (2000.0, 3.3507e-05, 0.8077), (3000.0, 4.5601e-05, 1.1885),
]


@pytest.mark.parametrize("T, mu_ref, k_ref", SVEHLA)
def test_svehla_reference_values(T, mu_ref, k_ref):
    # Compared at the precision the reference is written to (half its last digit):
    # viscosity is given to 5 significant figures, conductivity to 4 decimal places.
    assert pr.svehla_viscosity(np.array([T]))[0] == pytest.approx(mu_ref, abs=0.5e-9)
    assert pr.svehla_conductivity(np.array([T]))[0] == pytest.approx(k_ref, abs=0.5e-4)


def test_svehla_coefficients_match_data_file():
    from pathlib import Path
    rows = [ln.split(",") for ln in (Path(pr.DATA_DIR) / "svehla_H2.txt").read_text().splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("property")]
    from_file = {(r[0], float(r[1])): tuple(float(x) for x in r[3:7]) for r in rows}
    for name, table in (("viscosity", pr._SVEHLA_VISC), ("conductivity", pr._SVEHLA_COND)):
        for lo, hi, coeffs in table:
            assert coeffs == from_file[(name, lo)]


def test_svehla_ranges_meet_at_1000K():
    for f in (pr.svehla_viscosity, pr.svehla_conductivity):
        lo, hi = f(np.array([999.99, 1000.01]))
        assert abs(hi - lo) / hi < 1e-4


def test_janaf_h2_table_points_reproduced():
    assert np.allclose(pr._janaf_h2_cp(pr._H2_T), pr._H2_CP, rtol=0, atol=1e-9)
    for T, cp_ref in ((200.0, 13615.4), (1000.0, 14983.5), (2000.0, 17005.0), (3000.0, 18397.4)):
        assert float(pr._janaf_h2_cp(T)) == pytest.approx(cp_ref, rel=1e-4)


def test_janaf_zrc_fuel_cp():
    for T, cp_ref in ((1000.0, 517.2), (2000.0, 555.9), (3000.0, 590.0)):
        assert float(pr.fuel_cp(T)) == pytest.approx(cp_ref, rel=1e-3)


@pytest.mark.parametrize("name", ["rho", "mu", "k", "cp", "h"])
@pytest.mark.parametrize("seam", [950.0, 1000.0, 1050.0])
def test_no_kink_at_seams(name, seam):
    # v3: values and slopes continuous over 900-1100 K. A slope discontinuity can
    # only occur where formulas switch: the blend edges (950, 1050 K) and the
    # CoolProp / Svehla range limit (1000 K). Compare one-sided slopes there.
    # Inside the blend the curve is smooth by construction (smoothstep of two
    # smooth functions), though it can curve noticeably where the sources differ.
    f = getattr(pr, name)
    left = float(f(seam) - f(seam - 1.0))
    right = float(f(seam + 1.0) - f(seam))
    assert abs(right - left) / abs(0.5 * (left + right)) < 0.05


def test_enthalpy_consistent_with_cp():
    T = np.linspace(200.0, 4900.0, 48)
    dh = (pr.h(T + 0.5) - pr.h(T - 0.5))
    assert np.allclose(dh, pr.cp(T), rtol=2e-3)


def test_T_from_h_inverts_h():
    T = np.array([200.0, 950.0, 1000.0, 1050.0, 3000.0, 4999.0])
    assert np.allclose(pr.T_from_h(pr.h(T)), T, atol=1e-6)


def test_property_range_error_above_limit():
    for f in (pr.rho, pr.mu, pr.k, pr.cp, pr.h):
        with pytest.raises(pr.PropertyRangeError):
            f(5000.5)


def test_ideal_gas_error_is_small_at_1000K():
    assert abs(pr.ideal_gas_error_at_1000K()) < 0.03