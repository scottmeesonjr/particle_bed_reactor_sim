"""Verification: params.py derived values and rules (tests/reference_values.md)."""
import numpy as np
import pytest

import params as P
import properties as pr


def test_bed_volume_and_power():
    assert P.bed_volume() * 1e3 == pytest.approx(1.3069, rel=1e-4)
    assert P.total_power() / 1e6 == pytest.approx(26.138, rel=1e-4)


@pytest.mark.parametrize("name", ["uniform", "cosine", "asymmetric", "reflectors"])
def test_power_shapes_average_to_one(name):
    s = np.linspace(0.0, 1.0, 20001)
    assert np.trapezoid(P.power_shape(name, s), s) == pytest.approx(1.0, abs=1e-6)


def test_point_a_mean_relative_porosity_is_one():
    s = np.linspace(0.0, 1.0, 20001)
    eps = P.cold_frit_porosity(s, **P.POINT_A)
    assert np.trapezoid(eps / P.POINT_A["alpha0"], s) == pytest.approx(1.0, abs=1e-6)


def test_rule_D2_gives_3000K_outlet():
    m = P.nominal_mass_flow()
    h_out = pr.h(P.T_INLET) + P.total_power() / m
    assert float(pr.T_from_h(h_out)) == pytest.approx(P.T_OUT_TARGET, abs=0.5)


def test_rule_D2_close_to_hand_estimate():
    # reference_values.md: flag if the code differs from 0.5758 kg/s by more than 1%
    assert P.nominal_mass_flow() == pytest.approx(0.5758, rel=0.01)


def test_throttled_flow_scales_with_power():
    assert P.nominal_mass_flow(0.2) == pytest.approx(0.2 * P.nominal_mass_flow(), rel=1e-12)