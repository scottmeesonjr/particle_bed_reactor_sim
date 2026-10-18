"""Verification of elements.py against the Tsinghua paper's stated inputs (Energies 2019)."""
import pytest

import elements as E


def test_power_density_matches_paper():
    assert E.POWER_DENSITY_LARGE / 1e9 == pytest.approx(1.938, abs=0.0005)


@pytest.mark.parametrize("L, flow", [(1.2, 21.0), (1.0, 17.5), (0.8, 14.0)])
def test_power_to_flow_ratio_held_constant(L, flow):
    e = E.large_reactor(L)
    assert e["m_dot"] == flow
    assert E.POWER_DENSITY_LARGE * E._bed_volume(L) / flow == pytest.approx(1000e6 / 21.0, rel=1e-12)


def test_variants():
    assert E.large_reactor(variant="cold_dominated")["cold_frit_pore"] == 10e-6
    assert E.large_reactor(variant="cold_dominated")["cold_frit_eps"] == 0.2
    assert E.large_reactor(variant="hot_dominated")["hot_frit_pore"] == 400e-6
    b = E.large_reactor()
    assert (b["cold_frit_pore"], b["hot_frit_pore"], b["cold_frit_eps"], b["hot_frit_eps"]) == (0.004, 0.004, 0.1, 0.2)


def test_plenum_outer_radius_excludes_reflector():
    b = E.large_reactor()
    outer = b["r_hot_chan"] + b["t_hot_frit"] + b["t_bed"] + b["t_cold_frit"] + b["t_plenum"]
    assert outer == pytest.approx(0.396) and outer + 0.111 == pytest.approx(0.507)
