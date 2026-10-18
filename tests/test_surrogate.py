"""Verification of the Q5 procedure rules in surrogate.py (docs/model.md A3, A4)."""
import math

import numpy as np

import surrogate as S


def test_sampling_is_reproducible_and_in_range():
    a, b = S.sample(200, S.SEED), S.sample(200, S.SEED)
    assert a == b
    for c in a:
        assert 0.20 <= c["power_frac"] <= 1.00
        assert 1.5e-6 <= c["cold_pore"] <= 4.0e-6
        assert 30e-6 <= c["hot_pore"] <= 300e-6


def test_power_shape_alternates_uniform_then_cosine():
    shapes = [c["power_shape"] for c in S.sample(10, S.SEED)]
    assert shapes == ["uniform", "cosine"] * 5


def test_latin_hypercube_stratifies_each_input():
    n = 50
    cs = S.sample(n, S.SEED)
    pf = [(c["power_frac"] - 0.2) / 0.8 for c in cs]
    cold = [(math.log10(c["cold_pore"]) - math.log10(1.5e-6)) / (math.log10(4e-6) - math.log10(1.5e-6)) for c in cs]
    for u in (pf, cold):
        assert sorted(int(v * n) for v in u) == list(range(n))     # one point per stratum


def test_out_of_range_set_uses_its_own_seed_and_cold_range():
    o = S.sample(200, S.OOR_SEED, S.COLD_PORE_OOR)
    assert all(4.0e-6 <= c["cold_pore"] <= 6.0e-6 for c in o)
    assert o != S.sample(200, S.SEED, S.COLD_PORE_OOR)


def test_budget_formula():
    assert S.budget(0.7, 8) == 2000                         # capped
    assert S.budget(2.0, 1) == 900                          # floor(1800 / 2)
    assert S.budget(7.0, 1) == 257                          # floor(257.14)


def test_answer_rule():
    assert S.answer(0.95, 50.0, 2) == "yes"                 # both thresholds inclusive
    assert S.answer(0.949, 10.0, 2) == "no"
    assert S.answer(0.99, 50.1, 2) == "no"
    assert S.answer(None, 10.0, 1) == "not answerable"      # one class in held-out set


def test_labels():
    assert S.is_ok(dict(status="stable"))
    for s in ("unstable", "out_of_range", "no_solution"):
        assert not S.is_ok(dict(status=s))
    assert S.converged(dict(status="unstable")) and not S.converged(dict(status="out_of_range"))


def test_features_and_training_range():
    c = dict(power_frac=0.5, cold_pore=2e-6, hot_pore=1e-4, power_shape="cosine")
    assert np.allclose(S.features([c])[0], [0.5, math.log10(2e-6), -4.0, 1.0])
    m = dict(ranges=dict(power=S.POWER_RANGE, cold_pore=S.COLD_PORE_RANGE, hot_pore=S.HOT_PORE_RANGE))
    assert S.in_training_range(m, 0.5, 2e-6, 1e-4)
    assert not S.in_training_range(m, 0.5, 5e-6, 1e-4)
