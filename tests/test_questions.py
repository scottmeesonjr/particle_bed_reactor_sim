"""Verification of the Q2/Q4 summary rules in questions.py (procedure version 2)."""
import questions as Q


def pts(statuses):
    return [dict(power_fraction=f, status=s, static_unstable=(s == "unstable"),
                 eigen_unstable=(s == "unstable")) for f, s in statuses]


def test_all_stable_is_cured():
    o = Q.onset(pts([(1.0, "stable"), (0.5, "stable")]))
    assert o["cured"] and o["onset_fraction"] is None


def test_no_solution_is_not_cured():
    # The flaw procedure v2 fixed: no unstable points, but some had no solution.
    o = Q.onset(pts([(1.0, "no_solution"), (0.5, "stable")]))
    assert not o["cured"] and o["no_solution_fractions"] == [1.0]


def test_onset_is_highest_unstable_with_stable_bracket():
    o = Q.onset(pts([(1.0, "stable"), (0.75, "stable"), (0.5, "unstable"), (0.25, "unstable")]))
    assert o["onset_fraction"] == 0.5 and o["stable_above"] == 0.75 and not o["cured"]
