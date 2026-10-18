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


def _q1_pts(rule):
    """Synthetic Q1 points over 1-30 MW/L; rule(pd) gives the status."""
    return [dict(MW_per_L=float(pd), status=rule(pd)) for pd in Q.Q1_PD]


def _designs(hot_rule, cold_rule):
    return [dict(label="published", points=_q1_pts(cold_rule)),
            dict(label="mirrored", points=_q1_pts(hot_rule))]


def test_q1_yes_when_power_density_decides():
    hot = lambda pd: "stable" if pd <= 10 else "unstable"       # hot works low, fails high
    v = Q.q1_verdict(_designs(hot, lambda pd: "stable"))
    assert v["answer"] == "Yes" and v["h4"]


def test_q1_partly_when_hot_fails_everywhere():
    v = Q.q1_verdict(_designs(lambda pd: "unstable", lambda pd: "stable"))
    assert v["answer"] == "Partly" and not v["half_a"] and v["half_b"] and v["h4"]


def test_q1_no_solution_counts_as_failure_for_h4():
    v = Q.q1_verdict(_designs(lambda pd: "no_solution", lambda pd: "stable"))
    assert v["h4"] and not v["half_a"]
