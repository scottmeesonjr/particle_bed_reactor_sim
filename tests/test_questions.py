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


# --- Q3 (docs/model.md A3, A4) ---
def test_q3_flow_metrics_zero_for_equal_split():
    m = Q.flow_metrics([0.1] * 20)
    assert abs(m["flow_cov"]) < 1e-12 and abs(m["flow_max_over_min"] - 1.0) < 1e-12


def test_q3_h7_holds_when_u_is_more_uniform():
    h = Q.q3_h7(dict(Z=dict(flow_cov=0.05), U=dict(flow_cov=0.02)))
    assert h["verdict"] == "holds" and h["shift_size"] == 0.05


def test_q3_h7_does_not_hold_when_u_is_not_lower():
    assert Q.q3_h7(dict(Z=dict(flow_cov=0.02), U=dict(flow_cov=0.02)))["verdict"] == "does not hold"


def test_q3_h7_not_answerable_without_solution():
    h = Q.q3_h7(dict(Z=dict(status="out_of_range"), U=dict(flow_cov=0.02)))
    assert h["verdict"] == "not answerable"


# --- Tier 2 observations (docs/model.md A5) ---
def test_observations_mirror_the_v3_v4_cases_and_apply_no_rule():
    import observations as O
    import runner
    assert O.V3_LENGTHS == tuple(runner.V3_TARGETS)
    assert O.REFS == ("outlet", "inlet")
    row = O._row(dict(status="out_of_range", dp_total=float("nan"), peak_fuel_T=float("nan")))
    assert row == dict(status="out_of_range")                 # nothing recorded for a missing case
    assert "verdict" not in O._row(dict(status="stable", dp_total=5e5, beta=1.1, peak_fuel_T=3000.0, flow_cov=0.01))
    assert abs(O.pct_diff(110.0, 100.0) - 10.0) < 1e-12
