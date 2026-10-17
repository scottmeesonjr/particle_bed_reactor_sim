"""
runner.py -- runs the pre-registered validation checks and writes result files.

Why this exists: every pass/fail rule below is written as code and committed
BEFORE the solver produces any result. The Git history then proves the rules
were fixed first (ground rules 2, 3, 6, 11).

Usage
  python runner.py               run every Tier 1 check with the real solver
  python runner.py --channel-off fallback rule: checks that need the channel
                                 model are marked "not run"
  python runner.py --self-test   exercise the runner with a STAND-IN solver
                                 (nothing is written to results/)

SOLVER CONTRACT -- the teammate's code must provide a module `model` with:

  run_steady(case) -> dict
      Steady solution for one case. Must return:
        "status"      "stable", "runaway", "out_of_range", or "no_solution"
        "dp_total"    total element pressure drop, Pa
        "peak_fuel_T" K
        "flags"       list of correlation-range warnings
  find_onset(case, power_fractions, method) -> dict
      Sweeps power (mass flow follows rule D2 with throttling) and reports
      where static instability is found. method = "slope" or "eigen".
      Must return:
        "unstable_fractions"  list of power fractions found unstable
        "onset_fraction"      highest unstable fraction, or None if none
  nominal_slice(element) -> dict
      One slice at the element's nominal conditions, returning at least
      "dp_cold_frit" and "dp_hot_frit" in Pa (used by rule D8).

A `case` is a dict:
  element       element dict (J19B_ELEMENT or params.T19_ELEMENT style)
  power_shape   "uniform" or "cosine"
  cold_frit     "published", "none", or {"profile": params.POINT_A}
  m_dot         number in kg/s, or "rule_D2" (energy balance, scaled with power)
  channel_on    bool
  pressure_ref  "outlet" or "inlet" (where the element's pressure applies)
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np

import params as p

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Fixed settings for the sweeps (pre-registered here)
# ---------------------------------------------------------------------------
SWEEP_STEP = 0.025                                  # power fraction step (validation.md: 2.5%)
SWEEP = np.round(np.arange(p.POWER_SWEEP[0], p.POWER_SWEEP[1] + 1e-9, SWEEP_STEP), 4)
DP_TOLERANCE = 0.25                                 # +/-25% (v3 Section 6)
DP_UNCHANGED = 0.02                                 # "changes by less than 2%"
NASA_FACTOR = 3.0                                   # "within a factor of 3"

# "Strong cold frit" for V8 (pre-registered here, before any run):
# the published Ji et al. cold frit (2 um pores, porosity 0.5).
STRONG_COLD_FRIT = "published"

# The Ji et al. preliminary element in the same dict format as params.T19_ELEMENT
J19B_ELEMENT = dict(
    L=p.L_ELEMENT, r_hot_chan=p.R_HOT_CHAN, t_hot_frit=p.T_HOT_FRIT, t_bed=p.T_BED,
    t_cold_frit=p.T_COLD_FRIT, t_plenum=p.T_PLENUM, d_particle=p.D_PARTICLE, eps_bed=p.EPS_BED,
    cold_frit_pore=p.COLD_FRIT_PORE, cold_frit_eps=p.COLD_FRIT_EPS,
    hot_frit_pore=p.HOT_FRIT_PORE, hot_frit_eps=p.HOT_FRIT_EPS,
    power_density=p.POWER_DENSITY, m_dot=p.M_DOT_PUBLISHED, T_inlet=p.T_INLET, P=p.P_OUTLET,
)


CHANNEL_ON = True     # set False by --channel-off (validation.md fallback: V6-V9 run with channel off)


def case(element, **kw):
    base = dict(element=element, power_shape="uniform", cold_frit="published",
                m_dot=element["m_dot"], channel_on=CHANNEL_ON, pressure_ref="outlet")
    base.update(kw)
    return base


def within(value, target, tol):
    return abs(value - target) / target <= tol


# ---------------------------------------------------------------------------
# The checks. Each returns (verdict, details). Verdicts: "pass", "fail",
# "not run". Nothing here may change after the first validation run.
# ---------------------------------------------------------------------------
def v5(m, refs):
    out = {}
    for ref in refs:
        r = m.run_steady(case(p.T19_ELEMENT, pressure_ref=ref))
        ok = within(r["dp_total"], 1189.2e3, DP_TOLERANCE) and r["status"] == "stable"
        out[ref] = dict(verdict="pass" if ok else "fail", dp_kPa=r["dp_total"] / 1e3,
                        status=r["status"], flags=r["flags"])
    # Both pressure references are reported; no combined verdict (validation.md).
    return "see per-reference", out


def v6(m, refs):
    from rules import hot_frit_dominated
    variant, d8 = hot_frit_dominated(p.T19_ELEMENT, m.nominal_slice)
    out = {"D8": d8}
    for ref in refs:
        r = m.run_steady(case(variant, pressure_ref=ref))
        # validation.md amendment A1: no solution counts only after the continuation rule
        ok = r["status"] in ("runaway", "out_of_range", "no_solution")
        out[ref] = dict(verdict="pass" if ok else "fail", status=r["status"],
                        peak_fuel_T=r["peak_fuel_T"], flags=r["flags"])
    return "see per-reference", out


def _no_cold_frit_onset(m, method):
    return m.find_onset(case(J19B_ELEMENT, cold_frit="none", m_dot="rule_D2"), SWEEP, method)


def v7(m, refs):
    o = _no_cold_frit_onset(m, "slope")
    f = o["onset_fraction"]
    if f is None:
        return "fail", dict(onset_MW_per_L=None, note="no static instability found in the sweep")
    onset = f * p.POWER_DENSITY / 1e9
    lo, hi = p.NASA_THRESHOLD_MW_PER_L / NASA_FACTOR, p.NASA_THRESHOLD_MW_PER_L * NASA_FACTOR
    return ("pass" if lo <= onset <= hi else "fail"), dict(onset_MW_per_L=onset, allowed=[lo, hi])


def v8(m, refs):
    o = m.find_onset(case(J19B_ELEMENT, cold_frit=STRONG_COLD_FRIT, m_dot="rule_D2"), SWEEP, "slope")
    ok = len(o["unstable_fractions"]) == 0
    return ("pass" if ok else "fail"), dict(unstable_MW_per_L=[f * p.POWER_DENSITY / 1e9
                                                               for f in o["unstable_fractions"]])


def v9(m, refs):
    a = _no_cold_frit_onset(m, "slope")["onset_fraction"]
    b = _no_cold_frit_onset(m, "eigen")["onset_fraction"]
    if a is None and b is None:
        return "pass", dict(note="neither method found instability; they agree")
    if a is None or b is None:
        return "fail", dict(slope=a, eigen=b)
    ok = abs(a - b) <= SWEEP_STEP + 1e-9
    return ("pass" if ok else "fail"), dict(slope_fraction=a, eigen_fraction=b, step=SWEEP_STEP)


def _j19b(m, **kw):
    return m.run_steady(case(J19B_ELEMENT, m_dot=p.M_DOT_PUBLISHED, **kw))


def v10(m, refs):
    r = _j19b(m)
    ok = within(r["dp_total"], 529.8e3, DP_TOLERANCE)
    return ("pass" if ok else "fail"), dict(dp_kPa=r["dp_total"] / 1e3, flags=r["flags"])


def v11(m, refs):
    u, c = _j19b(m), _j19b(m, power_shape="cosine")
    ok = c["peak_fuel_T"] > u["peak_fuel_T"] and within(c["dp_total"], u["dp_total"], DP_UNCHANGED)
    return ("pass" if ok else "fail"), dict(
        peak_uniform_K=u["peak_fuel_T"], peak_cosine_K=c["peak_fuel_T"],
        dp_uniform_kPa=u["dp_total"] / 1e3, dp_cosine_kPa=c["dp_total"] / 1e3)


def v12(m, refs):
    base = _j19b(m, power_shape="cosine")
    a = _j19b(m, power_shape="cosine", cold_frit={"profile": p.POINT_A})
    ok = (a["peak_fuel_T"] < base["peak_fuel_T"] and a["dp_total"] < base["dp_total"]
          and within(a["dp_total"], 414.4e3, DP_TOLERANCE))
    return ("pass" if ok else "fail"), dict(
        peak_base_K=base["peak_fuel_T"], peak_A_K=a["peak_fuel_T"],
        dp_base_kPa=base["dp_total"] / 1e3, dp_A_kPa=a["dp_total"] / 1e3)


CHECKS = {"V5": v5, "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10, "V11": v11, "V12": v12}
# Tier 2 (V1-V4) and Tier 3 (V13) checks get added here, and committed,
# before they are first run.


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def git_state():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    return dict(commit=git("rev-parse", "--short", "HEAD"),
                uncommitted_changes=bool(git("status", "--porcelain")))


def run_all(model, channel_off=False):
    global CHANNEL_ON
    CHANNEL_ON = not channel_off
    refs = ("outlet", "inlet")
    results = []
    for vid, spec in p.VALIDATION.items():
        row = dict(id=vid, tier=spec["tier"], target=spec["target"], pass_rule=spec["pass_rule"])
        if vid not in CHECKS:
            row.update(verdict="not run", reason=f"Tier {spec['tier']} check not attempted")
        elif channel_off and spec["needs_channel"]:
            row.update(verdict="not run", reason="fallback rule: channel model not ready at checkpoint")
        else:
            t0 = dt.datetime.now()
            try:
                verdict, details = CHECKS[vid](model, refs)
                row.update(verdict=verdict, details=details)
            except Exception as e:      # a crash is reported, never hidden
                row.update(verdict="error", reason=f"{type(e).__name__}: {e}")
            row["seconds"] = (dt.datetime.now() - t0).total_seconds()
        results.append(row)
    return results


def write_results(results, channel_off):
    RESULTS_DIR.mkdir(exist_ok=True)
    state = git_state()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"validation_{state['commit'] or 'nocommit'}_{stamp}"
    meta = dict(params_version=p.PARAMS_VERSION, channel_off=channel_off,
                run_at=dt.datetime.now().isoformat(timespec="seconds"), **state)
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(dict(meta=meta, results=results), indent=2, default=float))
    lines = [f"# Validation results {stamp}", "",
             f"Commit {meta['commit']}, params {meta['params_version']}"
             + (", UNCOMMITTED CHANGES PRESENT" if meta["uncommitted_changes"] else ""), "",
             "| Check | Tier | Verdict | Details |", "| --- | --- | --- | --- |"]
    for r in results:
        detail = r.get("reason") or json.dumps(r.get("details", {}), default=float)
        lines.append(f"| {r['id']} | {r['tier']} | {r['verdict']} | {detail[:200]} |")
    (RESULTS_DIR / f"{name}.md").write_text("\n".join(lines) + "\n")
    return name, meta


def print_table(results):
    for r in results:
        print(f"{r['id']:>4}  tier {r['tier']}  {r['verdict']:<18} {r.get('reason', '')}")


# ---------------------------------------------------------------------------
# Self-test with a STAND-IN solver. Checks the runner's plumbing only.
# ---------------------------------------------------------------------------
class _StandIn:
    """Made-up numbers. Never used for results."""
    def run_steady(self, c):
        dp = 520e3 if c["element"] is J19B_ELEMENT else 1150e3
        peak = 3400.0 + (300.0 if c["power_shape"] == "cosine" else 0.0)
        if isinstance(c["cold_frit"], dict):
            dp, peak = 420e3, peak - 400.0
        hot_dominated = c["element"].get("cold_frit_pore", 0) > 5e-6
        return dict(status="runaway" if hot_dominated else "stable",
                    dp_total=dp, peak_fuel_T=peak, flags=[])

    def find_onset(self, c, fractions, method):
        bad = [f for f in fractions if c["cold_frit"] == "none" and f <= 0.21]
        return dict(unstable_fractions=bad, onset_fraction=max(bad) if bad else None)

    def nominal_slice(self, el):
        return dict(dp_cold_frit=900e3 * (2e-6 / el["cold_frit_pore"]) ** 2,
                    dp_hot_frit=30e3 * (1e-4 / el["hot_frit_pore"]) ** 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-off", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        print("SELF-TEST with a stand-in solver. These verdicts mean nothing.\n")
        print_table(run_all(_StandIn(), channel_off=a.channel_off))
        raise SystemExit(0)

    try:
        import model
    except ImportError:
        print("No solver yet (model.py not found). Every check is 'not run'.\n")
        for vid, spec in p.VALIDATION.items():
            print(f"{vid:>4}  tier {spec['tier']}  not run")
        raise SystemExit(0)

    state = git_state()
    if state["uncommitted_changes"]:
        print("WARNING: uncommitted changes. Commit first so results trace to exact code.\n")
    results = run_all(model, channel_off=a.channel_off)
    name, meta = write_results(results, a.channel_off)
    print_table(results)
    print(f"\nWrote results/{name}.json and .md (commit {meta['commit']})")
