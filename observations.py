"""
observations.py -- what the large-reactor cases that DID solve show (docs/model.md A5).

    python observations.py      # results/tier2_obs_<commit>.{json,md}

V1-V4 are "not evaluable" (validation.md A2): each needs at least one case with no steady
solution inside the 5000 K property range. This script re-solves exactly the V3 and V4
cases and records the ones that converged. These are observations: no pass rule is applied,
nothing here changes a verdict, and none of it is counted.
"""
import json

import model
import runner
from questions import commit, RESULTS

REFS = ("outlet", "inlet")
V3_LENGTHS = tuple(runner.V3_TARGETS)                    # 0.8, 1.0, 1.2 m (Energies Table 4)
V4_VARIANTS = ("baseline", "cold_dominated", "hot_dominated")


def _row(r):
    out = dict(status=r["status"])
    if r["status"] in ("stable", "runaway"):
        out.update(dp_kPa=r["dp_total"] / 1e3, beta=r["beta"], peak_fuel_T=r["peak_fuel_T"],
                   flow_cov=r["flow_cov"])
    return out


def pct_diff(value, target):
    return 100.0 * (value - target) / target


def observe():
    h, dirty = commit()
    out = dict(commit=h, uncommitted_changes=dirty, procedure="docs/model.md A5 (observations, not counted)",
               v3={}, v4={})
    for ref in REFS:
        out["v3"][ref] = {}
        for L in V3_LENGTHS:
            row = _row(runner._large(model, L=L, ref=ref))
            row["published_dp_kPa"] = runner.V3_TARGETS[L] / 1e3
            if "dp_kPa" in row:
                row["dp_diff_pct"] = pct_diff(row["dp_kPa"], row["published_dp_kPa"])
            out["v3"][ref][str(L)] = row
        out["v4"][ref] = {v: _row(runner._large(model, variant=v, ref=ref)) for v in V4_VARIANTS}
    return out, h


def _f(v, fmt):
    return format(v, fmt) if isinstance(v, (int, float)) else "n/a"


STATUS = {"stable": "stable", "runaway": "runaway", "out_of_range": "no solution below 5000 K",
          "no_solution": "no solution"}


def write_md(path, d):
    lines = ["# Tier 2 observations: the large-reactor cases that solved", "",
             f"Commit {d['commit']}{' (uncommitted changes present)' if d.get('uncommitted_changes') else ''}. "
             "docs/model.md A5. **Observations, not counted.** V1-V4 stay \"not evaluable\"; no pass rule "
             "is applied here.", "",
             "## V3: core height", "",
             "| Reference | Core height | Status | dp (kPa) | Published dp (kPa) | Difference | beta | Peak fuel (K) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ref, rows in d["v3"].items():
        for L, r in rows.items():
            lines.append(f"| {ref} | {float(L) * 100:.0f} cm | {STATUS.get(r['status'], r['status'])} | "
                         f"{_f(r.get('dp_kPa'), '.1f')} | {r['published_dp_kPa']:.1f} | "
                         f"{_f(r.get('dp_diff_pct'), '+.1f')}{'%' if 'dp_diff_pct' in r else ''} | "
                         f"{_f(r.get('beta'), '.3f')} | {_f(r.get('peak_fuel_T'), '.0f')} |")
    lines += ["", "## V4: frit-dominance variants (1.2 m)", "",
              "| Reference | Variant | Status | Peak fuel (K) | dp (kPa) | Flow CoV |",
              "| --- | --- | --- | --- | --- | --- |"]
    for ref, rows in d["v4"].items():
        for v, r in rows.items():
            lines.append(f"| {ref} | {v.replace('_', ' ')} | {STATUS.get(r['status'], r['status'])} | "
                         f"{_f(r.get('peak_fuel_T'), '.0f')} | {_f(r.get('dp_kPa'), '.1f')} | "
                         f"{_f(r.get('flow_cov'), '.4f')} |")
    lines += ["", "## Limits",
              "- Our gas temperatures run hotter than the Tsinghua papers' (NIST enthalpy), so cases near the "
              "5000 K property limit fail for us that may solve for them.",
              "- A missing case means the check cannot be judged. It is not evidence for or against the paper."]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    d, h = observe()
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"tier2_obs_{h}.json"
    p.write_text(json.dumps(d, indent=2, default=float))
    write_md(p.with_suffix(".md"), d)
    print(f"wrote {p.relative_to(RESULTS.parent)} and .md")
