"""
questions.py -- answers the research questions Q1, Q2, Q4 (Tier 1) and Q3 (Tier 2).

    python questions.py q2      # stability map: power vs cold frit share (no-frit onset, cure share)
    python questions.py q4      # point A across the throttle range
    python questions.py q1      # cold- vs hot-frit control across power density
    python questions.py q3      # flow shift, Z vs U (procedure: docs/model.md A3, A4)

Results go to results/q2_<commit>.{json,md,png} and results/q4_<commit>.{json,md,png}.
These are analyses, not validation checks: there is no pass/fail. Validation.md
records H2 and H3 for Q2 and no hypothesis for Q4; outcomes are reported either way.

PROCEDURE (fixed in this file before the first run; change only as a new, labeled version)

Common to Q2 and Q4
  - Element: Ji et al. preliminary design (nominal), channel and plenum on (Z pattern),
    outlet-referenced at 7.0 MPa, grid 20 x 20 (results/grid_convergence.txt).
  - Mass flow: rule D2, scaled with power, so the mixed outlet stays near 3000 K.
  - Power sweep: validation.md's 2.5% steps, swept from high power to low. Each point starts
    from the previous converged point; a failed point is approached by halving the power
    step (docs/model.md 3a). Points with no converged solution are listed separately and are
    not counted as unstable.
  - Stability at each converged point, reported separately and combined:
      static  = any slice with d(dp_slice)/dm < 0 (Ledinegg), or the residual-Jacobian
                determinant sign differs from the first (highest-power) converged point;
      eigen   = an eigenvalue of the fuel-temperature dynamics with positive real part.
      unstable = static or eigen.
  - Onset = highest swept power that is unstable (same rule as V7). The bracket reported is
    (onset, next stable swept power above it).

Q2 (Kerrebrock's question for this design)
  - Uniform axial power (Kerrebrock's case, as in V7 and V8). Published hot frit.
  - Cold frit share s_c = mean over slices of dp_cold_frit / dp_element, at nominal steady
    state (100% power, uniform power). Varied through the cold frit pore size; cold frit
    porosity held at the published 0.5 (as in rule D8).
  - Shares on the map: 0 (no cold frit), 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70,
    and the published cold frit.
  - Cure share: bisect between the largest mapped share that shows instability and the
    smallest that shows none, running the full power sweep at each trial, until the bracket
    is narrower than 0.005 in share or 8 bisections are done.

Q2 PROCEDURE VERSION 2 (changed before any Q2 result existed; reasons stated)
  - Share reference state. With the channel on, cold frits weaker than the published one
    have no steady solution at 100% uniform power (the bottom slice starves past 5000 K),
    so "share at nominal steady state" was undefined for most of the map. The share is now
    computed with channel effects OFF at 100% uniform power, where every slice is identical
    and a solution always exists. This makes the share a property of the frit design.
  - Cured. "No unstable points" is not "cured" when points have no solution. A share is
    cured only if every swept power converges AND is stable. The bisection uses this.
  - Context map (not the answer): the same map with channel effects off. Kerrebrock's
    parallel-passage analysis gives every passage the same plenum pressure, which is the
    channel-off case, so H2 is compared against this map as well. Reported, labeled.

Q1 (does power density decide whether cold- or hot-frit control works?)
  Ji et al.'s explanation of the debate: cold-frit control for high power density, hot-frit
  control for low. The model has no cross-flow, so only the stability part is tested.
  - Elements: Ji et al. (nominal 20 MW/L, flow by rule D2) and the Tsinghua element
    (params.T19_ELEMENT, nominal 30 MW/L, flow scaled from its published 0.396 kg/s,
    outlet-referenced at 8.0 MPa). Uniform axial power.
  - Frit designs: five per element with the total frit pressure drop held at the published
    design's value. Cold fraction phi = dp_cold / (dp_cold + dp_hot) in
    {published, 0.75, 0.50, 0.25, 1 - published}; the last mirrors the published split
    (as rule D8 does). Set at the element's nominal power with channel effects off, by
    adjusting both pore sizes at the published porosities.
  - Power density 1 to 30 MW/L in 0.5 MW/L steps (59 points), swept high to low with the
    same continuation, halving, and stability rules as Q2. Channel on = the answer;
    channel off = context.
  - Answer rule, per element (fixed before running):
      cold-dominated = published split; hot-dominated = mirrored split.
      half A: hot-dominated is cured (all converged and stable) over 1-5 MW/L, and is not
              cured somewhere in 20-30 MW/L;
      half B: cold-dominated is cured over 20-30 MW/L.
      "Yes" if A and B; "Partly" if exactly one; "No" if neither.
  - H4 (validation.md): at 20-30 MW/L the hot-dominated design is unstable or has no
    solution and the cold-dominated design is stable. Reported per element.

Q4 (point A across the throttle range; no hypothesis)
  - Point A cold frit profile (b = 0.434, c = 1.070, alpha0 = 0.533), cosine axial power.
  - Answer: stable or not at every swept power from 20% to 100% (the operational range).
  - Context, reported but not part of the answer: the same sweep from 5% to 17.5%, and the
    published uniform cold frit with cosine power (Ji et al.'s preliminary design) over the
    same powers. Peak fuel temperature against the 3800 K limit is reported at every point.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import model
import params as P
import steady

RESULTS = Path(__file__).resolve().parent / "results"
STEP = 0.025
SWEEP = np.round(np.arange(1.0, 0.05 - 1e-9, -STEP), 4)          # 100% down to 5%
Q2_SHARES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
Q2_BISECT_TOL, Q2_BISECT_MAX = 0.005, 8
Q4_OPERATIONAL = P.OPERATIONAL_RANGE[0]
COLD_EPS = P.COLD_FRIT_EPS


def commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return h, dirty
    except Exception:
        return "nocommit", True


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def solve(f, cold_profile, power_shape, x0=None, channel=True):
    return steady.solve_element(f, P.nominal_mass_flow(f), power_shape=power_shape,
                                cold_profile=cold_profile, channel=channel, x0=x0)


def _halve_toward(f_from, x_from, f_target, cold_profile, power_shape, channel=True):
    gap = f_from - f_target
    step, f_cur, x_cur, r = gap / 2.0, f_from, x_from, None
    while f_cur - f_target > 1e-12:
        f_try = max(f_target, f_cur - step)
        r = solve(f_try, cold_profile, power_shape, x_cur, channel)
        if r["converged"]:
            f_cur, x_cur = f_try, r["x"]
        else:
            step /= 2.0
            if step < gap / 256.0:
                return r
    return r


def sweep(cold_profile, power_shape, fractions=SWEEP, channel=True):
    points, x0, prev_f, first_sign = [], None, None, None
    for f in sorted(fractions, reverse=True):
        r = solve(f, cold_profile, power_shape, x0, channel)
        if not r["converged"] and prev_f is not None:
            r = _halve_toward(prev_f, x0, f, cold_profile, power_shape, channel)
        row = dict(power_fraction=float(f), MW_per_L=float(f * P.POWER_DENSITY / 1e9))
        if not r["converged"]:
            row.update(status="no_solution", reason=str(r["message"])[:200])
            points.append(row)
            continue
        prev_f, x0 = f, r["x"]
        st = model._point_stability(r)
        if first_sign is None:
            first_sign = st["det_sign"]
        static = st["ledinegg"] or st["det_sign"] != first_sign
        eigen = st["max_real"] > 0.0
        row.update(status="unstable" if (static or eigen) else "stable",
                   static_unstable=bool(static), eigen_unstable=bool(eigen),
                   ledinegg=st["ledinegg"], det_sign=st["det_sign"],
                   max_real=st["max_real"], oscillatory=st["oscillatory"],
                   peak_fuel_T=float(np.nanmax(r["T_fuel_max"])),
                   dp_element_kPa=float(r["dp_element"] / 1e3),
                   flow_max_over_min=float(r["m"].max() / r["m"].min()), path=r["path"])
        points.append(row)
    return points


def onset(points):
    """Highest unstable swept power, the next stable swept power above it, and counts."""
    unstable = [p["power_fraction"] for p in points if p["status"] == "unstable"]
    stable = [p["power_fraction"] for p in points if p["status"] == "stable"]
    no_sol = sorted(p["power_fraction"] for p in points if p["status"] == "no_solution")
    cured = (not unstable) and (not no_sol)
    if not unstable:
        return dict(onset_fraction=None, stable_above=None, n_unstable=0,
                    n_no_solution=len(no_sol), no_solution_fractions=no_sol, cured=cured)
    f = max(unstable)
    above = [s for s in stable if s > f]
    return dict(onset_fraction=f, stable_above=min(above) if above else None,
                n_unstable=len(unstable),
                unstable_fractions=sorted(unstable),
                static_fractions=sorted(p["power_fraction"] for p in points if p.get("static_unstable")),
                eigen_fractions=sorted(p["power_fraction"] for p in points if p.get("eigen_unstable")),
                n_no_solution=len(no_sol), no_solution_fractions=no_sol, cured=False)


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------
def cold_share(pore):
    """Procedure v2: share at 100% uniform power with channel effects off."""
    r = solve(1.0, (lambda s, d=pore: (d, COLD_EPS)), "uniform", channel=False)
    if not r["converged"]:
        raise RuntimeError(f"nominal solve failed at cold pore {pore:.3g} m: {r['message']}")
    return float(np.mean(r["dp_cold_frit"]) / r["dp_element"])


def pore_for_share(target):
    def resid(log_d):
        try:
            return cold_share(float(np.exp(log_d))) - target
        except RuntimeError:
            return 1.0                          # too resistive to solve: share far above target
    return float(np.exp(brentq(resid, np.log(1e-7), np.log(1e-2), xtol=1e-6)))


def q2():
    t0 = time.time()
    columns = [dict(label="none", share=0.0, pore=None, cold_profile="none")]
    for s in Q2_SHARES:
        d = pore_for_share(s)
        columns.append(dict(label=f"{s:.2f}", share=cold_share(d), pore=d,
                            cold_profile=(lambda z, d=d: (d, COLD_EPS))))
    columns.append(dict(label="published", share=cold_share(P.COLD_FRIT_PORE),
                        pore=P.COLD_FRIT_PORE, cold_profile=None))
    for c in columns:
        c["points"] = sweep(c["cold_profile"], "uniform")
        c["onset"] = onset(c["points"])
        c["points_channel_off"] = sweep(c["cold_profile"], "uniform", channel=False)
        c["onset_channel_off"] = onset(c["points_channel_off"])
        print(f"  share {c['share']:.3f}: channel on cured={c['onset']['cured']} "
              f"onset={c['onset']['onset_fraction']} no_solution={c['onset']['n_no_solution']}; "
              f"channel off cured={c['onset_channel_off']['cured']} "
              f"onset={c['onset_channel_off']['onset_fraction']}", flush=True)

    # Cure share (channel on) by bisection between the last uncured share and the first cured
    with_inst = [c for c in columns if not c["onset"]["cured"]]
    without = [c for c in columns if c["onset"]["cured"]]
    cure = None
    if with_inst and without:
        lo_c = max(with_inst, key=lambda c: c["share"])
        hi_c = min((c for c in without if c["share"] > lo_c["share"]), key=lambda c: c["share"],
                   default=None)
        if hi_c is not None:
            lo, hi, trials = lo_c["share"], hi_c["share"], []
            for _ in range(Q2_BISECT_MAX):
                if hi - lo < Q2_BISECT_TOL:
                    break
                mid = 0.5 * (lo + hi)
                d = pore_for_share(mid)
                pts = sweep((lambda z, d=d: (d, COLD_EPS)), "uniform")
                on = onset(pts)
                trials.append(dict(share=cold_share(d), pore=d, onset=on))
                print(f"  bisection share {trials[-1]['share']:.4f}: cured={on['cured']}", flush=True)
                if on["cured"]:
                    hi = mid
                else:
                    lo = mid
            cure = dict(last_unstable_share=lo, first_stable_share=hi, trials=trials)

    h, dirty = commit()
    out = dict(question="Q2", procedure_version=2, commit=h, uncommitted_changes=dirty,
               seconds=time.time() - t0,
               sweep_step=STEP, columns=[{k: v for k, v in c.items() if k != "cold_profile"}
                                         for c in columns], cure=cure)
    return out, h


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------
def q4():
    t0 = time.time()
    point_a = lambda s: (P.COLD_FRIT_PORE, float(P.cold_frit_porosity(s, **P.POINT_A)))
    a = sweep(point_a, "cosine")
    base = sweep(None, "cosine")
    op = [p for p in a if p["power_fraction"] >= Q4_OPERATIONAL - 1e-9]
    answer = dict(
        all_stable_in_operational_range=all(p["status"] == "stable" for p in op),
        unstable_in_operational_range=[p["power_fraction"] for p in op if p["status"] == "unstable"],
        no_solution_in_operational_range=[p["power_fraction"] for p in op if p["status"] == "no_solution"],
        above_fuel_limit_in_operational_range=[p["power_fraction"] for p in op
                                               if p.get("peak_fuel_T", 0) > P.T_LIMIT],
    )
    h, dirty = commit()
    out = dict(question="Q4", commit=h, uncommitted_changes=dirty, seconds=time.time() - t0,
               sweep_step=STEP, operational_from=Q4_OPERATIONAL, answer=answer,
               point_a=a, point_a_onset=onset(a), published_cosine=base,
               published_cosine_onset=onset(base))
    return out, h


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save(out, h, name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}_{h}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    return path


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------
Q1_PD = np.round(np.arange(30.0, 1.0 - 1e-9, -0.5), 3)          # MW/L, high to low
Q1_PHIS = ("published", 0.75, 0.50, 0.25, "mirrored")
Q1_LOW, Q1_HIGH = (1.0, 5.0), (20.0, 30.0)
Q1_ELEMENTS = {
    "ji": dict(name="Ji et al. element", geom=None, pd_nom=P.POWER_DENSITY / 1e9,
               cold=(P.COLD_FRIT_PORE, P.COLD_FRIT_EPS), hot=(P.HOT_FRIT_PORE, P.HOT_FRIT_EPS)),
    "t19": dict(name="Tsinghua element", geom=P.T19_ELEMENT, pd_nom=P.T19_ELEMENT["power_density"] / 1e9,
                cold=(P.T19_ELEMENT["cold_frit_pore"], P.T19_ELEMENT["cold_frit_eps"]),
                hot=(P.T19_ELEMENT["hot_frit_pore"], P.T19_ELEMENT["hot_frit_eps"])),
}


def q1_flow(key, f):
    return P.nominal_mass_flow(f) if key == "ji" else P.T19_ELEMENT["m_dot"] * f


def q1_solve(key, f, cold, hot, x0=None, channel=True):
    e = Q1_ELEMENTS[key]
    return steady.solve_element(f, q1_flow(key, f), geom=e["geom"], power_shape="uniform",
                                cold_profile=(lambda s, c=cold: c), hot=hot, channel=channel, x0=x0)


def q1_frit_drops(key, cold, hot):
    r = q1_solve(key, 1.0, cold, hot, channel=False)
    if not r["converged"]:
        raise RuntimeError(f"nominal solve failed: {r['message']}")
    return float(np.mean(r["dp_cold_frit"])), float(np.mean(r["dp_hot_frit"]))


def q1_design(key, phi):
    """Pores giving cold fraction phi with the published total frit drop (nominal, channel off)."""
    e = Q1_ELEMENTS[key]
    dc0, dh0 = q1_frit_drops(key, e["cold"], e["hot"])
    total, phi_pub = dc0 + dh0, dc0 / (dc0 + dh0)
    if phi == "published":
        return dict(phi=phi_pub, cold=e["cold"], hot=e["hot"], total_frit_dp=total)
    target = 1.0 - phi_pub if phi == "mirrored" else float(phi)
    cold, hot = e["cold"], e["hot"]

    def fit(which, want):
        def resid(log_d):
            c = (float(np.exp(log_d)), cold[1]) if which == "cold" else cold
            h = (float(np.exp(log_d)), hot[1]) if which == "hot" else hot
            try:
                dc, dh = q1_frit_drops(key, c, h)
            except RuntimeError:
                return 50.0                      # too resistive to solve: far above target
            return np.log(dc if which == "cold" else dh) - np.log(want)
        return float(np.exp(brentq(resid, np.log(1e-7), np.log(1e-2), xtol=1e-7)))

    for _ in range(3):                           # the two frits interact weakly; repeat
        cold = (fit("cold", target * total), cold[1])
        hot = (fit("hot", (1.0 - target) * total), hot[1])
    dc, dh = q1_frit_drops(key, cold, hot)
    return dict(phi=dc / (dc + dh), cold=cold, hot=hot, total_frit_dp=dc + dh)


def q1_sweep(key, cold, hot, channel):
    e = Q1_ELEMENTS[key]
    fracs = [pd / e["pd_nom"] for pd in Q1_PD]
    points, x0, prev_f, first_sign = [], None, None, None
    for f in fracs:
        r = q1_solve(key, f, cold, hot, x0, channel)
        if not r["converged"] and prev_f is not None:
            gap = prev_f - f; step, f_cur, x_cur = gap / 2.0, prev_f, x0
            while f_cur - f > 1e-12:
                f_try = max(f, f_cur - step)
                r = q1_solve(key, f_try, cold, hot, x_cur, channel)
                if r["converged"]:
                    f_cur, x_cur = f_try, r["x"]
                else:
                    step /= 2.0
                    if step < gap / 256.0:
                        break
        row = dict(power_fraction=float(f), MW_per_L=float(f * e["pd_nom"]))
        if not r["converged"]:
            row.update(status="no_solution", reason=str(r["message"])[:200])
            points.append(row)
            continue
        prev_f, x0 = f, r["x"]
        st = model._point_stability(r)
        if first_sign is None:
            first_sign = st["det_sign"]
        static = st["ledinegg"] or st["det_sign"] != first_sign
        eigen = st["max_real"] > 0.0
        row.update(status="unstable" if (static or eigen) else "stable",
                   static_unstable=bool(static), eigen_unstable=bool(eigen),
                   max_real=st["max_real"], peak_fuel_T=float(np.nanmax(r["T_fuel_max"])),
                   dp_element_kPa=float(r["dp_element"] / 1e3))
        points.append(row)
    return points


def q1_band_cured(points, band):
    sel = [p for p in points if band[0] - 1e-9 <= p["MW_per_L"] <= band[1] + 1e-9]
    return bool(sel) and all(p["status"] == "stable" for p in sel)


def q1_verdict(designs):
    hot = [d for d in designs if d["label"] == "mirrored"][0]["points"]
    cold = [d for d in designs if d["label"] == "published"][0]["points"]
    half_a = q1_band_cured(hot, Q1_LOW) and not q1_band_cured(hot, Q1_HIGH)
    half_b = q1_band_cured(cold, Q1_HIGH)
    hi = lambda pts: [p for p in pts if Q1_HIGH[0] - 1e-9 <= p["MW_per_L"] <= Q1_HIGH[1] + 1e-9]
    h4 = (all(p["status"] in ("unstable", "no_solution") for p in hi(hot))
          and all(p["status"] == "stable" for p in hi(cold)))
    answer = "Yes" if (half_a and half_b) else ("Partly" if (half_a or half_b) else "No")
    return dict(answer=answer, half_a=half_a, half_b=half_b, h4=h4)


Q1_PARTS = RESULTS / "q1_parts"


def q1_step():
    """Next unfinished (element, design) piece of Q1; assemble when all are done."""
    Q1_PARTS.mkdir(parents=True, exist_ok=True)
    for key in Q1_ELEMENTS:
        for lab in Q1_PHIS:
            f = Q1_PARTS / f"{key}_{lab}.json"
            if f.exists():
                continue
            t = time.time()
            dsg = q1_design(key, lab)
            part = dict(element=key, label=str(lab), phi=dsg["phi"], cold=dsg["cold"], hot=dsg["hot"],
                        total_frit_dp=dsg["total_frit_dp"], commit=commit()[0],
                        points=q1_sweep(key, dsg["cold"], dsg["hot"], True),
                        points_channel_off=q1_sweep(key, dsg["cold"], dsg["hot"], False))
            part["onset"] = onset(part["points"]); part["onset_channel_off"] = onset(part["points_channel_off"])
            f.write_text(json.dumps(part, default=float))
            print(f"{key} {lab}: phi {part['phi']:.3f}, cold pore {part['cold'][0]*1e6:.2f} um, hot pore "
                  f"{part['hot'][0]*1e6:.2f} um; channel on cured={part['onset']['cured']} "
                  f"(unstable {part['onset']['n_unstable']}, no solution {part['onset']['n_no_solution']}); "
                  f"channel off cured={part['onset_channel_off']['cured']} "
                  f"(unstable {part['onset_channel_off']['n_unstable']})  [{time.time()-t:.0f} s]")
            return True
    parts = {k: [json.loads((Q1_PARTS / f"{k}_{lab}.json").read_text()) for lab in Q1_PHIS]
             for k in Q1_ELEMENTS}
    commits = {d["commit"] for v in parts.values() for d in v}
    if commits != {commit()[0]}:
        sys.exit(f"q1 parts come from commits {sorted(map(str, commits))}, but the code is at "
                 f"{commit()[0]}. Delete results/q1_parts/ and rerun.")
    h, dirty = commit()
    out = dict(question="Q1", commit=h, uncommitted_changes=dirty, pd_step=0.5,
               low_band=Q1_LOW, high_band=Q1_HIGH,
               elements={k: dict(name=Q1_ELEMENTS[k]["name"], designs=v, verdict=q1_verdict(v))
                         for k, v in parts.items()})
    p = save(out, h, "q1")
    print(f"wrote {p.relative_to(RESULTS.parent)}")
    return False


# ---------------------------------------------------------------------------
# Q2 in resumable steps (same computation as q2(); each step saves its part)
# ---------------------------------------------------------------------------
PARTS = RESULTS / "q2_parts"


def _column(label):
    if label == "none":
        c = dict(label="none", share=0.0, pore=None, cold_profile="none")
    elif label == "published":
        c = dict(label="published", share=cold_share(P.COLD_FRIT_PORE), pore=P.COLD_FRIT_PORE,
                 cold_profile=None)
    else:
        d = pore_for_share(float(label))
        c = dict(label=label, share=cold_share(d), pore=d, cold_profile=(lambda z, d=d: (d, COLD_EPS)))
    c["points"] = sweep(c["cold_profile"], "uniform")
    c["onset"] = onset(c["points"])
    c["points_channel_off"] = sweep(c["cold_profile"], "uniform", channel=False)
    c["onset_channel_off"] = onset(c["points_channel_off"])
    c["commit"] = commit()[0]
    return {k: v for k, v in c.items() if k != "cold_profile"}


def q2_step():
    """Do the next unfinished piece of Q2 and save it. Returns False when all done."""
    PARTS.mkdir(parents=True, exist_ok=True)
    labels = ["none"] + [f"{s:.2f}" for s in Q2_SHARES] + ["published"]
    for lab in labels:
        f = PARTS / f"col_{lab}.json"
        if not f.exists():
            t = time.time(); c = _column(lab)
            f.write_text(json.dumps(c, default=float))
            print(f"column {lab}: share {c['share']:.4f}, channel on cured={c['onset']['cured']} "
                  f"(unstable {c['onset']['n_unstable']}, no solution {c['onset']['n_no_solution']}); "
                  f"channel off cured={c['onset_channel_off']['cured']} "
                  f"(unstable {c['onset_channel_off']['n_unstable']}, no solution "
                  f"{c['onset_channel_off']['n_no_solution']})  [{time.time()-t:.0f} s]")
            return True
    cols = [json.loads((PARTS / f"col_{lab}.json").read_text()) for lab in labels]
    commits = {c.get("commit") for c in cols}
    if commits != {commit()[0]}:
        sys.exit(f"q2 parts come from commits {sorted(map(str, commits))}, but the code is at "
                 f"{commit()[0]}. Delete results/q2_parts/ and rerun.")
    bis = PARTS / "bisection.json"
    state = json.loads(bis.read_text()) if bis.exists() else None
    if state is None:
        unc = [c for c in cols if not c["onset"]["cured"]]
        cur = [c for c in cols if c["onset"]["cured"]]
        if not unc or not cur:
            state = dict(lo=None, hi=None, trials=[], done=True)
        else:
            lo = max(c["share"] for c in unc)
            hi = min((c["share"] for c in cur if c["share"] > lo), default=None)
            state = dict(lo=lo, hi=hi, trials=[], done=hi is None)
    if not state["done"] and state["hi"] - state["lo"] >= Q2_BISECT_TOL and len(state["trials"]) < Q2_BISECT_MAX:
        mid = 0.5 * (state["lo"] + state["hi"])
        d = pore_for_share(mid)
        on = onset(sweep((lambda z, d=d: (d, COLD_EPS)), "uniform"))
        state["trials"].append(dict(share=cold_share(d), pore=d, onset=on))
        if on["cured"]:
            state["hi"] = mid
        else:
            state["lo"] = mid
        print(f"bisection share {state['trials'][-1]['share']:.4f}: cured={on['cured']} "
              f"-> bracket [{state['lo']:.4f}, {state['hi']:.4f}]")
        bis.write_text(json.dumps(state, default=float))
        return True
    state["done"] = True
    bis.write_text(json.dumps(state, default=float))
    h, dirty = commit()
    cure = (None if state["lo"] is None or state["hi"] is None else
            dict(last_unstable_share=state["lo"], first_stable_share=state["hi"], trials=state["trials"]))
    out = dict(question="Q2", procedure_version=2, commit=h, uncommitted_changes=dirty,
               sweep_step=STEP, columns=cols, cure=cure)
    p = save(out, h, "q2")
    print(f"wrote {p.relative_to(RESULTS.parent)}")
    return False



# ---------------------------------------------------------------------------
# Q3 (docs/model.md A3 and A4)
# ---------------------------------------------------------------------------
Q3_CASES = (("off", False, "Z"), ("Z", True, "Z"), ("U", True, "U"))


def single_point_status(res):
    """A4 single-point rule: Ledinegg or a positive eigenvalue real part = unstable."""
    if not res["converged"]:
        return ("out_of_range" if res.get("reason") == "out_of_range" else "no_solution"), None
    st = model._point_stability(res)
    return ("unstable" if (st["ledinegg"] or st["max_real"] > 0.0) else "stable"), st


def flow_metrics(m):
    m = np.asarray(m, dtype=float)
    return dict(flow_cov=float(np.std(m) / np.mean(m)), flow_max_over_min=float(np.max(m) / np.min(m)))


def q3_element(name, geom, m_total):
    out = dict(name=name, cases={})
    for label, channel, pattern in Q3_CASES:
        t = time.perf_counter()
        res = steady.solve_element(1.0, m_total, geom=geom, power_shape="uniform", channel=channel,
                                   pattern=pattern, pressure_ref="outlet")
        status, st = single_point_status(res)
        c = dict(status=status, channel=channel, pattern=pattern, path=res.get("path"),
                 seconds=time.perf_counter() - t)
        if res["converged"]:
            c.update(flow_metrics(res["m"]), m=[float(v) for v in res["m"]],
                     peak_fuel_T=float(np.nanmax(res["T_fuel_max"])),
                     dp_element_kPa=float(res["dp_element"]) / 1e3, stability=st)
        else:
            c.update(message=str(res.get("message")))
        out["cases"][label] = c
    out["h7"] = q3_h7(out["cases"])
    return out


def q3_h7(cases):
    z, u = cases["Z"], cases["U"]
    if "flow_cov" not in z or "flow_cov" not in u:
        return dict(verdict="not answerable", shift_size=z.get("flow_cov"),
                    note="Z or U has no steady solution inside the property range")
    return dict(verdict="holds" if u["flow_cov"] < z["flow_cov"] else "does not hold",
                shift_size=z["flow_cov"], cov_Z=z["flow_cov"], cov_U=u["flow_cov"])


def q3():
    import elements
    h, dirty = commit()
    t0 = time.time()
    big = elements.large_reactor(1.2, "baseline")
    out = dict(question="Q3", commit=h, uncommitted_changes=dirty, procedure="docs/model.md A3, A4",
               elements=[q3_element("Ji et al. nominal", None, P.nominal_mass_flow(1.0)),
                         q3_element("Large reactor (1.2 m, baseline)", big, big["m_dot"])])
    out["seconds"] = time.time() - t0
    return out, h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "q2":                       # fresh full run in one process
        import shutil
        shutil.rmtree(PARTS, ignore_errors=True)
        while q2_step():
            pass
    elif which == "q2-step":                # one step per call (for short time limits)
        q2_step()
    elif which == "q1":                     # fresh full run in one process
        import shutil
        shutil.rmtree(Q1_PARTS, ignore_errors=True)
        while q1_step():
            pass
    elif which == "q1-step":
        q1_step()
    elif which == "q4":
        out, h = q4()
        p = save(out, h, "q4")
        print(f"wrote {p.relative_to(RESULTS.parent)} ({out['seconds']:.0f} s)")
    elif which == "q3":
        out, h = q3()
        p = save(out, h, "q3")
        print(f"wrote {p.relative_to(RESULTS.parent)} ({out['seconds']:.0f} s)")
    else:
        sys.exit("usage: python questions.py q1|q1-step|q2|q2-step|q3|q4")
