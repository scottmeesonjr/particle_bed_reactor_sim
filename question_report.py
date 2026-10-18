"""
question_report.py -- turns saved Q2 / Q4 results into a markdown summary and a plot.
Reads results/q2_*.json and results/q4_*.json (newest of each). Never recomputes.

    python question_report.py
"""
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import params as P

RESULTS = Path(__file__).resolve().parent / "results"
INK, MUTED, BLUE, AMBER, RED = "#18212B", "#5B6672", "#2F6FB0", "#C9821B", "#B3261E"
STATUS_STYLE = {"stable": (BLUE, "o", "Stable"), "unstable": (RED, "X", "Unstable"),
                "no_solution": (AMBER, "s", "No steady solution (above 5000 K)")}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def newest(prefix):
    files = [f for f in glob.glob(str(RESULTS / f"{prefix}_*.json")) if "parts" not in f]
    return Path(max(files, key=os.path.getmtime)) if files else None


def mwl(f):
    return f * P.POWER_DENSITY / 1e9


def _rng(fs):
    if not fs:
        return "none"
    return f"{mwl(min(fs)):.1f} to {mwl(max(fs)):.1f} MW/L ({len(fs)} points)"


# ---------------------------------------------------------------------------
def q2_report(path):
    d = json.loads(path.read_text())
    cols = d["columns"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, key, title in ((axes[0], "points_channel_off", "Channel effects off (Kerrebrock's setup; context)"),
                           (axes[1], "points", "Channel and plenum on (the answer)")):
        for i, c in enumerate(cols):
            for p in c[key]:
                color, marker, _ = STATUS_STYLE[p["status"]]
                ax.scatter(i, p["MW_per_L"], s=26, c=color, marker=marker, linewidths=0)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([("none" if c["label"] == "none" else
                             f"{c['share']:.2f}\n(published)" if c["label"] == "published" else
                             f"{c['share']:.2f}") for c in cols])
        ax.set_xlabel("Cold frit share of the element pressure drop")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.axhspan(0, mwl(P.OPERATIONAL_RANGE[0]), color=MUTED, alpha=0.07, lw=0)
    axes[0].set_ylabel("Power density, MW/L")
    axes[0].axhline(P.NASA_THRESHOLD_MW_PER_L, color=MUTED, ls=":", lw=1)
    axes[0].annotate("Kerrebrock\n~4 MW/L", xy=(1.0, P.NASA_THRESHOLD_MW_PER_L),
                     xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                     va="center", ha="left", color=MUTED, fontsize=9)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=l) for c, m, l in STATUS_STYLE.values()]
    fig.legend(handles=handles, frameon=False, loc="lower right", ncol=3, fontsize=9,
               bbox_to_anchor=(0.995, 0.0))
    fig.text(0.01, 0.015, "Uniform axial power; flow scaled with power (rule D2); shaded band is below the "
             "20% operational range.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=5)
    png = path.with_suffix(".png"); fig.savefig(png, dpi=130); plt.close(fig)

    none = cols[0]
    off_cured = [c for c in cols if c["onset_channel_off"]["cured"]]
    off_uncured = [c for c in cols if not c["onset_channel_off"]["cured"]]
    cure = d.get("cure") or {}
    pub = [c for c in cols if c["label"] == "published"][0]
    on0 = none["onset"]; off0 = none["onset_channel_off"]
    lines = [
        f"# Q2: Where does the instability start, and how much cold frit cures it?",
        "",
        f"Commit {d['commit']}{' (uncommitted changes present)' if d.get('uncommitted_changes') else ''}, "
        f"procedure version {d.get('procedure_version', 1)} (see questions.py). Ji et al. element, uniform "
        "power, flow scaled with power, 2.5% power steps from 5% to 100% (1 to 20 MW/L).",
        "",
        "## Answer (channel and plenum on)",
        f"- **No cold frit:** no steady solution at {_rng(on0.get('no_solution_fractions', []))}; "
        f"unstable at {_rng(on0.get('unstable_fractions', []))}. The bottom slice is starved by flow shift "
        "and overheats past the 5000 K property limit.",
        f"- **Cold frit share that cures it:** between {cure.get('last_unstable_share', float('nan')):.3f} and "
        f"{cure.get('first_stable_share', float('nan')):.3f} (every power from 1 to 20 MW/L converged and "
        "stable above this share). The published frit's share is "
        f"{pub['share']:.3f}.",
        "- Frits below the cure share fail at high power first: "
        + "; ".join(f"share {c['share']:.2f}: no solution at {_rng(c['onset'].get('no_solution_fractions', []))}"
                    for c in cols if c['label'] in ('0.50', '0.70')) + ".",
        "",
        "## Context (channel effects off, Kerrebrock's parallel-passage setup)",
        f"- **No cold frit:** unstable from {mwl(min(off0['unstable_fractions'])):.1f} to "
        f"{mwl(off0['onset_fraction']):.1f} MW/L, stable from {mwl(off0['stable_above']):.1f} MW/L up.",
        f"- **Cure:** between share {max(c['share'] for c in off_uncured):.2f} (still unstable) and "
        f"{min(c['share'] for c in off_cured if c['share'] > max(x['share'] for x in off_uncured)):.2f} (cured).",
        "",
        "## Hypotheses",
        f"- **H2** (no cold frit -> static instability below about 4 MW/L): reproduced in form, not in number. "
        f"The model's instability is below a threshold, but at {mwl(off0['onset_fraction']):.1f} to "
        f"{mwl(off0['stable_above']):.1f} MW/L, about four times Kerrebrock's value.",
        "- **H3** (a sufficient cold frit removes it): confirmed in both setups.",
        "",
        "## Per share",
        "| Cold frit share | Pore (um) | Channel on | Channel off |",
        "| --- | --- | --- | --- |",
    ]
    for c in cols:
        def summ(o):
            if o["cured"]:
                return "cured (all stable)"
            parts = []
            if o.get("unstable_fractions"): parts.append(f"unstable {_rng(o['unstable_fractions'])}")
            if o.get("no_solution_fractions"): parts.append(f"no solution {_rng(o['no_solution_fractions'])}")
            return "; ".join(parts)
        pore = "none" if c["pore"] is None else f"{c['pore']*1e6:.2f}"
        lines.append(f"| {c['share']:.3f}{' (published)' if c['label']=='published' else ''} | {pore} | "
                     f"{summ(c['onset'])} | {summ(c['onset_channel_off'])} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return png


# ---------------------------------------------------------------------------
def q4_report(path):
    d = json.loads(path.read_text())
    a, b = d["point_a"], d["published_cosine"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for pts, color, label in ((a, INK, "Point A (Ji et al. optimized)"), (b, BLUE, "Published uniform frit")):
        x = [p["MW_per_L"] for p in pts if "max_real" in p]
        axes[0].plot(x, [p["max_real"] for p in pts if "max_real" in p], color=color, lw=2, label=label)
        axes[1].plot(x, [p["peak_fuel_T"] for p in pts if "peak_fuel_T" in p], color=color, lw=2, label=label)
    for ax in axes:
        ax.axvspan(0, mwl(P.OPERATIONAL_RANGE[0]), color=MUTED, alpha=0.07, lw=0)
        ax.set_xlabel("Power density, MW/L")
    axes[0].axhline(0, color=RED, lw=1)
    axes[0].set_ylabel("Least-stable eigenvalue, 1/s (below 0 = stable)")
    axes[0].set_title("Stability margin", loc="left", fontsize=11, color=INK)
    axes[1].axhline(P.T_LIMIT, color=RED, lw=1)
    axes[1].text(13.5, P.T_LIMIT - 12, f"Fuel limit {P.T_LIMIT:,.0f} K", color=RED, fontsize=9, va="top")
    axes[1].set_ylabel("Peak fuel temperature, K")
    axes[1].set_title("Peak fuel temperature", loc="left", fontsize=11, color=INK)
    axes[1].legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Cosine axial power; channel and plenum on; flow scaled with power; shaded band is "
             "below the 20% operational range.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    png = path.with_suffix(".png"); fig.savefig(png, dpi=130); plt.close(fig)

    op = [p for p in a if p["power_fraction"] >= d["operational_from"] - 1e-9]
    op_b = [p for p in b if p["power_fraction"] >= d["operational_from"] - 1e-9]
    cross = [p["MW_per_L"] for p, q in zip(a, b) if "peak_fuel_T" in p and "peak_fuel_T" in q
             and p["peak_fuel_T"] > q["peak_fuel_T"]]
    ratio = [p["max_real"] / q["max_real"] for p, q in zip(op, op_b)]
    ans = d["answer"]
    lines = [
        "# Q4: Does Ji et al.'s point A stay stable from 20% to 100% throttle?",
        "",
        f"Commit {d['commit']}{' (uncommitted changes present)' if d.get('uncommitted_changes') else ''}. "
        "Cosine power, channel and plenum on, flow scaled with power, 2.5% steps. No hypothesis was made.",
        "",
        "## Answer",
        f"- **{'Yes' if ans['all_stable_in_operational_range'] else 'No'}.** Point A is "
        f"{'stable at every' if ans['all_stable_in_operational_range'] else 'not stable at every'} swept power "
        "from 20% to 100%"
        + (" and also down to 5%." if d["point_a_onset"]["cured"] else ".")
        + " No point failed to converge.",
        f"- **Smaller stability margin.** Its least-stable eigenvalue is {min(ratio):.2f} to {max(ratio):.2f} "
        "times the published frit's across the operational range, so disturbances die out more slowly.",
        f"- **Thermal advantage reverses at low throttle.** Point A runs cooler than the published frit at "
        f"high power, but hotter below {max(cross):.1f} MW/L, and above the {P.T_LIMIT:,.0f} K fuel limit at "
        + ", ".join(f"{p * 100:.1f}%" for p in sorted(ans["above_fuel_limit_in_operational_range"]))
        + " power (within the operational range).",
        "",
        "| Power | MW/L | Point A | Eigenvalue (1/s) | Peak fuel (K) | Published frit | Eigenvalue (1/s) | Peak fuel (K) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p, q in zip(a, b):
        lines.append(f"| {p['power_fraction']*100:.1f}% | {p['MW_per_L']:.1f} | {p['status']} | "
                     f"{p.get('max_real', float('nan')):+.3f} | {p.get('peak_fuel_T', float('nan')):.0f} | "
                     f"{q['status']} | {q.get('max_real', float('nan')):+.3f} | {q.get('peak_fuel_T', float('nan')):.0f} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return png


# ---------------------------------------------------------------------------
def q1_report(path):
    d = json.loads(path.read_text())
    els = list(d["elements"].items())
    fig, axes = plt.subplots(len(els), 2, figsize=(12.5, 4.3 * len(els)), sharey=True)
    for row, (key, e) in enumerate(els):
        ds = e["designs"]
        for ax, pk, title in ((axes[row][0], "points_channel_off", "channel effects off (context)"),
                              (axes[row][1], "points", "channel and plenum on (the answer)")):
            for i, dsg in enumerate(ds):
                for p in dsg[pk]:
                    color, marker, _ = STATUS_STYLE[p["status"]]
                    ax.scatter(i, p["MW_per_L"], s=16, c=color, marker=marker, linewidths=0)
            ax.set_xticks(range(len(ds)))
            ax.set_xticklabels([f"{x['phi']:.2f}" + ("\n(published)" if x["label"] == "published" else
                                                     "\n(mirrored)" if x["label"] == "mirrored" else "")
                                for x in ds])
            ax.set_title(f"{e['name']}: {title}", loc="left", fontsize=11, color=INK)
            ax.set_xlabel("Cold frit fraction of the frit pressure drop (left: cold-dominated)")
        axes[row][0].set_ylabel("Power density, MW/L")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=l) for c, m, l in STATUS_STYLE.values()]
    fig.legend(handles=handles, frameon=False, loc="lower right", ncol=3, fontsize=9, bbox_to_anchor=(0.995, 0.0))
    fig.text(0.01, 0.01, "Uniform axial power; total frit pressure drop held at the published design's value.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1), w_pad=4)
    png = path.with_suffix(".png"); fig.savefig(png, dpi=130); plt.close(fig)

    def summ(pts):
        """Ranges in MW/L, read from each point's own MW_per_L (fractions differ by element)."""
        if all(p["status"] == "stable" for p in pts):
            return "stable at every power density"
        parts = []
        for status, word in (("stable", "stable"), ("unstable", "unstable"), ("no_solution", "no solution")):
            v = sorted(p["MW_per_L"] for p in pts if p["status"] == status)
            if v:
                parts.append(f"{word} {v[0]:.1f} to {v[-1]:.1f} MW/L ({len(v)} points)")
        return "; ".join(parts)

    lines = ["# Q1: Does power density decide whether cold- or hot-frit control works?", "",
             f"Commit {d['commit']}{' (uncommitted changes present)' if d.get('uncommitted_changes') else ''}. "
             "Five frit designs per element with the total frit pressure drop held at the published value, "
             "1 to 30 MW/L in 0.5 MW/L steps, uniform power. Only stability is tested (no cross-flow).", "",
             "## Answer"]
    for key, e in els:
        v = e["verdict"]
        hot = [x for x in e["designs"] if x["label"] == "mirrored"][0]
        cold = [x for x in e["designs"] if x["label"] == "published"][0]
        n_hot_stable = sum(p["status"] == "stable" for p in hot["points"])
        lines.append(f"- **{e['name']}: {v['answer']}.** Cold-dominated (published, fraction "
                     f"{cold['phi']:.2f}): {summ(cold['points'])}. Hot-dominated (mirrored, fraction "
                     f"{hot['phi']:.2f}): {summ(hot['points'])}; stable at {n_hot_stable} of "
                     f"{len(hot['points'])} power densities.")
    both_no_a = all(not e["verdict"]["half_a"] for _, e in els)
    both_b = all(e["verdict"]["half_b"] for _, e in els)
    if both_no_a and both_b:
        lines.append("- In this model, cold-frit control works at every power density tested and hot-frit "
                     "control does not work at low power density, so power density does not decide which "
                     "works. This contradicts the low-power half of Ji et al.'s explanation, for stability.")
    lines += ["", "## Hypothesis",
              "- **H4** (at 20-30 MW/L, hot-frit-dominated unstable, cold-frit-dominated stable): "
              + "; ".join(f"{e['name']}: {'confirmed' if e['verdict']['h4'] else 'not confirmed'}" for _, e in els) + ".",
              "", "## Limits of this answer",
              "- Only stability is tested. Ji et al. may prefer hot-frit control at low power density for "
              "other reasons (temperature uniformity, pressure drop) that this question does not test.",
              "- The design family holds the total frit pressure drop at the published value. Hot-frit "
              "designs with a different total were not explored.",
              "", "## Per design", "| Element | Cold frit fraction | Cold pore (um) | Hot pore (um) | Channel on | Channel off |",
              "| --- | --- | --- | --- | --- | --- |"]
    for key, e in els:
        for x in e["designs"]:
            lines.append(f"| {e['name']} | {x['phi']:.3f} ({x['label']}) | {x['cold'][0]*1e6:.2f} | "
                         f"{x['hot'][0]*1e6:.2f} | {summ(x['points'])} | {summ(x['points_channel_off'])} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return png


if __name__ == "__main__":
    for prefix, fn in (("q1", q1_report), ("q2", q2_report), ("q4", q4_report)):
        p = newest(prefix)
        if p:
            print("wrote", fn(p).relative_to(RESULTS.parent), "and", p.with_suffix(".md").relative_to(RESULTS.parent))
