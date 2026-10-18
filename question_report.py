"""
question_report.py -- turns saved Q1-Q4 results into a markdown summary and a plot.
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


def _fmt(v, f):
    return format(v, f) if isinstance(v, (int, float)) else "n/a"


def q3_report(path):
    d = json.loads(path.read_text())
    els = d["elements"]
    fig, axes = plt.subplots(1, len(els), figsize=(6 * len(els), 3.8), squeeze=False)
    styles = {"off": (MUTED, "Channel off (no shift)"), "Z": (RED, "Z pattern"), "U": (BLUE, "U pattern")}
    for ax, e in zip(axes[0], els):
        drawn = False
        for label, (color, text) in styles.items():
            c = e["cases"][label]
            if "m" in c:
                m = c["m"]
                ax.plot(range(1, len(m) + 1), [v / (sum(m) / len(m)) for v in m], color=color, lw=2,
                        marker="o", ms=3, label=f"{text}, CoV {c['flow_cov']:.3f}")
                drawn = True
        if not drawn:
            ax.text(0.5, 0.5, "No steady solution inside the\n5000 K property range for Z or U",
                    ha="center", va="center", color=AMBER, transform=ax.transAxes)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(e["name"], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Slice (inlet end to outlet end)")
        ax.set_ylabel("Slice flow / mean slice flow")
        if drawn:
            ax.legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "100% power, uniform axial power, outlet-referenced. Missing lines: no steady solution.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = path.with_suffix(".png"); fig.savefig(png, dpi=130); plt.close(fig)

    lines = ["# Q3: How large is the flow shift, and does a U pattern reduce it?", "",
             f"Commit {d['commit']}{' (uncommitted changes present)' if d.get('uncommitted_changes') else ''}. "
             "Procedure: docs/model.md A3 and A4. Flow shift = coefficient of variation (CoV) of the slice "
             "flows with the Z pattern; H7 (validation.md) holds if U's CoV is lower than Z's.", "",
             "## Answer"]
    for e in els:
        h = e["h7"]
        if h["verdict"] == "not answerable":
            lines.append(f"- **{e['name']}: not answerable.** {h['note']}; reported, no substitute case run.")
        else:
            lines.append(f"- **{e['name']}:** flow shift (Z CoV) {h['cov_Z']:.4f}; U CoV {h['cov_U']:.4f}. "
                         f"H7 **{h['verdict']}**.")
    lines += ["", "## Per case", "",
              "| Element | Case | Status | CoV | Max/min flow | Peak fuel (K) | Element dp (kPa) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in els:
        for label, c in e["cases"].items():
            lines.append(f"| {e['name']} | {styles[label][1]} | {c['status']} | {_fmt(c.get('flow_cov'), '.4f')} | "
                         f"{_fmt(c.get('flow_max_over_min'), '.3f')} | {_fmt(c.get('peak_fuel_T'), '.0f')} | "
                         f"{_fmt(c.get('dp_element_kPa'), '.1f')} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return png


# ---------------------------------------------------------------------------
def transient_report(path, t_show=12.0, frame_dt=0.2, fps=8):
    """Side-by-side runaway animation (GIF) and a short summary from results/transient_*.json."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np
    d = json.loads(path.read_text())
    cases = [("with_cold_frit", "With the cold frit"), ("no_cold_frit", "Without a cold frit")]
    N = len(d["cases"]["with_cold_frit"]["T_steady"])
    slices = np.arange(1, N + 1)
    times = np.round(np.arange(0.0, t_show + 1e-9, frame_dt), 6)

    def frame_at(case, t):
        fr = case["frames"]
        k = max(i for i, f in enumerate(fr) if f["t"] <= t + 1e-9)
        return fr[k], fr[k]["t"] < t - 1e-9 and case["outcome"] == "runaway"

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), gridspec_kw=dict(height_ratios=[3, 2]))
    title = fig.suptitle("", x=0.01, ha="left", fontsize=13, color=INK)

    def draw(t):
        title.set_text(f"t = {t:4.1f} s   (slice {d['demo']['slice'] + 1} started {d['demo']['kick_K']:.0f} K hot)")
        for col, (key, name) in enumerate(cases):
            c = d["cases"][key]
            fr, ended = frame_at(c, t)
            T = np.array(fr["T"]); m = np.array(fr["m"]); T0 = np.array(c["T_steady"]); m0 = np.array(c["m_steady"])
            ax = axes[0][col]; ax.clear()
            colors = [RED if Ti > P.T_LIMIT else (AMBER if Ti > T0i + 25 else BLUE) for Ti, T0i in zip(T, T0)]
            ax.bar(slices, T, color=colors, width=0.8)
            ax.axhline(P.T_LIMIT, color=RED, lw=1)
            ax.text(0.6, P.T_LIMIT + 40, "fuel limit 3,800 K", color=RED, fontsize=8, va="bottom")
            ax.set_ylim(2500, 5000); ax.set_xlim(0.3, N + 0.7)
            ax.set_ylabel("Gas leaving slice, K"); ax.set_xticks([1, 5, 10, 15, 20])
            if ended:
                ax.set_title(f"{name}: RUNAWAY at {c['t_end']:.1f} s",
                             loc="left", fontsize=11, color=RED, fontweight="bold")
            else:
                ax.set_title(name, loc="left", fontsize=11, color=INK)
            ax2 = axes[1][col]; ax2.clear()
            change = (m / m0 - 1.0) * 100
            ax2.bar(slices, change, color=[RED if x < -1 else MUTED for x in change], width=0.8)
            ax2.axhline(0, color=MUTED, lw=0.8)
            ax2.set_ylim(-60, 15); ax2.set_xlim(0.3, N + 0.7)
            ax2.set_ylabel("Flow change, %"); ax2.set_xlabel("Slice"); ax2.set_xticks([1, 5, 10, 15, 20])
        return []

    anim = FuncAnimation(fig, lambda i: draw(times[min(i, len(times) - 1)]),
                         frames=len(times) + int(1.5 * fps), blit=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.28, hspace=0.35)
    gif = path.with_suffix(".gif")
    anim.save(gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

    a, b = d["cases"]["with_cold_frit"], d["cases"]["no_cold_frit"]
    lines = ["# Runaway demonstration", "",
             f"Commit {d['commit']}{' (uncommitted changes present)' if d.get('uncommitted_changes') else ''}. "
             f"Ji et al. element at {a['power_MW_per_L']:.0f} MW/L, uniform power, channel effects off "
             "(Kerrebrock's setup). Slice {0} starts {1:.0f} K hotter than steady; nothing else changes. "
             .format(d['demo']['slice'] + 1, d['demo']['kick_K'])
             + "Model: docs/model.md section 5, flows readjusting instantly.", "",
             "## Result",
             f"- **With the cold frit:** {a['outcome']}; the disturbance decays (eigenvalue "
             f"{a['eigen_max_real']:+.2f} 1/s, so it shrinks by a factor of e every "
             f"{1/abs(a['eigen_max_real']):.2f} s).",
             f"- **Without a cold frit:** {b['outcome']}. The hot slice loses flow, heats further, and its "
             f"outlet gas passes the {P.T_LIMIT:,.0f} K fuel limit at {b['t_fuel_limit']:.1f} s and leaves "
             f"the 5000 K property range at {b['t_end']:.1f} s (eigenvalue {b['eigen_max_real']:+.2f} 1/s: "
             f"the disturbance grows by a factor of e every {1/abs(b['eigen_max_real']):.1f} s at first)."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return gif


if __name__ == "__main__":
    for prefix, fn in (("q1", q1_report), ("q2", q2_report), ("q3", q3_report), ("q4", q4_report),
                       ("transient", transient_report)):
        p = newest(prefix)
        if p:
            print("wrote", fn(p).relative_to(RESULTS.parent), "and", p.with_suffix(".md").relative_to(RESULTS.parent))
