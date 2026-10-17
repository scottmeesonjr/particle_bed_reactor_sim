"""
make_slice_curve.py -- slice pressure drop vs. mass flow for one nominal slice
(hypothesis H1 in validation.md). No model choices here; everything comes from
params.py and slice.slice_dp.

Fixed: N = 20, dz = L/N, q_slice = nominal element power / N,
p_in = P_OUTLET, T_in = T_INLET, n_r = 20, published hot frit.
Swept: m_dot from 0.05x to 3x the nominal per-slice flow, log spacing, 60 points.
Cases: (a) no cold frit, (b) published cold frit.

Writes results/slice_curve_<commit>.csv and .png. The commit gets a "-dirty"
suffix if the working tree has uncommitted changes.
"""
import csv
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import params as P
from slice import slice_dp

N = 20
N_R = 20
N_POINTS = 60
RATIO_RANGE = (0.05, 3.0)

DZ = P.L_ELEMENT / N
Q_SLICE = P.total_power() / N
M_NOM = P.nominal_mass_flow() / N
HOT = (P.HOT_FRIT_PORE, P.HOT_FRIT_EPS)
CASES = {
    "no_cold_frit": dict(cold=None, label="(a) No cold frit"),
    "published_cold_frit": dict(cold=(P.COLD_FRIT_PORE, P.COLD_FRIT_EPS),
                                label="(b) Published cold frit"),
}
FLAG_KEYS = ("re_m_low", "re_ht_low", "out_of_range", "pressure_collapse", "fuel_over_limit")
VALUE_KEYS = ("dp", "dp_cold_frit", "dp_bed", "dp_hot_frit", "T_gas_out", "T_fuel_max", "p_out")

# Chart colors: dataviz skill default palette (light mode), slots 1 and 2 in order.
COLORS = {"no_cold_frit": "#2a78d6", "published_cold_frit": "#eb6834"}
STYLES = {"no_cold_frit": "-", "published_cold_frit": "--"}
INK, INK_2, INK_MUTED, GRID, SURFACE = "#1f1f1e", "#52524e", "#8a8984", "#e6e5e0", "#fcfcfb"


def run(m_dot, cold):
    return slice_dp(m_dot, Q_SLICE, P.P_OUTLET, P.T_INLET, cold, HOT, DZ, N_R)


def commit_tag():
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", ".",
                            ":(exclude)__pycache__", ":(exclude)**/__pycache__", ":(exclude)results"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def interior_minima(m, dp, ok, cold):
    """Grid points that are strict local minima with valid neighbours on both
    sides, refined by a bounded scalar minimisation in log(m)."""
    from scipy.optimize import minimize_scalar
    out = []
    for i in range(1, len(m) - 1):
        if ok[i - 1] and ok[i] and ok[i + 1] and dp[i] < dp[i - 1] and dp[i] < dp[i + 1]:
            r = minimize_scalar(lambda x: run(math.exp(x), cold)["dp"],
                                bounds=(math.log(m[i - 1]), math.log(m[i + 1])),
                                method="bounded", options=dict(xatol=1e-6))
            out.append((math.exp(r.x), r.fun))
    return out


def crossings(m, dp, ok, target, cold):
    """Every m where dp = target, between adjacent valid grid points (brentq)."""
    out = []
    for i in range(len(m) - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        a, b = dp[i] - target, dp[i + 1] - target
        if a == 0.0:
            out.append(m[i])
        elif a * b < 0.0:
            out.append(brentq(lambda x: run(x, cold)["dp"] - target, m[i], m[i + 1], xtol=1e-12))
    if ok[-1] and dp[-1] == target:
        out.append(m[-1])
    return out


def bad_spans(m, bad):
    """Contiguous runs of flagged grid points, widened to the geometric midpoint
    of the neighbouring unflagged points (log axis)."""
    spans, i = [], 0
    while i < len(m):
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(m) and bad[j + 1]:
            j += 1
        lo = m[i] if i == 0 else math.sqrt(m[i - 1] * m[i])
        hi = m[j] if j == len(m) - 1 else math.sqrt(m[j] * m[j + 1])
        spans.append((lo, hi))
        i = j + 1
    return spans


if __name__ == "__main__":
    tag = commit_tag()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    ratios = np.logspace(math.log10(RATIO_RANGE[0]), math.log10(RATIO_RANGE[1]), N_POINTS)
    m = M_NOM * ratios
    nominal = run(M_NOM, CASES["published_cold_frit"]["cold"])
    dp_nom = nominal["dp"]

    data, summary = {}, {}
    for key, case in CASES.items():
        rows = [run(x, case["cold"]) for x in m]
        dp = np.array([r["dp"] for r in rows])
        oor = np.array([r["out_of_range"] for r in rows])
        col = np.array([r["pressure_collapse"] for r in rows])
        ok = ~(oor | col)
        data[key] = dict(rows=rows, dp=dp, oor=oor, col=col, ok=ok)
        summary[key] = dict(
            minima=interior_minima(m, dp, ok, case["cold"]),
            crossings=crossings(m, dp, ok, dp_nom, case["cold"]),
            oor_spans=bad_spans(m, oor),
            col_spans=bad_spans(m, col),
            # Is dp strictly increasing over the valid points?
            monotonic_up=bool(np.all(np.diff(dp[ok]) > 0)),
            valid_ratio_range=(float(ratios[ok].min()), float(ratios[ok].max())) if ok.any() else None,
        )

    # ---------------- CSV ----------------
    csv_path = out_dir / f"slice_curve_{tag}.csv"
    with csv_path.open("w", newline="") as fh:
        fh.write(f"# make_slice_curve.py, commit {tag}\n")
        fh.write(f"# N={N}, dz={DZ} m, q_slice={Q_SLICE} W, p_in={P.P_OUTLET} Pa, "
                 f"T_in={P.T_INLET} K, n_r={N_R}, hot frit={HOT}, m_nom={M_NOM} kg/s\n")
        fh.write(f"# nominal slice dp (published cold frit, m_nom) = {dp_nom} Pa\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["case", "m_dot_kg_s", "m_over_m_nom", *VALUE_KEYS, *FLAG_KEYS])
        for key in CASES:
            for x, rt, r in zip(m, ratios, data[key]["rows"]):
                w.writerow([key, f"{x:.8e}", f"{rt:.6f}",
                            *(f"{r[k]:.8e}" for k in VALUE_KEYS), *(r[k] for k in FLAG_KEYS)])

    # ---------------- Plot ----------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_MUTED,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Shaded regions (union of both cases; drawn per span so overlaps stay clear)
    oor_spans = sorted({s for k in CASES for s in summary[k]["oor_spans"]})
    for i, (lo, hi) in enumerate(oor_spans):
        ax.axvspan(lo, hi, color="#d9d8d2", alpha=0.6, linewidth=0,
                   label="Outside property data (out_of_range)" if i == 0 else None)
    col_spans = sorted({s for k in CASES for s in summary[k]["col_spans"]})
    for i, (lo, hi) in enumerate(col_spans):
        ax.axvspan(lo, hi, facecolor="none", edgecolor=INK_MUTED, hatch="//", linewidth=0,
                   label="Pressure collapse" if i == 0 else None)

    for key, case in CASES.items():
        d = data[key]
        dp_plot = np.where(d["ok"], d["dp"], np.nan) / 1e3
        ax.plot(m, dp_plot, STYLES[key], color=COLORS[key], linewidth=2, label=case["label"])
        for mm, dpm in summary[key]["minima"]:
            ax.plot(mm, dpm / 1e3, "o", ms=9, mfc=SURFACE, mec=COLORS[key], mew=2)
            ax.annotate(f"minimum\n{mm*1e3:.2f} g/s, {dpm/1e3:.1f} kPa", (mm, dpm / 1e3),
                        xytext=(0, 14), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=INK_2)
        for mc in summary[key]["crossings"]:
            ax.plot(mc, dp_nom / 1e3, "D", ms=8, mfc=COLORS[key], mec=SURFACE, mew=1.5)
            ax.annotate(f"{mc*1e3:.2f} g/s", (mc, dp_nom / 1e3), xytext=(6, 8),
                        textcoords="offset points", fontsize=9, color=INK_2)

    ax.axhline(dp_nom / 1e3, color=INK_MUTED, linewidth=1.2, linestyle=":",
               label=f"Nominal slice dp = {dp_nom/1e3:.1f} kPa")
    ax.axvline(M_NOM, color=GRID, linewidth=1.2, zorder=0)
    ax.annotate("nominal flow", (M_NOM, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=9, color=INK_MUTED)

    ax.set_xlabel("Slice mass flow m_dot (kg/s, log scale)")
    ax.set_ylabel("Slice pressure drop dp (kPa, log scale)")
    fig.text(0.01, 0.975, "Slice pressure drop vs. mass flow, nominal slice at fixed power",
             color=INK, fontsize=12, va="top")
    fig.text(0.01, 0.935, f"N = {N}, q_slice = {Q_SLICE/1e6:.3f} MW, p_in = {P.P_OUTLET/1e6:.1f} MPa, "
                     f"T_in = {P.T_INLET:.0f} K, n_r = {N_R}. Commit {tag}.",
            fontsize=8.5, color=INK_MUTED, va="top")
    leg = ax.legend(loc="upper left", frameon=True, fontsize=9)
    leg.get_frame().set_edgecolor(GRID)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    png_path = out_dir / f"slice_curve_{tag}.png"
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)

    # ---------------- Text summary ----------------
    print(f"commit {tag}; m_nom = {M_NOM:.6f} kg/s; nominal dp = {dp_nom/1e3:.3f} kPa")
    for key, case in CASES.items():
        s = summary[key]
        print(f"\n{case['label']}")
        print(f"  valid m/m_nom range: {s['valid_ratio_range']}")
        print("  out_of_range spans (m/m_nom): "
              + (", ".join(f"{lo/M_NOM:.3f}-{hi/M_NOM:.3f}" for lo, hi in s["oor_spans"]) or "none"))
        print("  pressure_collapse spans (m/m_nom): "
              + (", ".join(f"{lo/M_NOM:.3f}-{hi/M_NOM:.3f}" for lo, hi in s["col_spans"]) or "none"))
        print("  interior minima: " + (", ".join(
            f"m = {mm:.6f} kg/s ({mm/M_NOM:.3f} m_nom), dp = {d/1e3:.3f} kPa"
            for mm, d in s["minima"]) or "none"))
        print(f"  dp strictly increasing over valid points: {s['monotonic_up']}")
        print("  crossings of nominal dp: " + (", ".join(
            f"{c:.6f} kg/s ({c/M_NOM:.3f} m_nom)" for c in s["crossings"]) or "none"))
    print(f"\nwrote {csv_path} and {png_path}")
