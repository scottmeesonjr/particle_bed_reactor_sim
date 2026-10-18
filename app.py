"""
app.py -- the judge-facing simulator.

Run:  streamlit run app.py

Results come from the real solver through model.py (its docstring defines
what solve() accepts and returns). The app always calls the solver live
(ground rule 8); results are cached only for identical inputs.
"""
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

import params as p
import model
from model import solve

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
INK, MUTED, BLUE, AMBER, RED, PAPER = "#18212B", "#5B6672", "#2F6FB0", "#C9821B", "#B3261E", "#F2F4F6"
T_SCALE = (200.0, 4000.0)                # fixed so runs can be compared by eye

# Cold hydrogen blue -> dark -> incandescent orange -> white-hot
BED_CMAP = LinearSegmentedColormap.from_list("bed", [
    (0.00, "#3A78B8"), (0.18, "#24364F"), (0.30, "#33262A"), (0.48, "#8C1D18"),
    (0.66, "#D2561A"), (0.84, "#F2A93B"), (1.00, "#FFF4D6")])

st.set_page_config(page_title="Particle bed stability simulator", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=Barlow:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption, label, p, li {{
  font-family: 'Barlow', 'Helvetica Neue', Arial, sans-serif;
}}
h1, h2, h3 {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600; color: {INK}; }}
h1 {{ font-size: 2.6rem; line-height: 1.05; margin-bottom: 0.2rem; }}
.lede {{ color: {MUTED}; font-size: 1.05rem; max-width: 70ch; margin-bottom: 1.2rem; }}
.status {{ display: flex; align-items: baseline; gap: 1.2rem; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
           border-left: 6px solid var(--c); background: white; }}
.status .word {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 2.2rem; color: var(--c); }}
.status .why {{ color: {MUTED}; }}
.metric .value {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 2rem; color: {INK}; line-height: 1.1; }}
.metric .value.over {{ color: {RED}; }}
.metric .label {{ color: {MUTED}; font-size: 0.95rem; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})



# ---------------------------------------------------------------------------
# Validation results (read from the newest runner.py output; never recomputed here)
# ---------------------------------------------------------------------------
def latest_validation():
    import glob, json, os
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "results", "validation_*.json")), key=os.path.getmtime)
    return json.load(open(files[-1])) if files else None


def verdict_text(r):
    v = r.get("verdict", "")
    if v == "see per-reference" and isinstance(r.get("details"), dict):
        parts = [f"{ref}: {d['verdict']}" for ref, d in r["details"].items()
                 if isinstance(d, dict) and "verdict" in d]
        if parts and all(x.endswith(": pass") for x in parts):
            return "Pass (" + ", ".join(ref for ref, d in r["details"].items()
                                        if isinstance(d, dict) and "verdict" in d) + ")"
        return "Mixed (" + "; ".join(parts) + ")" if parts else v
    return {"pass": "Pass", "fail": "Fail", "not run": "Not run", "error": "Error"}.get(v, v)


def detail_text(r):
    import json
    if r.get("reason"):
        return r["reason"]
    d = r.get("details", {})
    out = []
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk not in ("verdict", "flags")}
        if isinstance(v, float):
            v = round(v, 3)
        out.append(f"{k}: {json.dumps(v, default=lambda x: round(float(x), 3))}")
    return "; ".join(out)[:300]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "no commit found"


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Reactor")
    power_pct = st.slider("Power, % of nominal", 5, 100, 100, step=5)
    pd_mw_l = p.POWER_DENSITY * power_pct / 100 / 1e9
    if power_pct < 20:
        st.caption(f"{pd_mw_l:.0f} MW/L. Below the 20–100% operational range.")
    else:
        st.caption(f"{pd_mw_l:.0f} MW/L. Within the 20–100% operational range.")
    shape_label = st.radio("Axial power shape", ["Uniform", "Cosine, peaks mid-element"])
    power_shape = "uniform" if shape_label == "Uniform" else "cosine"

    st.header("Cold frit")
    cold_on = st.toggle("Cold frit installed", value=True)
    use_a = st.checkbox("Use the Ji et al. optimized frit (point A)", value=False, disabled=not cold_on)
    cold_eps = st.slider("Cold frit porosity", 0.30, 0.70, p.COLD_FRIT_EPS, step=0.01,
                         disabled=(not cold_on) or use_a,
                         help="Higher porosity means less resistance. The published design uses 0.50.")

    st.header("Hot frit")
    hot_pore_um = st.select_slider("Hot frit pore size, µm", options=[10, 20, 50, 100, 200, 500, 1000],
                                   value=int(p.HOT_FRIT_PORE * 1e6))

    st.header("Disturbance")
    disturbance_pct = st.slider("Extra power in one slice, %", 0, 30, 0)
    disturbed = st.slider("Disturbed slice", 1, p.N_SLICES, p.N_SLICES // 2, disabled=disturbance_pct == 0)

    st.header("Model")
    channel_ready = getattr(model, "CHANNEL_AVAILABLE", True)
    channel_on = st.toggle("Include hot-channel and plenum pressure", value=channel_ready,
                           disabled=not channel_ready,
                           help="Turn off to see the element with uniform plenum and channel pressure.")
    if not channel_ready:
        st.caption("The channel and plenum model isn't built yet. Every slice sees the same pressure drop.")


@st.cache_data(show_spinner="Solving…")
def run(power_fraction, power_shape, cold_on, cold_eps, use_a, hot_pore, dist, dist_slice, channel_on):
    return solve(power_fraction=power_fraction, power_shape=power_shape, cold_frit_on=cold_on,
                 cold_frit_porosity=cold_eps, use_point_a=use_a, hot_frit_pore=hot_pore,
                 disturbance=dist, disturbed_slice=dist_slice, channel_on=channel_on)


try:
    res = run(power_pct / 100, power_shape, cold_on, cold_eps, use_a and cold_on,
              hot_pore_um * 1e-6, disturbance_pct / 100, disturbed - 1, channel_on)
except NotImplementedError as e:
    st.title("Can a frit stop a runaway?")
    st.error(f"This setting isn't supported yet. {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Header and status
# ---------------------------------------------------------------------------
st.title("Can a frit stop a runaway?")
st.markdown('<p class="lede">One particle bed fuel element from a published nuclear rocket design, split into '
            f'{p.N_SLICES} parallel flow paths. Change the settings on the left and the solver recomputes how '
            'hydrogen divides between the paths and how hot the fuel gets.</p>', unsafe_allow_html=True)

colors = {"stable": BLUE, "runaway": RED, "out_of_range": AMBER, "no_solution": AMBER, "unknown": MUTED}
words = {"stable": "Stable", "runaway": "Runaway", "out_of_range": "No steady solution",
         "no_solution": "No steady solution", "unknown": "Steady state found"}
st.markdown(f'<div class="status" style="--c:{colors[res["status"]]}"><span class="word">'
            f'{words[res["status"]]}</span><span class="why">{res["status_reason"]}</span></div>',
            unsafe_allow_html=True)


def fmt(v, pattern):
    return pattern.format(v) if np.isfinite(v) else "n/a"


def metric(col, value, label, over=False):
    col.markdown(f'<div class="metric"><div class="value{" over" if over else ""}">{value}</div>'
                 f'<div class="label">{label}</div></div>', unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)
metric(c1, fmt(res["peak_fuel_T"], "{:,.0f} K"), f"Peak fuel temperature (limit {p.T_LIMIT:,.0f} K)",
       over=res["peak_fuel_T"] > p.T_LIMIT)
metric(c2, fmt(res["dp_total"] / 1e3, "{:,.0f} kPa"), f"Pressure drop (limit {p.DP_LIMIT/1e3:,.0f} kPa)",
       over=res["dp_total"] > p.DP_LIMIT)
metric(c3, fmt(res["mixed_outlet_T"], "{:,.0f} K"), "Mixed outlet gas temperature")
metric(c4, f'{res["m_dot_total"]:.3f} kg/s', "Hydrogen flow")

for flag in res["flags"]:
    st.warning(flag)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_map, tab_val, tab_model, tab_q = st.tabs(
    ["Fuel element", "Stability map", "Validation", "What's in the model", "Open questions"])

with tab_live:
    # Bed temperature map (the paper's Fig. 8 view): axial along x, radial down the page
    if res["fuel_T"] is None:
        st.info("The bed temperature map needs radial profiles from slice.py (docs/model.md section 1a). "
                "The per-slice results are below.")
    else:
        fig, ax = plt.subplots(figsize=(11, 2.8))
        extent = [0, p.L_ELEMENT, p.R_BED_IN * 1e3, p.R_BED_OUT * 1e3]
        im = ax.imshow(res["fuel_T"].T, aspect="auto", origin="upper", extent=extent,
                       cmap=BED_CMAP, vmin=T_SCALE[0], vmax=T_SCALE[1], interpolation="bilinear")
        if res["fuel_T"].max() > p.T_LIMIT:
            X, R = np.meshgrid(res["x"], res["r"] * 1e3, indexing="ij")
            ax.contour(X, R, res["fuel_T"], levels=[p.T_LIMIT], colors=[RED], linewidths=1.5)
        ax.set_xlabel("Position along the element, m")
        ax.set_ylabel("Radius, mm")
        ax.set_title("Fuel temperature in the bed. Hydrogen enters at the top (cold frit) and leaves at the "
                     "bottom (hot frit).", loc="left", fontsize=10, color=MUTED)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("K")
        st.pyplot(fig)
        plt.close(fig)


    left, right = st.columns(2)
    slice_no = np.arange(1, len(res["m_dot"]) + 1)

    with left:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        bar_colors = [AMBER if (disturbance_pct and i == disturbed) else BLUE for i in slice_no]
        ax.bar(slice_no, res["m_dot"] * 1e3, color=bar_colors, width=0.8)
        ax.axhline(res["m_dot_total"] / len(slice_no) * 1e3, color=MUTED, lw=1, ls="--")
        ax.text(len(slice_no) + 0.4, res["m_dot_total"] / len(slice_no) * 1e3, " equal split",
                va="center", color=MUTED, fontsize=9)
        ax.set_xlabel("Slice (inlet end to outlet end)")
        ax.set_ylabel("Hydrogen flow, g/s")
        ax.set_title("Flow per slice", loc="left", fontsize=11, color=INK)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(slice_no, res["T_fuel_max_slice"], color=INK, lw=2, label="Peak fuel")
        ax.plot(slice_no, res["gas_outlet_T"], color=BLUE, lw=2, label="Gas leaving slice")
        for T in p.APP_WARNING_TEMPS_K[:-1]:
            ax.axhline(T, color=AMBER, lw=0.8, ls=":")
        ax.axhline(p.T_LIMIT, color=RED, lw=1.2)
        ax.text(1, p.T_LIMIT + 40, f"Fuel limit {p.T_LIMIT:,.0f} K", color=RED, fontsize=9)
        ax.set_xlabel("Slice (inlet end to outlet end)")
        ax.set_ylabel("Temperature, K")
        ax.set_title("Temperatures per slice", loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Dotted lines: 2800, 3100, and 3500 K warning levels (SNTP program). "
               "Solid red: 3800 K fuel melting limit (Ji et al.).")

with tab_map:
    st.subheader("Where the element becomes unstable")
    st.write("This map shows power against frit resistance, with each run marked stable or runaway. "
             "It appears once the stability sweeps have been run with the real solver.")

with tab_val:
    st.subheader("Checks against published results")
    st.write("Every check and its pass rule were written down and committed before the first run. "
             "Failures are reported, not tuned away.")
    vrun = latest_validation()
    if vrun is None:
        st.info("No validation run found in results/. Run `python3 runner.py` to produce one.")
    else:
        meta = vrun["meta"]
        by_id = {r["id"]: r for r in vrun["results"]}
        tier1 = [r for r in by_id.values() if r.get("tier") == 1]
        n_pass = sum(verdict_text(r).startswith("Pass") for r in tier1)
        n_fail = sum("fail" in verdict_text(r).lower() for r in tier1)
        st.caption(f"Official run at commit {meta.get('commit')}, params {meta.get('params_version')}, "
                   f"{meta.get('run_at', '')}"
                   + (". Channel model OFF (fallback rule)." if meta.get("channel_off") else ".")
                   + f" Tier 1: {n_pass} passed, {n_fail} failed.")
    ordered = sorted(p.VALIDATION.items(), key=lambda kv: (kv[1]["tier"], int(kv[0][1:])))
    rows = []
    for k, v in ordered:
        r = by_id.get(k) if vrun else None
        rows.append({"Check": k, "Tier": v["tier"], "Published result": v["target"],
                     "Pass rule": v["pass_rule"],
                     "Result": verdict_text(r) if r else "Not run yet",
                     "Details": detail_text(r) if r else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if vrun and by_id.get("V7", {}).get("verdict") == "fail":
        st.write("**V7 failed.** The model's no-cold-frit instability onset falls outside the window we "
                 "set before running (a factor of 3 around Kerrebrock's ~4 MW/L). We report it as a fail "
                 "and changed nothing in response; see the write-up for our explanation.")

with tab_model:
    st.subheader("What's in the model")
    st.markdown(f"""
**Real physics, model predictions.** Every result here comes from solving equations, not from measurements.

- **Geometry and operating point:** the preliminary fuel element from Ji et al. (2019), Nuclear Engineering and Design 355. Inlet 200 K, outlet pressure 7.0 MPa, 20 MW/L.
- **Hydrogen properties:** CoolProp below 1000 K; NIST-JANAF heat capacity and NASA (Svehla) viscosity and conductivity above, valid to 5000 K.
- **Pressure drop:** Ergun equation for both frits; KTA correlation for the particle bed.
- **Heat transfer:** KTA correlation between gas and fuel particles.
- **Mass flow:** set so the gas leaves at 3000 K using NIST enthalpy, {p.nominal_mass_flow():.3f} kg/s at full power. It scales with power when throttled.
- **Stability:** a slice is statically unstable where its pressure drop falls as flow rises. A second, independent eigenvalue test checks the same boundary.

**Limitations.** Paths don't exchange flow or heat sideways, as real beds do. Frits are treated as uniform resistances. Hydrogen dissociation above about 3500 K is ignored. Our temperatures run hotter than the Tsinghua papers' because NIST data gives about 13–15% less enthalpy rise than their property model, so we compare trends, not absolute temperatures.
""")

with tab_q:
    st.subheader("Questions we're answering")
    st.markdown("""
- **Q1.** Does power density decide whether cold-frit or hot-frit control works?
- **Q2.** Where does the instability start with no cold frit, and how much cold frit cures it?
- **Q3.** How large is the flow shift along the element, and does a U flow pattern reduce it?
- **Q4.** Does Ji et al.'s optimized point A stay stable across the 20–100% throttle range? We made no prediction.
- **Q5.** Can a fast trained model stand in for the solver?

No results yet. Each answer appears here once the runs exist.
""")

st.caption(f'Solve time {res["solve_time_s"]*1e3:.0f} ms. Parameters {p.PARAMS_VERSION}. '
           f'Code version {git_hash()}.')