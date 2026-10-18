"""
surrogate.py -- Q5: can a fast trained model stand in for the solver?
(Ji et al. name a surrogate for their slow CFD as future work.)

    python surrogate.py                 # official run: results/q5_<commit>.{json,md,png,joblib}
    python surrogate.py --smoke DIR     # pipeline check only: 40 + 10 cases, written to DIR, never results/

Procedure: docs/model.md Amendment A3 (Q5) and Amendment A4 (details), both committed
before the first run. Summary:
  - Ji et al. nominal element, channel on, Z pattern, published porosities, flow by rule D2.
  - Inputs from a Latin hypercube (seed 20261018): power fraction 0.20-1.00, cold frit pore
    1.5-4.0 um and hot frit pore 30-300 um (both log-uniform); uniform power for even case
    numbers, cosine for odd.
  - Labels: ok = converged and stable (single-point rule, A4); not ok = unstable or no
    solution. Peak fuel temperature for converged cases.
  - Budget: min(2000, floor(1800 s x workers / seconds per case)), seconds per case measured
    on the first 20 cases.
  - 80/20 train / held-out split; 200-case out-of-range set (cold frit pore 4.0-6.0 um).
  - GradientBoostingClassifier / GradientBoostingRegressor, default settings.
  - "Yes, it can stand in" if held-out balanced accuracy >= 0.95 and held-out
    peak-temperature MAE <= 50 K.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import params as P

RESULTS = Path(__file__).resolve().parent / "results"
SEED, OOR_SEED = 20261018, 20261019
N_MAX, N_TIMING, N_OOR = 2000, 20, 200
TIME_BUDGET_S = 1800.0
POWER_RANGE = (0.20, 1.00)
COLD_PORE_RANGE = (1.5e-6, 4.0e-6)
COLD_PORE_OOR = (4.0e-6, 6.0e-6)
HOT_PORE_RANGE = (30e-6, 300e-6)
BAL_ACC_MIN, MAE_MAX_K = 0.95, 50.0
FEATURES = ("power fraction", "log10 cold frit pore (m)", "log10 hot frit pore (m)", "cosine power")


# ---------------------------------------------------------------------------
# Sampling (A4)
# ---------------------------------------------------------------------------
def sample(n, seed, cold_range=COLD_PORE_RANGE):
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=3, seed=seed).random(n)

    def log_uniform(x, lo, hi):
        return float(10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo))))

    return [dict(k=k, power_frac=float(POWER_RANGE[0] + u[k, 0] * (POWER_RANGE[1] - POWER_RANGE[0])),
                 cold_pore=log_uniform(u[k, 1], *cold_range), hot_pore=log_uniform(u[k, 2], *HOT_PORE_RANGE),
                 power_shape="uniform" if k % 2 == 0 else "cosine") for k in range(n)]


def budget(seconds_per_case, workers):
    return int(min(N_MAX, math.floor(TIME_BUDGET_S * workers / seconds_per_case)))


# ---------------------------------------------------------------------------
# One solver case (runs in a worker)
# ---------------------------------------------------------------------------
def run_case(c):
    import steady
    import questions
    t = time.perf_counter()
    try:
        cold, eps = c["cold_pore"], P.COLD_FRIT_EPS
        res = steady.solve_element(c["power_frac"], P.nominal_mass_flow(c["power_frac"]),
                                   power_shape=c["power_shape"], cold_profile=lambda s: (cold, eps),
                                   hot=(c["hot_pore"], P.HOT_FRIT_EPS), channel=True, pattern="Z",
                                   pressure_ref="outlet")
        status, st = questions.single_point_status(res)
        out = dict(c, status=status, path=res.get("path"))
        if res["converged"]:
            out.update(peak_fuel_T=float(np.nanmax(res["T_fuel_max"])), max_real=st["max_real"],
                       ledinegg=st["ledinegg"])
    except Exception as e:                                 # A4: recorded, excluded, counted
        out = dict(c, status="error", error=f"{type(e).__name__}: {e}")
    out["seconds"] = time.perf_counter() - t
    return out


def generate(workers, n_override=None, n_oor=N_OOR, log=print):
    """Timing cases, then the budgeted remainder, then the out-of-range set."""
    plan = sample(N_MAX, SEED)
    with Pool(workers) as pool:
        first = pool.map(run_case, plan[:N_TIMING], chunksize=1)
        sec = float(np.mean([r["seconds"] for r in first]))
        n = budget(sec, workers) if n_override is None else n_override
        log(f"{sec:.2f} s per case on {workers} workers -> budget {n} cases")
        rest = []
        for i, r in enumerate(pool.imap(run_case, plan[N_TIMING:max(n, N_TIMING)], chunksize=2), 1):
            rest.append(r)
            if i % 100 == 0:
                log(f"  {N_TIMING + i} / {n}")
        oor = pool.map(run_case, sample(n_oor, OOR_SEED, COLD_PORE_OOR), chunksize=2)
    return dict(cases=(first + rest)[:n], oor=oor, seconds_per_case=sec, workers=workers, budget=n)


# ---------------------------------------------------------------------------
# Models (A3, A4)
# ---------------------------------------------------------------------------
def features(cases):
    return np.array([[c["power_frac"], math.log10(c["cold_pore"]), math.log10(c["hot_pore"]),
                      1.0 if c["power_shape"] == "cosine" else 0.0] for c in cases])


def is_ok(c):
    return c["status"] == "stable"


def converged(c):
    return c["status"] in ("stable", "unstable")


def answer(balanced_accuracy, mae_K, n_classes_held_out):
    if n_classes_held_out < 2 or balanced_accuracy is None or mae_K is None:
        return "not answerable"
    return "yes" if (balanced_accuracy >= BAL_ACC_MIN and mae_K <= MAE_MAX_K) else "no"


def _class_scores(clf, cases):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    if not cases:
        return dict(n=0)
    y = np.array([is_ok(c) for c in cases], dtype=int)
    yp = clf.predict(features(cases))
    out = dict(n=len(cases), n_ok=int(y.sum()), accuracy=float(accuracy_score(y, yp)),
               confusion=dict(zip(("true_not_ok_pred_not_ok", "true_not_ok_pred_ok", "true_ok_pred_not_ok",
                                   "true_ok_pred_ok"), map(int, confusion_matrix(y, yp, labels=[0, 1]).ravel()))),
               predicted_ok=[int(v) for v in yp])
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, yp)) if len(set(y)) == 2 else None
    return out


def _reg_scores(reg, cases):
    conv = [c for c in cases if converged(c)]
    if not conv:
        return dict(n=0, mae_K=None)
    y = np.array([c["peak_fuel_T"] for c in conv])
    err = reg.predict(features(conv)) - y
    return dict(n=len(conv), mae_K=float(np.mean(np.abs(err))), max_abs_error_K=float(np.max(np.abs(err))))


def train_and_test(data):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    usable = [c for c in data["cases"] if c["status"] != "error"]
    train, held = train_test_split(usable, test_size=0.2, random_state=SEED)
    oor = [c for c in data["oor"] if c["status"] != "error"]

    clf = GradientBoostingClassifier(random_state=SEED).fit(
        features(train), np.array([is_ok(c) for c in train], dtype=int))
    conv_train = [c for c in train if converged(c)]
    reg = GradientBoostingRegressor(random_state=SEED).fit(
        features(conv_train), np.array([c["peak_fuel_T"] for c in conv_train]))

    t = time.perf_counter()
    Xh = features(held)
    clf.predict(Xh); reg.predict(Xh)
    pred_s = (time.perf_counter() - t) / max(len(held), 1)

    hc, hr = _class_scores(clf, held), _reg_scores(reg, held)
    n_classes = len({is_ok(c) for c in held})
    scores = dict(
        n_train=len(train), n_train_converged=len(conv_train), n_held_out=len(held),
        held_out=dict(classifier=hc, regressor=hr),
        out_of_range=dict(classifier=_class_scores(clf, oor), regressor=_reg_scores(reg, oor)),
        prediction_seconds_per_case=pred_s, solver_seconds_per_case=data["seconds_per_case"],
        speedup=data["seconds_per_case"] / pred_s if pred_s > 0 else None,
        answer=answer(hc.get("balanced_accuracy"), hr.get("mae_K"), n_classes))
    return clf, reg, held, scores


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _status_counts(cases):
    out = {}
    for c in cases:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out


def write_png(path, held, clf, reg, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, RED, AMBER = "#18212B", "#5B6672", "#2F6FB0", "#B3261E", "#C9821B"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    pred = clf.predict(features(held)) if held else []
    for c, p_ok in zip(held, pred):
        truth = is_ok(c)
        color = BLUE if truth else (AMBER if c["status"] in ("no_solution", "out_of_range") else RED)
        ax.scatter(c["cold_pore"] * 1e6, c["power_frac"] * 100, s=18, color=color,
                   marker="o" if bool(p_ok) == truth else "x", lw=1.2)
    ax.set_xscale("log")
    ticks = [1.5, 2, 2.5, 3, 4]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("Cold frit pore, µm (log scale)")
    ax.set_ylabel("Power, % of nominal")
    ax.set_title("Held-out cases: solver's verdict (color), surrogate wrong (x)", loc="left", fontsize=10.5, color=INK)
    for color, label in ((BLUE, "Stable"), (RED, "Unstable"), (AMBER, "No steady solution")):
        ax.scatter([], [], color=color, s=18, label=label)
    ax.scatter([], [], color=MUTED, marker="x", s=18, label="Surrogate got it wrong")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    conv = [c for c in held if converged(c)]
    if conv:
        y = np.array([c["peak_fuel_T"] for c in conv]); yp = reg.predict(features(conv))
        lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":")
        ax.scatter(y, yp, s=14, color=INK)
        ax.axvline(P.T_LIMIT, color=RED, lw=0.8); ax.axhline(P.T_LIMIT, color=RED, lw=0.8)
    mae = scores["held_out"]["regressor"].get("mae_K")
    ax.set_xlabel("Solver peak fuel temperature, K")
    ax.set_ylabel("Surrogate prediction, K")
    ax.set_title("Held-out peak fuel temperature" + (f" (MAE {mae:.0f} K)" if mae is not None else ""),
                 loc="left", fontsize=10.5, color=INK)
    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_md(path, out):
    s = out["scores"]
    hc, hr = s["held_out"]["classifier"], s["held_out"]["regressor"]
    oc, orr = s["out_of_range"]["classifier"], s["out_of_range"]["regressor"]
    f = lambda v, fmt: format(v, fmt) if isinstance(v, (int, float)) else "n/a"
    word = {"yes": "Yes, it can stand in", "no": "No, not by our rule", "not answerable": "Not answerable"}[s["answer"]]
    lines = [
        "# Q5: Can a fast trained model stand in for the solver?", "",
        f"Commit {out['commit']}{' (uncommitted changes present)' if out.get('uncommitted_changes') else ''}. "
        "Procedure: docs/model.md A3 and A4. Ji et al. element, channel on, Z pattern.", "",
        "## Answer",
        f"- **{word}.** Rule: held-out balanced accuracy >= {BAL_ACC_MIN} and peak-temperature MAE <= "
        f"{MAE_MAX_K:.0f} K. Measured: balanced accuracy {f(hc.get('balanced_accuracy'), '.3f')}, "
        f"MAE {f(hr.get('mae_K'), '.1f')} K.",
        f"- **Speed.** {f(s['prediction_seconds_per_case'] * 1e6, '.0f')} µs per prediction vs "
        f"{s['solver_seconds_per_case']:.2f} s per solver case "
        f"(about {f(s['speedup'], ',.0f')} times faster).",
        f"- **Outside the training range** (cold frit pore 4-6 µm, does not change the answer): accuracy "
        f"{f(oc.get('accuracy'), '.3f')}, balanced accuracy {f(oc.get('balanced_accuracy'), '.3f')}, "
        f"MAE {f(orr.get('mae_K'), '.1f')} K on {orr.get('n', 0)} converged cases.", "",
        "## Data",
        f"- Budget: {out['budget']} cases ({out['seconds_per_case']:.2f} s per case on {out['workers']} worker{'s' if out['workers'] != 1 else ''}). "
        f"Status counts: {json.dumps(out['status_counts'])}.",
        f"- Out-of-range set: {len(out['oor'])} cases, status counts {json.dumps(out['oor_status_counts'])}.",
        f"- Train {s['n_train']} ({s['n_train_converged']} converged, used for the temperature model); "
        f"held out {s['n_held_out']}.", "",
        "## Held-out details",
        f"- Classifier: accuracy {f(hc.get('accuracy'), '.3f')}; confusion {json.dumps(hc.get('confusion'))}.",
        f"- Temperature model: {hr.get('n', 0)} converged cases, max error {f(hr.get('max_abs_error_K'), '.0f')} K.", "",
        "## Limits",
        "- The surrogate only knows this element, channel on, Z pattern, and the sampled ranges.",
        "- It learns the solver, not the reactor: anything the solver gets wrong, it inherits.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(out_dir=RESULTS, workers=None, n_override=None, n_oor=N_OOR, tag=None):
    import joblib
    import questions
    h, dirty = questions.commit()
    tag = tag or h
    workers = workers or os.cpu_count() or 1
    t0 = time.time()
    data = generate(workers, n_override, n_oor)
    clf, reg, held, scores = train_and_test(data)
    out = dict(question="Q5", commit=h, uncommitted_changes=dirty, procedure="docs/model.md A3, A4",
               seed=SEED, oor_seed=OOR_SEED, workers=data["workers"], budget=data["budget"],
               seconds_per_case=data["seconds_per_case"], status_counts=_status_counts(data["cases"]),
               oor_status_counts=_status_counts(data["oor"]), features=list(FEATURES), scores=scores,
               held_out_ks=[c["k"] for c in held], cases=data["cases"], oor=data["oor"])
    out["seconds"] = time.time() - t0
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"q5_{tag}"
    base.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    joblib.dump(dict(classifier=clf, regressor=reg, features=list(FEATURES), commit=h,
                     ranges=dict(power=POWER_RANGE, cold_pore=COLD_PORE_RANGE, hot_pore=HOT_PORE_RANGE),
                     seconds_per_case=data["seconds_per_case"]), base.with_suffix(".joblib"))
    write_md(base.with_suffix(".md"), out)
    write_png(base.with_suffix(".png"), held, clf, reg, scores)
    return base, out


# ---------------------------------------------------------------------------
# App interface: load the newest trained models and predict (never retrains)
# ---------------------------------------------------------------------------
def load_latest(results=RESULTS):
    import joblib
    files = sorted(Path(results).glob("q5_*.joblib"), key=os.path.getmtime)
    return joblib.load(files[-1]) if files else None


def predict(models, power_frac, cold_pore, hot_pore, power_shape):
    c = dict(power_frac=power_frac, cold_pore=cold_pore, hot_pore=hot_pore, power_shape=power_shape)
    t = time.perf_counter()
    X = features([c])
    p_ok = float(models["classifier"].predict_proba(X)[0, 1])
    peak = float(models["regressor"].predict(X)[0])
    return dict(ok=p_ok >= 0.5, p_ok=p_ok, peak_fuel_T=peak, seconds=time.perf_counter() - t)


def in_training_range(models, power_frac, cold_pore, hot_pore):
    r = models["ranges"]
    return (r["power"][0] <= power_frac <= r["power"][1] and r["cold_pore"][0] <= cold_pore <= r["cold_pore"][1]
            and r["hot_pore"][0] <= hot_pore <= r["hot_pore"][1])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--smoke":
        base, out = run(out_dir=sys.argv[2], n_override=40, n_oor=10, tag="smoke")
        print(f"smoke run wrote {base}.* (pipeline check only; not a result)")
    elif len(sys.argv) == 1:
        base, out = run()
        print(f"wrote {base.relative_to(RESULTS.parent)}.json/.md/.png/.joblib ({out['seconds']:.0f} s)")
    else:
        sys.exit("usage: python surrogate.py  |  python surrogate.py --smoke DIR")
