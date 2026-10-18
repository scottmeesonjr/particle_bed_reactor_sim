"""Grid convergence check (game plan: double slices and radial steps until peak fuel
temperature and total pressure drop change by under 1%) and recorded solve time.
Case: cosine power, 80%, channel on, published frits. Not a validation case."""
import subprocess
import model

P = model.P
GRIDS = [(10, 10), (20, 20), (40, 40)]

rows = []
for n_s, n_r in GRIDS:
    r = model.solve(0.8, "cosine", True, P.COLD_FRIT_EPS, False, P.HOT_FRIT_PORE,
                    0.0, 0, True, n_slices=n_s, n_radial=n_r)
    rows.append((n_s, n_r, r["status"], r["peak_fuel_T"], r["dp_total"], r["solve_time_s"]))

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
out = [f"Grid convergence, commit {commit}",
       "Case: cosine power 80%, channel on, published frits (not a validation case)",
       f"{'grid':>8} {'status':>9} {'peak fuel T (K)':>16} {'dp total (kPa)':>15} "
       f"{'change T':>9} {'change dp':>10} {'time (s)':>9}"]
for i, (n_s, n_r, st, T, dp, t) in enumerate(rows):
    if i == 0:
        dT = ddp = ""
    else:
        T0, dp0 = rows[i - 1][3], rows[i - 1][4]
        dT, ddp = f"{abs(T - T0) / T0 * 100:.2f}%", f"{abs(dp - dp0) / dp0 * 100:.2f}%"
    out.append(f"{n_s:>3}x{n_r:<4} {st:>9} {T:16.1f} {dp / 1e3:15.2f} {dT:>9} {ddp:>10} {t:9.2f}")

(T1, dp1), (T2, dp2) = rows[1][3:5], rows[2][3:5]
ok = abs(T2 - T1) / T1 < 0.01 and abs(dp2 - dp1) / dp1 < 0.01
out.append("")
out.append(f"20x20 vs 40x40 within 1%: {'YES -> use 20x20' if ok else 'NO -> 20x20 not converged'}")
out.append(f"Recorded solve time at 20x20: {rows[1][5]:.2f} s")

text = "\n".join(out)
print(text)
open("results/grid_convergence.txt", "w").write(text + "\n")
