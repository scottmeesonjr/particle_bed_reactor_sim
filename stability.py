"""
stability.py -- stability tests at a converged steady state, docs/model.md
section 5 and Amendment A1 (5a). Channel off only (all slices share one
pressure drop and the total flow is fixed).

static_check(result) -> {"ledinegg": bool, "det_sign": +1 or -1}
    ledinegg: any slice with d(dp_slice_i)/d(m_i) < 0 at fixed slice power
    det_sign: sign of the determinant of the residual Jacobian (steady.py)

eigen_check(result) -> {"max_real", "oscillatory", "eigenvalues", "fuel_cp_invalid"}
    Transient model, one state per slice (outlet gas temperature T_i):
        C_i dT_i/dt = Q_i - m_i (h(T_i) - h_in)
        C_i = rho_fuel * fuel_cp(T_mean,i) * (1 - alpha) * V_bed_slice * 0.5,
        T_mean,i = (T_in + T_i) / 2   (mean of the linear profile; A1 5a)
    Flows adjust instantly so every slice keeps the same pressure drop and the
    flows still sum to the total. Each slice's pressure drop is a function of
    its flow m_i and outlet temperature T_i, evaluated with slice_dp using the
    heat that gives that outlet temperature, q = m_i (h(T_i) - h_in).

    Linearizing: a_i = d(dp_i)/d(m_i) at fixed T_i, b_i = d(dp_i)/d(T_i) at
    fixed m_i. Equal pressure drop with fixed total flow gives
        dm_i/dT_j = ((b_j / a_j) / S - b_i * delta_ij) / a_i,   S = sum_k 1/a_k
    so the Jacobian of dT/dt is
        J_ij = -(dh_i * dm_i/dT_j + m_i * cp_i * delta_ij) / C_i,
    with dh_i = h(T_i) - h_in. Any eigenvalue with positive real part = unstable.
"""
import numpy as np

import params as P
import properties as pr
import slice as sl

FD_REL_M = 1.0e-4       # relative step in flow for a_i
FD_REL_T = 1.0e-4       # relative step in temperature for b_i
ZRC_MELT_K = 3805.0     # JANAF C-110 crystal-liquid transition


def static_check(result):
    if not result.get("converged") or result.get("jacobian") is None:
        raise ValueError("static_check needs a converged steady.solve_element result")
    sign, _ = np.linalg.slogdet(result["jacobian"])
    return {"ledinegg": bool(np.any(result["ddp_dm"] < 0.0)),
            "det_sign": int(np.sign(sign)) if sign != 0 else 0}


def _dp(si, i, m, T_out):
    q_eff = m * float(pr.h(T_out) - pr.h(si["T_in"]))
    r = sl.slice_dp(m, q_eff, si["p_in"], si["T_in"], si["colds"][i], si["hot"],
                    si["dz"], si["n_r"], geom=si["geom"])
    if r["out_of_range"] or r["pressure_collapse"]:
        raise RuntimeError(f"eigen_check: slice {i} left the valid range during differencing")
    return r["dp"]


def eigen_check(result):
    if not result.get("converged"):
        raise ValueError("eigen_check needs a converged steady.solve_element result")
    si = result["slice_inputs"]
    m, T = np.asarray(result["m"]), np.asarray(result["T_gas_out"])
    N = len(m)

    r_ch, r_hf, r_bed, r_cf, d_p, alpha = sl._geometry(si["geom"])
    v_bed = np.pi * (r_bed**2 - r_hf**2) * si["dz"]
    T_mean = 0.5 * (si["T_in"] + T)
    C = P.FUEL_DENSITY * pr.fuel_cp(T_mean) * (1.0 - alpha) * v_bed * 0.5

    a, b = np.empty(N), np.empty(N)
    for i in range(N):
        hm, hT = FD_REL_M * m[i], FD_REL_T * T[i]
        a[i] = (_dp(si, i, m[i] + hm, T[i]) - _dp(si, i, m[i] - hm, T[i])) / (2.0 * hm)
        b[i] = (_dp(si, i, m[i], T[i] + hT) - _dp(si, i, m[i], T[i] - hT)) / (2.0 * hT)
    if np.any(a == 0.0):
        raise RuntimeError("eigen_check: a slice has zero d(dp)/d(m) at fixed temperature")

    S = np.sum(1.0 / a)
    dmdT = ((b / a)[None, :] / S - np.diag(b)) / a[:, None]
    dh = pr.h(T) - pr.h(si["T_in"])
    J = -(dh[:, None] * dmdT + np.diag(m * pr.cp(T))) / C[:, None]

    eig = np.linalg.eigvals(J)
    k = int(np.argmax(eig.real))
    lead = eig[k]
    return {"max_real": float(lead.real),
            "oscillatory": bool(abs(lead.imag) > 1e-9 * max(abs(lead.real), 1.0)),
            "eigenvalues": eig,
            "fuel_cp_invalid": bool(np.nanmax(result["T_fuel_max"]) > ZRC_MELT_K)}
