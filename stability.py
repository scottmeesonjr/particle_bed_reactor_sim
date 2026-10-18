"""
stability.py -- stability tests at a converged steady state, docs/model.md
section 5 and Amendments A1 (5a) and A2 (channel on).

static_check(result) -> {"ledinegg": bool, "det_sign": +1 or -1}
    ledinegg: any slice with d(dp_slice_i)/d(m_i) < 0 at fixed slice power
    det_sign: sign of the determinant of the residual Jacobian (steady.py)

eigen_check(result) -> {"max_real", "oscillatory", "eigenvalues", "fuel_cp_invalid"}
    Transient model, one state per slice (outlet gas temperature T_i):
        C_i dT_i/dt = Q_i - m_i (h(T_i) - h_in)
        C_i = rho_fuel * fuel_cp(T_mean,i) * (1 - alpha) * V_bed_slice * 0.5,
        T_mean,i = (T_in + T_i) / 2   (mean of the linear profile; A1 5a)
    Flows adjust instantly so every slice's pressure drop matches the plenum
    minus channel pressure at its height (with the channel off: one shared drop),
    and the flows still sum to the total. Each slice's pressure drop is a function
    of its flow m_i, outlet temperature T_i and inlet pressure, evaluated with
    slice_dp using the heat that gives that outlet temperature,
    q = m_i (h(T_i) - h_in). The channel pressure also depends on T_i (mixing).

    Linearizing the full system R(x, T) = 0 gives dx/dT = -(dR/dx)^-1 dR/dT, so
        J_ij = -(dh_i * dm_i/dT_j + m_i * cp_i * delta_ij) / C_i,
    with dh_i = h(T_i) - h_in. Any eigenvalue with positive real part = unstable.
    With the channel off this equals the closed form
        dm_i/dT_j = ((b_j / a_j) / S - b_i * delta_ij) / a_i,  S = sum_k 1/a_k.
"""
import numpy as np

import params as P
import properties as pr
import slice as sl
import steady

FD_REL_M = 1.0e-4       # relative step in flow for a_i
FD_REL_T = 1.0e-4       # relative step in temperature for b_i
ZRC_MELT_K = 3805.0     # JANAF C-110 crystal-liquid transition


def static_check(result):
    if not result.get("converged") or result.get("jacobian") is None:
        raise ValueError("static_check needs a converged steady.solve_element result")
    sign, _ = np.linalg.slogdet(result["jacobian"])
    return {"ledinegg": bool(np.any(result["ddp_dm"] < 0.0)),
            "det_sign": int(np.sign(sign)) if sign != 0 else 0}


def _dp(si, i, m, T_out, p_in=None):
    """Slice pressure drop at flow m and outlet temperature T_out (heat set to match)."""
    p = si["p_slices"][i] if p_in is None else p_in
    q_eff = m * float(pr.h(T_out) - pr.h(si["T_in"]))
    r = sl.slice_dp(m, q_eff, p, si["T_in"], si["colds"][i], si["hot"],
                    si["dz"], si["n_r"], geom=si["geom"])
    if r["out_of_range"] or r["pressure_collapse"]:
        raise RuntimeError(f"eigen_check: slice {i} left the valid range during differencing")
    return r["dp"]


def eigen_check(result):
    if not result.get("converged"):
        raise ValueError("eigen_check needs a converged steady.solve_element result")
    si = result["slice_inputs"]
    st, m_avg, m_total = si["setup"], si["m_avg"], si["m_total"]
    m, T = np.asarray(result["m"]), np.asarray(result["T_gas_out"])
    N = len(m)

    r_ch, r_hf, r_bed, r_cf, d_p, alpha = sl._geometry(si["geom"])
    v_bed = np.pi * (r_bed**2 - r_hf**2) * si["dz"]
    T_mean = 0.5 * (si["T_in"] + T)
    C = P.FUEL_DENSITY * pr.fuel_cp(T_mean) * (1.0 - alpha) * v_bed * 0.5

    # Slice sensitivities at fixed outlet temperature: a = d(dp)/dm, b = d(dp)/dT, c = d(dp)/dp_in
    a, b, c = np.empty(N), np.empty(N), np.empty(N)
    for i in range(N):
        p_i = si["p_slices"][i]
        hm, hT, hp = FD_REL_M * m[i], FD_REL_T * T[i], FD_REL_M * p_i
        a[i] = (_dp(si, i, m[i] + hm, T[i]) - _dp(si, i, m[i] - hm, T[i])) / (2.0 * hm)
        b[i] = (_dp(si, i, m[i], T[i] + hT) - _dp(si, i, m[i], T[i] - hT)) / (2.0 * hT)
        c[i] = (_dp(si, i, m[i], T[i], p_i + hp) - _dp(si, i, m[i], T[i], p_i - hp)) / (2.0 * hp)

    # Manifold sensitivities (zero except the end pressures when the channel is off)
    _, _, dPpl, dPch_x, dPch_T = steady.manifold_derivatives(m, T, si["x_N"], st, m_avg)

    # Linearized residuals R(x, T) = 0 (unscaled): R_i = p_pl,i - p_ch,i - dp_i, R_N = mass
    Rx = np.zeros((N + 1, N + 1))
    Rx[:N] = dPpl - dPch_x - c[:, None] * dPpl
    Rx[np.arange(N), np.arange(N)] -= a * m_avg
    Rx[N, :N] = m_avg / m_total
    RT = np.zeros((N + 1, N))
    RT[:N] = -dPch_T - np.diag(b)
    dmdT = m_avg * (-np.linalg.solve(Rx, RT))[:N]

    dh = pr.h(T) - pr.h(si["T_in"])
    J = -(dh[:, None] * dmdT + np.diag(m * pr.cp(T))) / C[:, None]

    eig = np.linalg.eigvals(J)
    k = int(np.argmax(eig.real))
    lead = eig[k]
    return {"max_real": float(lead.real),
            "oscillatory": bool(abs(lead.imag) > 1e-9 * max(abs(lead.real), 1.0)),
            "eigenvalues": eig,
            "fuel_cp_invalid": bool(np.nanmax(result["T_fuel_max"]) > ZRC_MELT_K)}
