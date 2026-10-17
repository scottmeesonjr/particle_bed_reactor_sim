"""smoke_slice.py -- one nominal slice through slice_dp (no model choices here)."""
import params as P
import properties as pr
from slice import slice_dp

N = 20
dz = P.L_ELEMENT / N
m_dot = P.nominal_mass_flow() / N
q_slice = P.total_power() / N
# CLAUDE.md names these P_OUT and T_IN; params.py defines P_OUTLET and T_INLET.
p_in, T_in = P.P_OUTLET, P.T_INLET

res = slice_dp(m_dot, q_slice, p_in, T_in,
               cold=(P.COLD_FRIT_PORE, P.COLD_FRIT_EPS),
               hot=(P.HOT_FRIT_PORE, P.HOT_FRIT_EPS),
               dz=dz, n_r=20)

print(f"inputs: m_dot {m_dot:.6f} kg/s, q_slice {q_slice:.1f} W, dz {dz:.4f} m, "
      f"p_in {p_in/1e6:.2f} MPa, T_in {T_in:.1f} K, n_r 20")
print(f"dp           {res['dp']/1e3:10.3f} kPa")
print(f"  cold frit  {res['dp_cold_frit']/1e3:10.3f} kPa")
print(f"  bed        {res['dp_bed']/1e3:10.3f} kPa")
print(f"  hot frit   {res['dp_hot_frit']/1e3:10.3f} kPa")
print(f"T_gas_out    {res['T_gas_out']:10.2f} K")
print(f"T_fuel_max   {res['T_fuel_max']:10.2f} K")
print(f"p_out        {res['p_out']/1e6:10.6f} MPa")
for k in ("re_m_low", "re_ht_low", "out_of_range", "pressure_collapse", "fuel_over_limit"):
    print(f"{k:<18} {res[k]}")
q_per_kg = q_slice / m_dot
err = abs(float(pr.h(res["T_gas_out"]) - pr.h(T_in)) - q_per_kg) / q_per_kg
print(f"energy check {err:.3e}")
