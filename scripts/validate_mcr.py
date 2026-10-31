#!/usr/bin/env python3
"""Phase 1 acceptance gate: reproduce the MCR report's own published numbers.

If any of these checks fail the dynamics or the constants are wrong, and
nothing downstream can be trusted. Run this before every training campaign.

Reference values, MCR report sec. 7.2 / Table 13:
    impulsive two-impulse delta-v  3049.7 m/s
    flight time                    41.1 min
    apoapsis altitude              360 km
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.dynamics import IDX_R, IDX_RDOT, IDX_THETA, PlanarVehicle, propagate, rk4_step
from phenei_gym.guidance import min_energy_arc
from phenei_gym.params import PARAMS

REPORT_DV = 3049.7      # m/s   MCR Table 13
REPORT_TOF_MIN = 41.1   # min   MCR sec. 7.2
REPORT_APOAPSIS = 360.0  # km   MCR sec. 7.2

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, got: str, want: str) -> None:
    results.append((ok, f"{'PASS' if ok else 'FAIL'}  {label:38s} got {got:>14s}  want {want}"))


# ---------------------------------------------------------------- analytic arc
arc = min_energy_arc()
veh = PlanarVehicle.from_params(thrust_max=6000.0)  # MCR assumed 6 kN

check(
    abs(arc.delta_v_impulsive - REPORT_DV) / REPORT_DV < 2e-4,
    "impulsive delta-v [m/s]",
    f"{arc.delta_v_impulsive:.1f}", f"{REPORT_DV} (<0.02% err)",
)
check(
    abs(arc.time_of_flight / 60.0 - REPORT_TOF_MIN) < 0.1,
    "ballistic flight time [min]",
    f"{arc.time_of_flight / 60.0:.2f}", f"{REPORT_TOF_MIN} (+-0.1)",
)
check(
    abs(arc.apoapsis_altitude / 1e3 - REPORT_APOAPSIS) < 1.0,
    "apoapsis altitude [km]",
    f"{arc.apoapsis_altitude / 1e3:.1f}", f"{REPORT_APOAPSIS} (+-1)",
)
check(
    abs(veh.surface_gravity - PARAMS.value("SIM-E01")) < 1e-4,
    "surface gravity vs SIM-E01 [m/s2]",
    f"{veh.surface_gravity:.4f}", f"{PARAMS.value('SIM-E01'):.4f}",
)

# --------------------------------------------- numerical propagation of the arc
# Launch at the south pole end of the arc and coast. If the EOM are right the
# vehicle must come back to the surface at theta = 0 after exactly the
# analytic time of flight. This validates eom() and rk4_step() end to end.
r0 = arc.radius
theta0 = -arc.range_angle
x0 = np.array([
    r0,
    theta0,
    arc.v_surface * math.sin(arc.fpa_surface),
    arc.v_surface * math.cos(arc.fpa_surface) / r0,
    900.0,  # arbitrary: no thrust, so mass does not matter here
])


def coast(_t: float, _x: np.ndarray) -> tuple[float, float]:
    return 0.0, 0.0


def hit_surface(t: float, x: np.ndarray) -> str | None:
    return "surface" if t > 60.0 and x[IDX_R] <= arc.radius else None


traj = propagate(x0, coast, veh, dt=0.05, t_max=4000.0, stop=hit_surface)
theta_f = traj.x[-1, IDX_THETA]
check(
    abs(math.degrees(theta_f)) < 0.05,
    "ballistic arc closes at theta=0 [deg]",
    f"{math.degrees(theta_f):.4f}", "0.0 (+-0.05)",
)
check(
    abs(traj.t[-1] - arc.time_of_flight) < 1.0,
    "numerical vs analytic TOF [s]",
    f"{traj.t[-1]:.1f}", f"{arc.time_of_flight:.1f} (+-1)",
)
apo_num = traj.x[:, IDX_R].max() - arc.radius
check(
    abs(apo_num - arc.apoapsis_altitude) < 100.0,
    "numerical apoapsis [km]",
    f"{apo_num / 1e3:.2f}", f"{arc.apoapsis_altitude / 1e3:.2f} (+-0.1)",
)

# ------------------------------------------------------- conserved quantities
r, th, rd, thd, _ = traj.x.T
energy = 0.5 * (rd**2 + (r * thd) ** 2) - veh.mu / r
angmom = r**2 * thd
check(
    float(np.ptp(energy) / abs(energy[0])) < 1e-9,
    "specific energy conserved (coast)",
    f"{np.ptp(energy) / abs(energy[0]):.2e}", "< 1e-9 relative drift",
)
check(
    float(np.ptp(angmom) / abs(angmom[0])) < 1e-9,
    "angular momentum conserved (coast)",
    f"{np.ptp(angmom) / abs(angmom[0]):.2e}", "< 1e-9 relative drift",
)

# ----------------------------------------------------- RK4 convergence order
# Halving dt must cut the error by ~16x for a 4th-order method. The reference
# must be ONE fixed high-accuracy solution, not one per step size.
T_CONV = 600.0
_ref = x0.copy()
for _ in range(int(T_CONV / 0.01)):
    _ref = rk4_step(_ref, 0.0, 0.0, 0.01, veh)


def coast_error(dt: float) -> float:
    traj_c = propagate(x0, coast, veh, dt=dt, t_max=T_CONV)
    return float(np.linalg.norm(traj_c.x[-1, :IDX_RDOT + 1] - _ref[:IDX_RDOT + 1]))


# Step sizes must (a) divide T_CONV exactly and (b) keep the error well
# above roundoff -- at dt=2 s the error is ~4e-9 m on a 1.7e6 m radius, i.e.
# machine epsilon, and the measured order becomes meaningless.
errs = {dt: coast_error(dt) for dt in (20.0, 10.0, 5.0)}
order = math.log2(errs[20.0] / errs[10.0])
order2 = math.log2(errs[10.0] / errs[5.0])
check(
    3.7 < order < 4.3 and 3.7 < order2 < 4.3,
    "RK4 convergence order (20->10, 10->5 s)",
    f"{order:.2f}/{order2:.2f}", "4.0 (3.7-4.3)",
)
results.append((True, f"      errors: " + "  ".join(f"dt={k}s:{v:.3e}" for k, v in errs.items())))

# ------------------------------------- approach state matches SIM-A05..A08
st = arc.state_at_altitude(PARAMS.value("SIM-A01") * 1e3)  # SIM-A05..A08 were derived here
for pid, key, scale, unit in (
    ("SIM-A05", "speed", 1.0, "m/s"),
    ("SIM-A07", "v_vertical", -1.0, "m/s"),
    ("SIM-A08", "v_horizontal", 1.0, "m/s"),
):
    want = PARAMS.value(pid)
    got = st[key] * scale
    check(abs(got - want) / want < 1e-3, f"{pid} {key} [{unit}]", f"{got:.1f}", f"{want:.1f}")
fpa_deg = -math.degrees(st["fpa"])
check(
    abs(fpa_deg - PARAMS.value("SIM-A06")) < 0.01,
    "SIM-A06 flight-path angle [deg]",
    f"{fpa_deg:.2f}", f"{PARAMS.value('SIM-A06'):.2f}",
)

# ------------------------------------------------------------------- report
print(f"\nPHENEI planar dynamics -- MCR reproduction gate\n{'=' * 74}")
print(f"analytic arc: a={arc.semi_major_axis / 1e3:.1f} km  e={arc.eccentricity:.5f}  "
      f"periapsis={arc.periapsis_altitude / 1e3:.0f} km (inside the Moon: sub-orbital)\n")
for _, line in results:
    print("  " + line)
n_fail = sum(1 for ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
print("\n" + PARAMS.audit())
sys.exit(1 if n_fail else 0)
