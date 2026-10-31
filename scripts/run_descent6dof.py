#!/usr/bin/env python3
"""Fly the full 6-DOF descent and write a Rerun recording with terrain and plumes.

    python scripts/run_descent6dof.py
    python scripts/run_descent6dof.py --altitude 70 --mass 1000 --seed 7
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.envs import Descent6DOF, Descent6DOFConfig
from phenei_gym.params import PARAMS
from phenei_gym.terrain import generate
from phenei_gym.viz.rerun6dof import log_descent6dof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thrust", type=float, default=6000.0)
    ap.add_argument("--altitude", type=float, default=75.0, help="ignition altitude [km]")
    ap.add_argument("--mass", type=float, default=950.0)
    ap.add_argument("--cap", type=float, default=0.90, help="main throttle cap")
    ap.add_argument("--omega-n", type=float, default=1.5)
    ap.add_argument("--k-vz", type=float, default=0.2)
    ap.add_argument("--dt", type=float, default=1.0 / 200.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--align", type=int, default=4, help="downrange alignment passes")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--out", default="studies/descent6dof.rrd")
    a = ap.parse_args()

    def fly(x0: float):
        cfg = Descent6DOFConfig(thrust_max=a.thrust, thrust_cap=a.cap, k_vz=a.k_vz,
                                omega_n=a.omega_n, mass=a.mass, initial_downrange=x0,
                                ignition_altitude=a.altitude * 1e3, dt=a.dt, t_max=900.0)
        return Descent6DOF(cfg).run()

    # Two passes: fly once from the origin to measure the downrange covered,
    # then restart that far upstream so touchdown lands ON the landing site.
    # Unlike the 3-DOF polar model this is not exactly translation-invariant
    # (gravity points at the Moon's centre), so it is iterated rather than
    # shifted analytically.
    res = fly(0.0)
    for _ in range(a.align):
        res = fly(res.state[0, 0] - float(res.state[-1, 0]))
    td = res.touchdown()
    tilt = np.degrees(res.tilt)

    print(f"\nPHENEI 6-DOF descent -- ConOps phases 7-8\n{'=' * 64}")
    print(f"  ignition        {a.altitude:6.1f} km   mass {a.mass:6.1f} kg   "
          f"thrust {a.thrust / 1e3:.1f} kN capped at {a.cap:.0%}")
    print(f"  outcome         {res.reason} after {res.t[-1]:.1f} s")
    print(f"  propellant      {res.propellant:6.1f} kg")
    print(f"  peak tilt       {tilt.max():6.1f} deg   (SIM-A16: a body-fixed nadir sensor is "
          f"mispointed for most of the burn)")
    print("\n  touchdown vs requirements")
    for label, got, pid in (("descent rate [m/s]", abs(td["v_vertical"]), "SIM-T02"),
                            ("horizontal   [m/s]", td["v_horizontal"], "SIM-T03"),
                            ("tilt         [deg]", td["tilt_deg"], "SIM-T04"),
                            ("body rate  [deg/s]", td["rate_deg_s"], "SIM-T05"),
                            ("miss         [m]  ", td["miss_m"], "SIM-T01")):
        lim = PARAMS.value(pid)
        print(f"    {'PASS' if got <= lim else 'FAIL'}  {label}  {got:9.3f}  <= {lim:6.1f}  ({pid})")

    terrain = generate(seed=a.seed, extent=300.0, n=193)
    print(f"\n  terrain  slope {terrain.slope_deg:.2f} deg, roughness {terrain.roughness:.3f} m, "
          f"albedo {terrain.albedo:.3f}, {len(terrain.hazards)} hazards")
    out = log_descent6dof(res, terrain, a.out, serve=a.serve)
    print(f"  recording       {out.resolve()}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  view it         ./scripts/view.sh {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
