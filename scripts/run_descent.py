#!/usr/bin/env python3
"""Fly the reference braking-burn + terminal-descent trajectory and visualise it.

    .venv/bin/python scripts/run_descent.py                  # write a .rrd
    .venv/bin/python scripts/run_descent.py --serve           # live web viewer
    .venv/bin/python scripts/run_descent.py --thrust 4000     # sweep thrust
    .venv/bin/python scripts/run_descent.py --altitude 39     # the MCR value (fails)

Episode = ConOps phases 7 and 8 only: braking-burn ignition to touchdown.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.dynamics import IDX_M, PlanarVehicle, propagate, to_local
from phenei_gym.guidance import min_energy_arc
from phenei_gym.guidance.descent import DescentGuidance, touchdown_stop
from phenei_gym.params import PARAMS
from phenei_gym.viz import log_trajectory


def ignition_state(arc, altitude_m: float, mass: float) -> np.ndarray:
    """Initial condition on the descending leg of the hop arc."""
    st = arc.state_at_altitude(altitude_m)
    r0 = st["radius"]
    return np.array([r0, 0.0, st["v_vertical"], st["v_horizontal"] / r0, mass])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thrust", type=float, default=6000.0, help="total max thrust [N]")
    ap.add_argument("--altitude", type=float, default=None, help="ignition altitude [km]")
    ap.add_argument("--mass", type=float, default=None, help="mass at ignition [kg]")
    ap.add_argument("--dt", type=float, default=0.02, help="integrator step [s]")
    ap.add_argument("--serve", action="store_true", help="serve a live web viewer")
    ap.add_argument("--out", default="studies/reference_descent.rrd")
    args = ap.parse_args()

    veh = PlanarVehicle.from_params(thrust_max=args.thrust)
    arc = min_energy_arc()
    alt = (args.altitude if args.altitude is not None else PARAMS.value("SIM-A19")) * 1e3
    mass = args.mass if args.mass is not None else PARAMS.value("SIM-M03")

    guidance = DescentGuidance.from_params(veh, n_engines_terminal=2)
    # The vertical profile is tied to the ignition condition, so recompute it
    # whenever the ignition altitude is not the SIM-A19 baseline.
    st = arc.state_at_altitude(alt)
    guidance.a_dec = st["v_vertical"] ** 2 / (2.0 * alt)

    x0 = ignition_state(arc, alt, mass)
    traj = propagate(x0, guidance, veh, dt=args.dt, t_max=900.0, stop=touchdown_stop(veh))
    # The EOM do not depend on theta, so the trajectory is translation-invariant
    # in it. Shift so that theta = 0 is the touchdown point, which is what the
    # landing-accuracy requirement is measured against.
    from phenei_gym.dynamics import IDX_THETA
    traj.x[:, IDX_THETA] -= traj.x[-1, IDX_THETA]
    final = to_local(traj.x[-1], veh)
    loc = traj.local(veh)

    lim = {k: PARAMS.value(k) for k in ("SIM-T01", "SIM-T02", "SIM-T03")}
    propellant = mass - traj.x[-1, IDX_M]
    tilt = np.abs(np.degrees(traj.beta - math.pi / 2.0))

    print(f"\nPHENEI reference descent -- ConOps phases 7-8")
    print("=" * 62)
    print(f"  ignition           {alt / 1e3:8.1f} km   {st['speed']:7.1f} m/s   "
          f"fpa {math.degrees(st['fpa']):.1f} deg")
    print(f"  thrust / mass      {veh.thrust_max / 1e3:8.2f} kN   {mass:7.1f} kg   "
          f"(T/W {veh.thrust_max / (mass * veh.surface_gravity):.2f})")
    print(f"  vertical profile   a_dec = {guidance.a_dec:.2f} m/s2")
    print(f"  outcome            {traj.reason}  after {traj.t[-1]:.1f} s")
    print(f"  propellant used    {propellant:8.1f} kg  "
          f"(ideal dv {veh.exhaust_velocity * math.log(mass / traj.x[-1, IDX_M]):.0f} m/s)")
    print(f"  max tilt from vertical  {tilt.max():.1f} deg   "
          f"(SIM-A16 predicts ~68 deg during braking)")
    print("\n  touchdown vs requirements")
    for label, got, want, pid in (
        ("descent rate  [m/s]", abs(final["v_vertical"]), lim["SIM-T02"], "SIM-T02"),
        ("horizontal    [m/s]", abs(final["v_horizontal"]), lim["SIM-T03"], "SIM-T03"),
        ("miss distance [m]  ", abs(final["downrange"]), lim["SIM-T01"], "SIM-T01"),
    ):
        ok = got <= want
        print(f"    {'PASS' if ok else 'FAIL'}  {label}  {got:9.3f}  <= {want:6.1f}  ({pid})")

    out = log_trajectory(traj, veh, args.out, stride=max(1, int(0.2 / args.dt)), serve=args.serve)
    if out:
        print(f"\n  rerun recording  {out.resolve()}  ({out.stat().st_size / 1e6:.1f} MB)")
        print(f"  open it with     .venv/bin/python -m rerun {out}")

    # quick-look figure, so there is something to look at without Rerun
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(loc["downrange"] / 1e3, loc["altitude"] / 1e3, lw=1.6)
    ax[0, 0].set(xlabel="downrange to target [km]", ylabel="altitude [km]", title="trajectory")
    ax[0, 1].plot(traj.t, loc["v_horizontal"], label="horizontal")
    ax[0, 1].plot(traj.t, loc["v_vertical"], label="vertical")
    ax[0, 1].set(xlabel="t [s]", ylabel="velocity [m/s]", title="velocity")
    ax[0, 1].legend()
    ax[1, 0].plot(traj.t, 100.0 * traj.thrust / veh.thrust_max)
    ax[1, 0].set(xlabel="t [s]", ylabel="throttle [%]", title="throttle")
    ax[1, 0].axhline(PARAMS.value("SIM-P04"), ls="--", c="r", lw=0.9)
    ax[1, 1].plot(traj.t, tilt)
    ax[1, 1].set(xlabel="t [s]", ylabel="tilt from local vertical [deg]",
                 title="attitude (thrust axis)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    png = Path(args.out).with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"  quick-look plot  {png.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
