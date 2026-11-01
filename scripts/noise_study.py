#!/usr/bin/env python3
"""Phase 6: how much sensor noise does the GNC loop tolerate?

One of the three engineering questions the project was set up to answer.  It
is only answerable with navigation **closed** -- in a replay the noise cannot
feed back into the trajectory, so every noise level lands in the same place.

Sweeps ``nav_noise_scale``, a single dial that multiplies every noise density
and every bias in the suite together, so a result reads as "this sensor suite
is k times worse than spec". k = 1 is the ``SIM-S*`` baseline.

    .venv/bin/python scripts/noise_study.py --x0 -173.550
    .venv/bin/python scripts/noise_study.py --x0 -173.550 --seeds 8 --cant 60
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig  # noqa: E402
from phenei_gym.params import PARAMS  # noqa: E402

LIMITS = {k: PARAMS.value(k) for k in
          ("SIM-T01", "SIM-T02", "SIM-T03", "SIM-T04", "SIM-T05")}


def one(job):
    scale, seed, x0, base, cant, config, terrain = job
    cfg = Descent6DOFConfig(
        initial_downrange=x0, navigation=True, nav_noise_scale=scale,
        nav_seed=seed, nav_cant_deg=cant, nav_config=config,
        nav_terrain=terrain, **base)
    res = Descent6DOF(cfg).run()
    td = res.touchdown()
    nav = res.nav_error
    return dict(
        scale=scale, seed=seed,
        miss=td["miss_m"], vv=abs(td["v_vertical"]), vh=td["v_horizontal"],
        tilt=td["tilt_deg"], rate=td["rate_deg_s"],
        nav_pos=float(nav[-1, 1]) if nav is not None else float("nan"),
        nav_vel=float(nav[-1, 2]) if nav is not None else float("nan"),
        reason=res.reason, propellant=res.propellant,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scales", default="0.0,0.5,1.0,2.0,5.0,10.0")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--x0", type=float, required=True, help="km")
    p.add_argument("--thrust", type=float, default=6000.0)
    p.add_argument("--altitude", type=float, default=None)
    p.add_argument("--cant", type=float, default=0.0)
    p.add_argument("--config", default="1", choices=["1", "2"])
    p.add_argument("--terrain", action="store_true",
                   help="cast altimeter beams against real terrain (~3x slower)")
    a = p.parse_args()

    if a.altitude is None:
        a.altitude = PARAMS.value("SIM-A19")
    base = dict(thrust_max=a.thrust, ignition_altitude=a.altitude * 1e3, t_max=900.0)
    scales = [float(x) for x in a.scales.split(",")]
    x0 = a.x0 * 1e3

    jobs = [(sc, sd, x0, base, a.cant, a.config, a.terrain)
            for sc in scales for sd in range(a.seeds)]
    print(f"{len(jobs)} closed-loop descents "
          f"({len(scales)} noise levels x {a.seeds} seeds), "
          f"cant {a.cant:.0f} deg, Config {a.config}, "
          f"terrain {'on' if a.terrain else 'off (FLAT -- slope bias not included)'}",
          flush=True)

    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            out.append(r)
            if i % a.seeds == 0:
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n=== sensor noise tolerance (navigation closed, cant {a.cant:.0f} deg) ===")
    print(f"  k = 1.0 is the SIM-S* baseline; every noise density and bias "
          f"scales together\n")
    hdr = (f"    {'k':>6} {'nav pos':>9} {'nav vel':>9} | {'miss':>8} {'vv':>7} "
           f"{'vh':>7} {'tilt':>7} | {'all 5':>7}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for sc in scales:
        g = [r for r in out if r["scale"] == sc]
        miss = np.array([r["miss"] for r in g])
        vv = np.array([r["vv"] for r in g])
        vh = np.array([r["vh"] for r in g])
        tilt = np.array([r["tilt"] for r in g])
        rate = np.array([r["rate"] for r in g])
        npos = np.array([r["nav_pos"] for r in g])
        nvel = np.array([r["nav_vel"] for r in g])
        ok = ((miss <= LIMITS["SIM-T01"]) & (vv <= LIMITS["SIM-T02"])
              & (vh <= LIMITS["SIM-T03"]) & (tilt <= LIMITS["SIM-T04"])
              & (rate <= LIMITS["SIM-T05"]))
        print(f"    {sc:6.1f} {npos.mean():9.1f} {nvel.mean():9.3f} | "
              f"{miss.mean():8.1f} {vv.mean():7.2f} {vh.mean():7.2f} "
              f"{tilt.mean():7.2f} | {100.0 * ok.mean():6.0f}%")
    print(f"\n  limits: miss <= {LIMITS['SIM-T01']:.0f} m, "
          f"vv <= {LIMITS['SIM-T02']:.1f} m/s, vh <= {LIMITS['SIM-T03']:.1f} m/s, "
          f"tilt <= {LIMITS['SIM-T04']:.0f} deg")
    print(f"  (nav pos/vel are the FINAL navigation errors, m and m/s)")

    bad = [r for r in out if r["reason"] != "touchdown"]
    if bad:
        reasons: dict[str, int] = {}
        for r in bad:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(f"  non-touchdown terminations: {reasons}")
    print(f"\n  {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
