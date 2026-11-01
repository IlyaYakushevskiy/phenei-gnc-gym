#!/usr/bin/env python3
"""Phase 5: the Monte Carlo campaign.

Flies the classical 6-DOF descent many times with dispersed initial
conditions and vehicle performance, and reports landing statistics against
the touchdown requirements ``SIM-T01``-``T05``.

**What the ignition point means here.** The nominal ignition point is solved
once, by iterating the descent until touchdown lands on the site, and then
every dispersed case starts from *that same* point. This is what the real
mission does: the ignition state is planned from the nominal trajectory and
uplinked, and an injection error the vehicle does not know about is what
produces a landing dispersion. Re-solving the alignment per case would be
re-planning the mission around the error and would measure nothing.

Dispersions are 1-sigma Gaussian, truncated at 3 sigma. The injection
dispersion itself (``SIM-A15``) is **unknown** -- mission analysis has not run
TR-08 -- so ``--inject-scale`` sweeps it and the campaign reports sensitivity
rather than a single fabricated answer.

    .venv/bin/python scripts/mc_campaign.py --cases 200
    .venv/bin/python scripts/mc_campaign.py --cases 200 --inject-scale 0.25
    .venv/bin/python scripts/mc_campaign.py --cases 500 --workers 12 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig  # noqa: E402
from phenei_gym.mc import SIGMA, SIGMA_SCALE, draw as mc_draw, truncated_normal  # noqa: E402,F401
from phenei_gym.params import PARAMS  # noqa: E402


@dataclass
class CaseResult:
    seed: int
    miss_m: float
    v_vertical: float
    v_horizontal: float
    tilt_deg: float
    rate_deg_s: float
    propellant_kg: float
    duration_s: float
    reason: str
    ok_t01: bool
    ok_t02: bool
    ok_t03: bool
    ok_t04: bool
    ok_t05: bool

    @property
    def all_ok(self) -> bool:
        return all([self.ok_t01, self.ok_t02, self.ok_t03, self.ok_t04, self.ok_t05])


def solve_nominal_ignition(base: dict, passes: int = 4) -> tuple[float, object]:
    """Iterate the undispersed descent until touchdown lands on the site."""
    def fly(x0: float):
        return Descent6DOF(Descent6DOFConfig(initial_downrange=x0, **base)).run()

    res = fly(0.0)
    x0 = 0.0
    for _ in range(passes):
        x0 = res.state[0, 0] - float(res.state[-1, 0])
        res = fly(x0)
    return x0, res


def run_case(args) -> CaseResult:
    seed, x0, base, inject_scale, perf_scale = args
    rng = np.random.default_rng(seed)
    disp = mc_draw(rng, inject_scale=inject_scale, perf_scale=perf_scale)

    cfg = Descent6DOFConfig(initial_downrange=x0, **base, **disp)
    res = Descent6DOF(cfg).run()
    td = res.touchdown()
    lim = {k: PARAMS.value(k) for k in
           ("SIM-T01", "SIM-T02", "SIM-T03", "SIM-T04", "SIM-T05")}
    return CaseResult(
        seed=seed,
        miss_m=td["miss_m"],
        v_vertical=abs(td["v_vertical"]),
        v_horizontal=td["v_horizontal"],
        tilt_deg=td["tilt_deg"],
        rate_deg_s=td["rate_deg_s"],
        propellant_kg=res.propellant,
        duration_s=float(res.t[-1]),
        reason=res.reason,
        ok_t01=td["miss_m"] <= lim["SIM-T01"],
        ok_t02=abs(td["v_vertical"]) <= lim["SIM-T02"],
        ok_t03=td["v_horizontal"] <= lim["SIM-T03"],
        ok_t04=td["tilt_deg"] <= lim["SIM-T04"],
        ok_t05=td["rate_deg_s"] <= lim["SIM-T05"],
    )


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def report(results: list[CaseResult], nominal, x0: float, args) -> None:
    n = len(results)
    miss = np.array([r.miss_m for r in results])
    vv = np.array([r.v_vertical for r in results])
    vh = np.array([r.v_horizontal for r in results])
    tilt = np.array([r.tilt_deg for r in results])
    rate = np.array([r.rate_deg_s for r in results])
    prop = np.array([r.propellant_kg for r in results])

    print(f"\nPHENEI Monte Carlo campaign -- {n} cases")
    print("=" * 78)
    print(f"  nominal ignition solved at downrange {x0 / 1e3:.2f} km, "
          f"altitude {args.altitude:.0f} km, thrust {args.thrust / 1e3:.1f} kN")
    ntd = nominal.touchdown()
    print(f"  nominal touchdown: miss {ntd['miss_m']:.1f} m, "
          f"{abs(ntd['v_vertical']):.2f} m/s down, {ntd['v_horizontal']:.2f} m/s across, "
          f"{ntd['tilt_deg']:.2f} deg")
    print(f"  injection dispersion scale {args.inject_scale:.2f}, "
          f"engine performance scale {args.perf_scale:.2f}")

    print("\n  1-sigma dispersions applied")
    for k, v in SIGMA.items():
        print(f"    {k:18s} {v * args.inject_scale:10.2f}")
    for k, v in SIGMA_SCALE.items():
        print(f"    {k:18s} {v * args.perf_scale * 100:10.2f} %")

    print("\n  touchdown statistics vs requirements")
    print(f"    {'quantity':>22} {'limit':>9} {'mean':>9} {'50th':>9} "
          f"{'95th':>9} {'max':>9} {'pass':>7}")
    rows = [
        ("miss [m] (SIM-T01)", "SIM-T01", miss),
        ("descent rate [m/s] (T02)", "SIM-T02", vv),
        ("horizontal [m/s] (T03)", "SIM-T03", vh),
        ("tilt [deg] (T04)", "SIM-T04", tilt),
        ("body rate [deg/s] (T05)", "SIM-T05", rate),
    ]
    for label, pid, arr in rows:
        lim = PARAMS.value(pid)
        passes = 100.0 * float((arr <= lim).mean())
        print(f"    {label:>22} {lim:9.2f} {arr.mean():9.2f} {pct(arr, 50):9.2f} "
              f"{pct(arr, 95):9.2f} {arr.max():9.2f} {passes:6.1f}%")

    all_ok = sum(r.all_ok for r in results)
    print(f"\n  ALL FIVE CRITERIA: {all_ok}/{n} = {100.0 * all_ok / n:.1f}%")

    # Which requirement is actually binding?
    fails = {pid: sum(not getattr(r, f"ok_t0{i}") for r in results)
             for i, pid in enumerate(["SIM-T01", "SIM-T02", "SIM-T03",
                                      "SIM-T04", "SIM-T05"], start=1)}
    print("  cases failing each criterion:")
    for pid, k in sorted(fails.items(), key=lambda kv: -kv[1]):
        print(f"    {pid}  {k:5d}  ({100.0 * k / n:5.1f}%)")

    print(f"\n  propellant used: mean {prop.mean():.1f} kg, "
          f"95th {pct(prop, 95):.1f} kg, max {prop.max():.1f} kg "
          f"(loaded {PARAMS.value('SIM-M03') - PARAMS.value('SIM-M01'):.0f} kg)")

    reasons: dict[str, int] = {}
    for r in results:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    print("  termination reasons: " + ", ".join(f"{k} x{v}" for k, v in reasons.items()))

    # CEP: the radius containing half the landings, the usual landing-ellipse
    # figure of merit.
    print(f"\n  CEP (50% radius) {pct(miss, 50):8.1f} m")
    print(f"  R95              {pct(miss, 95):8.1f} m   "
          f"requirement {PARAMS.value('SIM-T01'):.0f} m")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=int, default=100)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--altitude", type=float, default=None, help="km; default SIM-A19")
    p.add_argument("--thrust", type=float, default=6000.0)
    p.add_argument("--inject-scale", type=float, default=1.0,
                   help="multiplier on the injection dispersion (SIM-A15 is unknown)")
    p.add_argument("--perf-scale", type=float, default=1.0,
                   help="multiplier on the engine performance dispersion")
    p.add_argument("--guidance", default="profile", choices=["profile", "range", "zemzev"])
    p.add_argument("--csv", default=None)
    p.add_argument("--x0", type=float, default=None,
                   help="km; skip the nominal-ignition solve and use this "
                        "downrange start (it takes 5 full descents to find)")
    a = p.parse_args()

    if a.altitude is None:
        a.altitude = PARAMS.value("SIM-A19")

    base = dict(thrust_max=a.thrust, ignition_altitude=a.altitude * 1e3,
                guidance_mode=a.guidance, t_max=900.0)

    t0 = time.time()
    if a.x0 is not None:
        x0 = a.x0 * 1e3
        print(f"using x0 = {x0 / 1e3:.3f} km (given), flying the nominal once...",
              flush=True)
        nominal = Descent6DOF(Descent6DOFConfig(initial_downrange=x0, **base)).run()
    else:
        print(f"solving the nominal ignition point ({a.guidance} guidance)...",
              flush=True)
        x0, nominal = solve_nominal_ignition(base)
    print(f"  x0 = {x0 / 1e3:.3f} km  ({time.time() - t0:.0f} s)", flush=True)

    jobs = [(s, x0, base, a.inject_scale, a.perf_scale) for s in range(a.cases)]
    results: list[CaseResult] = []
    print(f"flying {a.cases} dispersed cases on {a.workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_case, jobs), 1):
            results.append(r)
            if i % 20 == 0:
                print(f"  {i}/{a.cases}  ({time.time() - t0:.0f} s)", flush=True)

    report(results, nominal, x0, a)

    if a.csv:
        path = Path(a.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
