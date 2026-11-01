#!/usr/bin/env python3
"""Phase 7: build the deliverable recording.

One ``.rrd`` holding everything the project set out to show:

* the hopper flying the landing manoeuvre over randomised lunar terrain, with
  nozzle geometry and plumes driven by the actual commanded throttles;
* sensor telemetry -- a per-sensor availability timeline and the navigation
  error against the filter's own covariance;
* a Monte Carlo fan of dispersed trajectories converging on the landing site,
  with the touchdown scatter and the requirement circle.

    .venv/bin/python scripts/make_video.py
    .venv/bin/python scripts/make_video.py --fan 40 --cant 60
    .venv/bin/python scripts/make_video.py --x0 -173.55 --fan 0   # skip the fan

The nominal ignition point takes five full descents to solve, so pass
``--x0`` to reuse a known value.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym import terrain as terrain_mod  # noqa: E402
from phenei_gym.dynamics.rigid6dof import I_POS, I_QUAT, I_VEL, gravity, quat_conj, quat_mul  # noqa: E402
from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig  # noqa: E402
from phenei_gym.nav import MEKF, MEKFConfig, NavState  # noqa: E402
from phenei_gym.params import PARAMS  # noqa: E402
from phenei_gym.sensors import SensorSuite, Truth  # noqa: E402
from phenei_gym.viz import rerun_nav as rnav  # noqa: E402
from phenei_gym.viz.rerun6dof import log_descent6dof  # noqa: E402

from phenei_gym.mc import draw as mc_draw  # noqa: E402

RADIUS = PARAMS.value("SIM-E03") * 1e3
MU = PARAMS.value("SIM-E02")
SENSORS = ["star", "trn", "DLEM30", "DLEM17", "RADAR", "velocity"]


# ----------------------------------------------------------------------
def solve_x0(base: dict, passes: int = 4) -> tuple[float, object]:
    def fly(x0):
        return Descent6DOF(Descent6DOFConfig(initial_downrange=x0, **base)).run()
    res, x0 = fly(0.0), 0.0
    for _ in range(passes):
        x0 = res.state[0, 0] - float(res.state[-1, 0])
        res = fly(x0)
    return x0, res


def fan_case(args):
    """One dispersed trajectory, decimated for logging.

    Returns the decimated track AND the true landing position, not just the
    miss magnitude: the touchdown scatter has to be plotted where the cases
    actually landed.
    """
    seed, x0, base = args
    rng = np.random.default_rng(10000 + seed)
    disp = mc_draw(rng)
    res = Descent6DOF(Descent6DOFConfig(initial_downrange=x0, **base, **disp)).run()
    stride = max(1, len(res.t) // 400)
    return (res.state[::stride, I_POS].copy(),
            res.state[-1, I_POS].copy(),
            res.touchdown())


# ----------------------------------------------------------------------
def log_nav_track(result, cant_deg: float, config: str, terrain, seed: int = 0) -> dict:
    """Replay the nominal descent through the sensors and log the telemetry."""
    rng = np.random.default_rng(seed)
    t, x = result.t, result.state
    v = x[:, I_VEL]
    a = np.zeros_like(v)
    a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
    a[0], a[-1] = a[1], a[-2]

    suite = SensorSuite.baseline(rng, config=config, cant_deg=cant_deg)
    mcfg = MEKFConfig.from_imu(suite.imu)
    nav0 = NavState(pos=x[0, I_POS] + rng.normal(0.0, 100.0, 3),
                    vel=x[0, I_VEL] + rng.normal(0.0, 3.0, 3),
                    quat=x[0, I_QUAT].copy())
    filt = MEKF(mcfg, nav0)

    dt_sim = float(t[1] - t[0])
    stride = max(1, int(round(0.005 / dt_sim)))
    log_every = max(1, int(round(0.1 / (stride * dt_sim))))
    counts = {k: 0 for k in SENSORS}
    frames = 0

    for i in range(0, len(t) - stride, stride):
        dt = float(t[i + stride] - t[i])
        s = x[i]
        tr = Truth(t=float(t[i]), pos=s[I_POS].copy(), vel=s[I_VEL].copy(),
                   quat=s[I_QUAT].copy(), omega=s[10:13].copy(),
                   accel_inertial=a[i].copy(),
                   gravity=gravity(s[I_POS], MU, RADIUS),
                   terrain_height=float(terrain.height_at(s[I_POS][0], s[I_POS][1])))
        om, ac = suite.imu.sample(tr, dt, rng)
        filt.propagate(om, ac, dt)

        fired = {k: False for k in SENSORS}
        if (q := suite.star.sample(tr, rng)) is not None and \
                filt.update_star(q, suite.star.covariance):
            fired["star"] = True; counts["star"] += 1
        fix = suite.trn.sample(tr, rng)
        if fix is not None and filt.update_trn(fix[0], fix[1]):
            fired["trn"] = True; counts["trn"] += 1
        for alt in suite.altimeters:
            r = alt.sample(tr, rng, terrain)
            if r is not None and filt.update_altimeter(r, alt.boresight_body,
                                                       alt.sigma_at(r)):
                fired[alt.name] = True; counts[alt.name] += 1
        vm = suite.velocity.sample(tr, rng)
        if vm is not None and filt.update_velocity(
                vm[0], vm[1], horizontal_only=(config != "2")):
            fired["velocity"] = True; counts["velocity"] += 1

        frames += 1
        if frames % log_every:
            continue
        dq = quat_mul(s[I_QUAT], quat_conj(filt.x.quat))
        if dq[3] < 0:
            dq = -dq
        rnav.log_nav_frame(
            t=float(t[i]), truth_pos=s[I_POS], est_pos=filt.x.pos,
            pos_err=float(np.linalg.norm(filt.x.pos - s[I_POS])),
            pos_sigma=filt.rss("pos"),
            vel_err=float(np.linalg.norm(filt.x.vel - s[I_VEL])),
            vel_sigma=filt.rss("vel"),
            att_err_deg=math.degrees(2.0 * np.linalg.norm(dq[:3])),
            att_sigma_deg=math.degrees(filt.rss("att")),
            fired=fired, trn_fix=(fix[0] if fix is not None else None))

    rnav.log_availability_summary(counts, frames)
    # Report what was actually logged. Rerun batches rows into compressed
    # chunks, so neither the file size nor a string scan of the .rrd tells
    # you how many frames landed -- the final recording is 6.0 MB against a
    # 14.6 MB one holding strictly less, purely from chunking. This counter
    # is the only honest check that the telemetry is in there.
    counts["_nav_frames_logged"] = frames // log_every
    counts["_filter_steps"] = frames
    return counts


def log_fan(tracks, landings, touchdowns) -> None:
    """Monte Carlo trajectories and the touchdown scatter, logged static."""
    if not tracks:
        return
    rr.log("world/mc_fan",
           rr.LineStrips3D(tracks, colors=[[120, 140, 170]], radii=[12.0]),
           static=True)
    scatter = [[float(p[0]), float(p[1]), 1.0] for p in landings]
    rr.log("world/mc_touchdowns",
           rr.Points3D(scatter, colors=[[250, 120, 110]], radii=[3.0]), static=True)
    # the requirement circle
    lim = PARAMS.value("SIM-T01")
    th = np.linspace(0, 2 * np.pi, 180)
    ring = np.stack([lim * np.cos(th), lim * np.sin(th), np.full_like(th, 0.5)], axis=-1)
    rr.log("world/requirement_circle",
           rr.LineStrips3D([ring], colors=[[250, 200, 90]], radii=[1.2]), static=True)

    miss = np.array([td["miss_m"] for td in touchdowns])
    inside = 100.0 * float((miss <= lim).mean())
    doc = [
        f"# Monte Carlo fan ({len(touchdowns)} cases)", "",
        f"- CEP (50 %): **{np.percentile(miss, 50):.0f} m**",
        f"- R95: **{np.percentile(miss, 95):.0f} m**",
        f"- inside the {lim:.0f} m requirement: **{inside:.0f} %**", "",
        "Dispersions are `SIM-MC01`-`MC08`, all `Assumption (LLM)` pending TR-08.",
    ]
    rr.log("notes/monte_carlo",
           rr.TextDocument("\n".join(doc), media_type=rr.MediaType.MARKDOWN),
           static=True)


# ----------------------------------------------------------------------
def _deliverable_blueprint():
    """Layout for the deliverable recording.

    Four questions, four panes: where did it fly, what did it look like from
    alongside, what did navigation know and how well did it know it was wrong,
    and which sensors were actually available when. Rerun's automatic layout
    puts all of that in one 3D view and one shared plot axis, which is how the
    first version of this recording managed to contain every result and
    communicate none of them.
    """
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Descent + Monte Carlo fan", origin="/",
                contents=["world/**", "marker/**", "vehicle/**", "chase_cam/**"]),
            rrb.Vertical(
                # See rerun6dof._blueprint: a 2D view on a pinhole shows only
                # what is logged beneath that pinhole, so the chase pane is a
                # 3D view centred on a translation-only follow anchor, with
                # contents limited to the vehicle so auto-fit does not zoom
                # back out to the whole 175 km trajectory.
                rrb.Spatial3DView(name="Vehicle close-up", origin="chase_anchor",
                                  contents=["vehicle/**", "chase_anchor/**"]),
                rrb.TimeSeriesView(
                    name="Navigation error vs the filter's own sigma", origin="/",
                    contents=["nav/pos_error_m", "nav/pos_sigma_m",
                              "nav/vel_error_ms", "nav/vel_sigma_ms"]),
                rrb.TimeSeriesView(name="Sensor availability", origin="sensors"),
                rrb.TimeSeriesView(
                    name="Flight state", origin="/",
                    contents=["telemetry/altitude_m", "telemetry/v_vertical",
                              "telemetry/v_horizontal", "telemetry/tilt_deg"]),
            ),
            column_shares=[2.0, 1.0],
        ),
        collapse_panels=False,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="studies/phenei_landing.rrd")
    p.add_argument("--thrust", type=float, default=6000.0)
    p.add_argument("--altitude", type=float, default=None, help="km; default SIM-A19")
    p.add_argument("--cant", type=float, default=60.0,
                   help="sensor boresight cant aft, deg (TR-SIM-14)")
    p.add_argument("--config", default="1", choices=["1", "2"])
    p.add_argument("--fan", type=int, default=24, help="Monte Carlo trajectories")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--x0", type=float, default=None, help="km, skip the solve")
    p.add_argument("--seed", type=int, default=7, help="terrain seed")
    a = p.parse_args()

    if a.altitude is None:
        a.altitude = PARAMS.value("SIM-A19")
    base = dict(thrust_max=a.thrust, ignition_altitude=a.altitude * 1e3, t_max=900.0)

    t0 = time.time()
    if a.x0 is None:
        print("solving the nominal ignition point (5 descents)...", flush=True)
        x0, nominal = solve_x0(base)
    else:
        x0 = a.x0 * 1e3
        print(f"flying the nominal from x0 = {x0 / 1e3:.2f} km...", flush=True)
        nominal = Descent6DOF(Descent6DOFConfig(initial_downrange=x0, **base)).run()
    td = nominal.touchdown()
    print(f"  x0 = {x0 / 1e3:.3f} km, touchdown miss {td['miss_m']:.1f} m, "
          f"{abs(td['v_vertical']):.2f} m/s down  ({time.time() - t0:.0f} s)", flush=True)

    terrain = terrain_mod.generate(np.random.default_rng(a.seed))

    tracks, landings, touchdowns = [], [], []
    if a.fan > 0:
        print(f"flying {a.fan} dispersed cases for the fan...", flush=True)
        jobs = [(s, x0, base) for s in range(a.fan)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for tr, land, tdc in ex.map(fan_case, jobs):
                tracks.append(tr)
                landings.append(land)
                touchdowns.append(tdc)
        print(f"  done ({time.time() - t0:.0f} s)", flush=True)

    print("building the recording...", flush=True)
    rr.init("phenei_landing", spawn=False)
    rr.send_blueprint(_deliverable_blueprint())
    rnav.log_static_nav_setup(SENSORS)
    log_descent6dof(nominal, terrain, a.out, own_recording=False)
    counts = log_nav_track(nominal, a.cant, a.config, terrain)
    log_fan(tracks, landings, touchdowns)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    size = out.stat().st_size / 1e6
    print(f"\nwrote {out.resolve()}  ({size:.1f} MB)")
    print(f"  sensor fixes accepted: {counts}")
    print(f"  total {time.time() - t0:.0f} s")
    print(f"\nview it with:  scripts/view.sh {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
