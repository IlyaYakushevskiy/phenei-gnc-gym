#!/usr/bin/env python3
"""Phase 4 acceptance: navigation error through the real descent trajectory.

Flies the 6-DOF reference descent to get truth, then replays that truth
through the sensor suite and the MEKF and reports the navigation error
against the filter's own covariance.

Replay, not closed loop, on purpose: it isolates the *estimator* from the
controller.  If the policy flew on the estimate, a navigation error and a
control error would be inseparable, and the first question to answer is
whether the filter is consistent at all.  Closing the loop is ``--closed``.

The headline numbers this produces are the ones GNC-05 and GNC-06 are
currently ``TBD`` for.

    .venv/bin/python scripts/run_nav.py
    .venv/bin/python scripts/run_nav.py --cant 45 --config 2
    .venv/bin/python scripts/run_nav.py --seeds 20 --quiet
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.dynamics.rigid6dof import (  # noqa: E402
    I_MASS, I_OMEGA, I_POS, I_QUAT, I_VEL, gravity, quat_conj, quat_mul,
    quat_to_matrix,
)
from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig  # noqa: E402
from phenei_gym.nav import MEKF, MEKFConfig, NavState  # noqa: E402
from phenei_gym.params import PARAMS  # noqa: E402
from phenei_gym.sensors import SensorSuite, Truth  # noqa: E402

RADIUS = PARAMS.value("SIM-E03") * 1e3
MU = PARAMS.value("SIM-E02")


def truth_trajectory(cfg: Descent6DOFConfig):
    """Run the reference descent and return time, state and inertial accel."""
    res = Descent6DOF(cfg).run()
    t, x = res.t, res.state
    # Inertial acceleration by central difference of the stored velocity.
    # The integrator is RK4 at 400 Hz, so this is accurate to ~1e-6 m/s^2 --
    # far below the 7 mg accelerometer bias we are trying to measure.
    v = x[:, I_VEL]
    a = np.zeros_like(v)
    a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
    a[0], a[-1] = a[1], a[-2]
    return res, t, x, a


def run_case(seed: int, cant_deg: float, config: str, terrain, args):
    rng = np.random.default_rng(seed)
    cfg = Descent6DOFConfig()
    res, t, x, a = truth_trajectory(cfg)

    suite = SensorSuite.baseline(rng, config=config, cant_deg=cant_deg)
    mcfg = MEKFConfig.from_imu(suite.imu)

    # The filter is initialised with a realistic handover error, not truth:
    # at braking ignition the state comes from the hop's own navigation, and
    # SIM-S13's 100 m TRN accuracy is the best it can be.
    x0 = x[0]
    nav0 = NavState(
        pos=x0[I_POS] + rng.normal(0.0, mcfg.sigma0_pos / math.sqrt(3), 3),
        vel=x0[I_VEL] + rng.normal(0.0, mcfg.sigma0_vel / math.sqrt(3), 3),
        quat=x0[I_QUAT].copy(),
        bias_gyro=np.zeros(3), bias_accel=np.zeros(3),
    )
    dth = rng.normal(0.0, mcfg.sigma0_att / math.sqrt(3), 3)
    nav0.quat = quat_mul(np.r_[0.5 * dth, 1.0] / np.linalg.norm(np.r_[0.5 * dth, 1.0]),
                         nav0.quat)
    filt = MEKF(mcfg, nav0)

    n = len(t)
    stride = max(1, int(round(args.imu_dt / (t[1] - t[0]))))
    log_t, log_perr, log_verr, log_aerr = [], [], [], []
    log_psig, log_vsig, log_asig, log_alt = [], [], [], []
    avail_count = {"star": 0, "trn": 0, "DLEM30": 0, "DLEM17": 0,
                   "RADAR": 0, "velocity": 0}
    samples = 0
    first_alt_fix = None

    for i in range(0, n - stride, stride):
        dt = float(t[i + stride] - t[i])
        s = x[i]
        tr = Truth(t=float(t[i]), pos=s[I_POS].copy(), vel=s[I_VEL].copy(),
                   quat=s[I_QUAT].copy(), omega=s[I_OMEGA].copy(),
                   accel_inertial=a[i].copy(),
                   gravity=gravity(s[I_POS], MU, RADIUS),
                   terrain_height=(0.0 if terrain is None else
                                   float(terrain.height_at(s[I_POS][0], s[I_POS][1]))))

        omega_m, accel_m = suite.imu.sample(tr, dt, rng)
        filt.propagate(omega_m, accel_m, dt)

        if (q := suite.star.sample(tr, rng)) is not None:
            if filt.update_star(q, suite.star.covariance):
                avail_count["star"] += 1
        if (fix := suite.trn.sample(tr, rng)) is not None:
            if filt.update_trn(fix[0], fix[1]):
                avail_count["trn"] += 1
        for alt in suite.altimeters:
            if (r := alt.sample(tr, rng, terrain)) is not None:
                if filt.update_altimeter(r, alt.boresight_body, alt.sigma_at(r)):
                    avail_count[alt.name] += 1
                    if first_alt_fix is None:
                        first_alt_fix = (float(t[i]), tr.altitude, alt.name)
        if (vm := suite.velocity.sample(tr, rng)) is not None:
            horiz = config != "2"
            if filt.update_velocity(vm[0], vm[1], horizontal_only=horiz):
                avail_count["velocity"] += 1

        samples += 1
        dq = quat_mul(s[I_QUAT], quat_conj(filt.x.quat))
        if dq[3] < 0:
            dq = -dq
        log_t.append(float(t[i]))
        log_perr.append(float(np.linalg.norm(filt.x.pos - s[I_POS])))
        log_verr.append(float(np.linalg.norm(filt.x.vel - s[I_VEL])))
        log_aerr.append(math.degrees(2.0 * np.linalg.norm(dq[:3])))
        log_psig.append(filt.rss("pos"))
        log_vsig.append(filt.rss("vel"))
        log_asig.append(math.degrees(filt.rss("att")))
        log_alt.append(tr.altitude)

    return {
        "res": res, "filt": filt, "suite": suite,
        "t": np.array(log_t), "alt": np.array(log_alt),
        "perr": np.array(log_perr), "verr": np.array(log_verr),
        "aerr": np.array(log_aerr),
        "psig": np.array(log_psig), "vsig": np.array(log_vsig),
        "asig": np.array(log_asig),
        "avail": avail_count, "samples": samples, "first_alt_fix": first_alt_fix,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cant", type=float, default=0.0,
                   help="sensor boresight cant aft, deg (TR-SIM-14 mitigation)")
    p.add_argument("--config", default="1", choices=["1", "2"],
                   help="1 = camera feature tracking, 2 = Doppler LIDAR")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--imu-dt", type=float, default=0.005,
                   help="filter propagation step, s (IMU runs at 2 kHz)")
    p.add_argument("--terrain", action="store_true",
                   help="cast altimeter beams against real terrain")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    terrain = None
    if a.terrain:
        from phenei_gym import terrain as terrain_mod
        terrain = terrain_mod.generate(np.random.default_rng(7))

    finals = []
    for seed in range(a.seeds):
        out = run_case(seed, a.cant, a.config, terrain, a)
        finals.append(out)
        if a.seeds == 1 and not a.quiet:
            report(out, a)
    if a.seeds > 1:
        summarise(finals, a)
    return 0


def report(out, args) -> None:
    t, alt = out["t"], out["alt"]
    print(f"\n=== Navigation error, Config {args.config}, "
          f"boresight cant {args.cant:.0f} deg ===")
    print(f"trajectory {t[-1]:.0f} s, {out['samples']} filter steps "
          f"at {args.imu_dt * 1000:.0f} ms\n")

    print("  measurement updates accepted")
    for k, v in out["avail"].items():
        frac = 100.0 * v / max(out["samples"], 1)
        print(f"    {k:9s} {v:7d}  ({frac:5.1f}% of steps)")
    r = out["filt"].rejected
    if sum(r.values()):
        print(f"    rejected by the chi-square gate: {r}")
    if out["first_alt_fix"]:
        ft, fa, fn = out["first_alt_fix"]
        print(f"    first altimeter fix: {fn} at t={ft:.0f} s, altitude {fa:.0f} m")

    print("\n  navigation error vs the filter's own 1-sigma")
    bands = [(60000, 10000), (10000, 3000), (3000, 500), (500, 50), (50, -1)]
    print(f"    {'altitude band':>18} | {'pos err':>8} {'pos sig':>8} | "
          f"{'vel err':>8} {'vel sig':>8} | {'att err':>8} {'att sig':>8}")
    for hi, lo in bands:
        m = (alt <= hi) & (alt > lo)
        if not m.any():
            continue
        label = f"{hi/1000:.0f}-{max(lo,0)/1000:.1f} km" if hi > 1000 else f"{hi:.0f}-{max(lo,0):.0f} m"
        print(f"    {label:>18} | {out['perr'][m].mean():8.1f} {out['psig'][m].mean():8.1f} | "
              f"{out['verr'][m].mean():8.2f} {out['vsig'][m].mean():8.2f} | "
              f"{out['aerr'][m].mean():8.3f} {out['asig'][m].mean():8.3f}")

    print("\n  AT TOUCHDOWN")
    print(f"    position error  {out['perr'][-1]:8.1f} m   (filter says {out['psig'][-1]:.1f})")
    print(f"    velocity error  {out['verr'][-1]:8.2f} m/s (filter says {out['vsig'][-1]:.2f})")
    print(f"    attitude error  {out['aerr'][-1]:8.3f} deg (filter says {out['asig'][-1]:.3f})")
    b = out["filt"].x.bias_gyro
    tb = out["suite"].imu.bias_gyro
    print(f"    gyro bias est   {np.degrees(b) * 3600} deg/h")
    print(f"    gyro bias true  {np.degrees(tb) * 3600} deg/h")

    td = out["res"].touchdown()
    print(f"\n  (truth touchdown: {td['v_vertical']:.2f} m/s vertical, "
          f"{td['v_horizontal']:.2f} m/s horizontal, {td['tilt_deg']:.2f} deg tilt)")


def summarise(finals, args) -> None:
    pe = np.array([f["perr"][-1] for f in finals])
    ve = np.array([f["verr"][-1] for f in finals])
    ae = np.array([f["aerr"][-1] for f in finals])
    print(f"\n=== {len(finals)} seeds, Config {args.config}, cant {args.cant:.0f} deg ===")
    for name, arr, unit in (("position", pe, "m"), ("velocity", ve, "m/s"),
                            ("attitude", ae, "deg")):
        print(f"  touchdown {name:9s} error  mean {arr.mean():8.2f}  "
              f"95th {np.percentile(arr, 95):8.2f}  max {arr.max():8.2f}  {unit}")


if __name__ == "__main__":
    raise SystemExit(main())
