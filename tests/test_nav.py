"""MEKF tests.

Two things are worth testing in a Kalman filter and almost nothing else is:

1. **The Jacobians.** A sign error in ``H`` does not crash, it quietly
   degrades the estimate or diverges slowly. Every analytic Jacobian here is
   checked against a finite difference of the measurement function.
2. **Consistency.** The estimate error must actually be distributed the way
   the covariance claims. A filter that is confidently wrong is worse than
   no filter, because the controller trusts it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phenei_gym.dynamics.rigid6dof import (
    gravity, quat_conj, quat_mul, quat_to_matrix,
)
from phenei_gym.nav import MEKF, MEKFConfig, NavState, skew, small_angle_quat
from phenei_gym.nav.mekf import I_DBA, I_DBG, I_DP, I_DTH, I_DV, NX
from phenei_gym.params import PARAMS
from phenei_gym.sensors import IMU, StarTracker, Truth

RADIUS = PARAMS.value("SIM-E03") * 1e3
MU = PARAMS.value("SIM-E02")


def make_filter(seed: int = 0, **state_kw) -> MEKF:
    imu = IMU.from_params(np.random.default_rng(seed))
    cfg = MEKFConfig.from_imu(imu)
    st = NavState(pos=np.array([120.0, -40.0, 900.0]),
                  vel=np.array([-8.0, 2.0, -22.0]),
                  quat=small_angle_quat(np.array([0.05, -0.1, 0.02])))
    for k, v in state_kw.items():
        setattr(st, k, v)
    return MEKF(cfg, st)


def perfect_imu() -> IMU:
    """Noise-free, bias-free: isolates the deterministic propagation."""
    imu = IMU.from_params(np.random.default_rng(0))
    imu.arw = imu.vrw = imu.rrw_gyro = imu.rrw_accel = 0.0
    imu.bias_gyro = np.zeros(3)
    imu.bias_accel = np.zeros(3)
    return imu


# ----------------------------------------------------------------------
# Algebra
# ----------------------------------------------------------------------
def test_skew_is_the_cross_product():
    a, b = np.array([1.0, -2.0, 3.0]), np.array([0.5, 4.0, -1.0])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_small_angle_quat_matches_an_exact_rotation():
    theta = np.array([1e-4, -2e-4, 3e-5])
    q = small_angle_quat(theta)
    R = quat_to_matrix(q)
    assert np.allclose(R, np.eye(3) + skew(theta), atol=1e-7)


# ----------------------------------------------------------------------
# Propagation
# ----------------------------------------------------------------------
def test_dead_reckoning_with_a_perfect_imu_is_exact():
    """With no noise, no bias and no measurements, the filter is just an
    integrator and must track truth to integration accuracy."""
    imu = perfect_imu()
    cfg = MEKFConfig.from_imu(imu)
    pos0 = np.array([0.0, 0.0, 1000.0])
    vel0 = np.array([5.0, 0.0, -10.0])
    f = MEKF(cfg, NavState(pos=pos0.copy(), vel=vel0.copy()))
    # Free fall: specific force is zero, gravity does the work.
    dt, n = 0.005, 2000
    for _ in range(n):
        f.propagate(np.zeros(3), np.zeros(3), dt)
    t = dt * n
    g = gravity(pos0, MU, RADIUS)
    expect = pos0 + vel0 * t + 0.5 * g * t * t
    assert np.allclose(f.x.pos, expect, rtol=1e-4, atol=0.05)


def test_propagation_grows_the_covariance():
    f = make_filter()
    p0 = f.rss("pos")
    for _ in range(400):
        f.propagate(np.zeros(3), np.array([0.0, 0.0, 1.6242]), 0.005)
    assert f.rss("pos") > p0


def test_covariance_stays_symmetric_and_positive_semidefinite():
    f = make_filter()
    rng = np.random.default_rng(1)
    for _ in range(500):
        f.propagate(rng.normal(0, 0.01, 3),
                    np.array([0.0, 0.0, 1.6242]) + rng.normal(0, 0.05, 3), 0.005)
    assert np.allclose(f.P, f.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(f.P).min() > -1e-12


def test_gyro_bias_is_what_makes_attitude_drift():
    """Attitude error after a quiet coast should be bias x time."""
    imu = perfect_imu()
    cfg = MEKFConfig.from_imu(imu)
    f = MEKF(cfg, NavState(pos=np.array([0.0, 0.0, 5000.0])))
    bias = np.array([math.radians(0.5) / 3600.0, 0.0, 0.0])
    dt, n = 0.005, 20000              # 100 s
    for _ in range(n):
        f.propagate(bias, np.zeros(3), dt)       # filter's bias estimate is 0
    dq = quat_mul(f.x.quat, quat_conj(np.array([0.0, 0.0, 0.0, 1.0])))
    drift = 2.0 * np.linalg.norm(dq[:3])
    assert drift == pytest.approx(np.linalg.norm(bias) * dt * n, rel=0.01)


# ----------------------------------------------------------------------
# Jacobians, checked against finite differences
# ----------------------------------------------------------------------
def _perturb(f: MEKF, dx: np.ndarray) -> NavState:
    """Apply an error-state perturbation to a copy of the nominal state."""
    s = f.x.copy()
    s.quat = quat_mul(small_angle_quat(dx[I_DTH]), s.quat)
    s.quat /= np.linalg.norm(s.quat)
    s.vel = s.vel + dx[I_DV]
    s.pos = s.pos + dx[I_DP]
    s.bias_gyro = s.bias_gyro + dx[I_DBG]
    s.bias_accel = s.bias_accel + dx[I_DBA]
    return s


def _numeric_jacobian(f: MEKF, predict, m: int, eps: float = 1e-6) -> np.ndarray:
    J = np.zeros((m, NX))
    for i in range(NX):
        dp, dm = np.zeros(NX), np.zeros(NX)
        dp[i], dm[i] = eps, -eps
        J[:, i] = (predict(_perturb(f, dp)) - predict(_perturb(f, dm))) / (2 * eps)
    return J


def test_altimeter_jacobian_matches_finite_differences():
    """The one Jacobian with real structure: slant range depends on position
    through 1/cos and on attitude through the beam direction. A sign error
    here is silent -- it just makes the filter slowly worse."""
    f = make_filter()
    bore = np.array([-0.3, 0.0, -np.sqrt(1 - 0.09)])
    bore /= np.linalg.norm(bore)

    def predict(s: NavState) -> np.ndarray:
        b = quat_to_matrix(s.quat) @ bore
        return np.array([s.pos[2] / (-b[2])])

    rot = quat_to_matrix(f.x.quat)
    b = rot @ bore
    c = -b[2]
    rho = f.x.pos[2] / c
    H = np.zeros((1, NX))
    H[0, 8] = 1.0 / c
    H[0, I_DTH] = -(rho / c) * skew(b)[2, :]

    Hn = _numeric_jacobian(f, predict, 1)
    assert np.allclose(H, Hn, rtol=2e-4, atol=1e-6), f"\nanalytic {H}\nnumeric  {Hn}"


def test_trn_jacobian_matches_finite_differences():
    f = make_filter()

    def predict(s: NavState) -> np.ndarray:
        return s.pos[:2].copy()

    H = np.zeros((2, NX)); H[0, 6] = 1.0; H[1, 7] = 1.0
    assert np.allclose(H, _numeric_jacobian(f, predict, 2), atol=1e-8)


def test_star_tracker_jacobian_matches_finite_differences():
    f = make_filter()
    q_nominal = f.x.quat.copy()

    def predict(s: NavState) -> np.ndarray:
        dq = quat_mul(s.quat, quat_conj(q_nominal))
        if dq[3] < 0:
            dq = -dq
        return 2.0 * dq[:3]

    H = np.zeros((3, NX)); H[:, I_DTH] = np.eye(3)
    assert np.allclose(H, _numeric_jacobian(f, predict, 3), atol=1e-6)


def test_velocity_jacobian_matches_finite_differences():
    f = make_filter()

    def predict(s: NavState) -> np.ndarray:
        return s.vel.copy()

    H = np.zeros((3, NX)); H[:, I_DV] = np.eye(3)
    assert np.allclose(H, _numeric_jacobian(f, predict, 3), atol=1e-8)


# ----------------------------------------------------------------------
# Updates
# ----------------------------------------------------------------------
def test_each_update_shrinks_the_right_uncertainty():
    f = make_filter()
    a0, p0, v0 = f.rss("att"), f.rss("pos"), f.rss("vel")
    assert f.update_star(f.x.quat.copy(), StarTracker.from_params().covariance)
    assert f.rss("att") < a0
    assert f.update_trn(f.x.pos[:2].copy(), 100.0)
    assert f.rss("pos") < p0
    assert f.update_velocity(f.x.vel.copy(), 0.1)
    assert f.rss("vel") < v0


def test_a_perfect_measurement_does_not_move_the_estimate():
    """Zero residual must produce zero correction, whatever the gain."""
    f = make_filter()
    p, v, q = f.x.pos.copy(), f.x.vel.copy(), f.x.quat.copy()
    f.update_trn(p[:2].copy(), 100.0)
    f.update_velocity(v.copy(), 0.5)
    f.update_star(q.copy(), StarTracker.from_params().covariance)
    assert np.allclose(f.x.pos, p, atol=1e-9)
    assert np.allclose(f.x.vel, v, atol=1e-9)
    assert np.allclose(f.x.quat, q, atol=1e-9)


def test_outlier_gate_rejects_a_wild_measurement():
    f = make_filter()
    f.P[I_DP, I_DP] = np.eye(3) * 1.0          # confident in position
    assert not f.update_trn(f.x.pos[:2] + 5000.0, 1.0)
    assert f.rejected["trn"] == 1


def test_altimeter_rejects_a_beam_not_pointing_down():
    f = make_filter()
    assert not f.update_altimeter(500.0, np.array([0.0, 0.0, 1.0]), 1.0)


def test_update_moves_the_estimate_toward_the_measurement():
    f = make_filter()
    before = f.x.pos[0]
    target = f.x.pos[0] + 150.0
    assert f.update_trn(np.array([target, f.x.pos[1]]), 50.0)
    assert before < f.x.pos[0] <= target


# ----------------------------------------------------------------------
# Consistency: the thing that actually matters
# ----------------------------------------------------------------------
def test_filter_is_consistent_over_many_seeds():
    """NEES-style check on position. Truth is a quiet coast; the filter sees
    a noisy IMU plus 1 Hz position fixes. The RMS error must sit inside the
    covariance the filter reports -- and not be absurdly far below it, which
    would mean the filter is throwing information away.
    """
    errors, sigmas = [], []
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        imu = IMU.from_params(rng)
        cfg = MEKFConfig.from_imu(imu)
        pos_t = np.array([0.0, 0.0, 4000.0])
        vel_t = np.array([3.0, -1.0, -15.0])
        f = MEKF(cfg, NavState(pos=pos_t + rng.normal(0, 100.0, 3),
                               vel=vel_t.copy()))
        dt = 0.005
        for k in range(8000):                  # 40 s
            g = gravity(pos_t, MU, RADIUS)
            # hold truth in a straight coast: measured specific force is zero
            omega_m = imu.bias_gyro + rng.normal(0, imu.sigma_gyro(dt), 3)
            accel_m = imu.bias_accel + rng.normal(0, imu.sigma_accel(dt), 3)
            pos_t = pos_t + vel_t * dt + 0.5 * g * dt * dt
            vel_t = vel_t + g * dt
            f.propagate(omega_m, accel_m, dt)
            if k % 200 == 0:
                f.update_trn(pos_t[:2] + rng.normal(0, 100.0, 2), 100.0)
        errors.append(np.linalg.norm(f.x.pos - pos_t))
        sigmas.append(f.rss("pos"))
    rms = float(np.sqrt(np.mean(np.square(errors))))
    claimed = float(np.mean(sigmas))
    assert rms < 3.0 * claimed, f"overconfident: rms {rms:.1f} vs sigma {claimed:.1f}"
    assert rms > 0.05 * claimed, f"absurdly pessimistic: rms {rms:.1f} vs {claimed:.1f}"


def test_position_is_unobservable_without_fixes():
    """The headline navigation finding, as a test: with no TRN and no
    altimeter -- the state of affairs from 60 km down to a few km -- position
    error grows without bound and the filter correctly says so."""
    rng = np.random.default_rng(7)
    imu = IMU.from_params(rng)
    f = MEKF(MEKFConfig.from_imu(imu), NavState(pos=np.array([0.0, 0.0, 30000.0])))
    dt, marks = 0.005, []
    for k in range(1, 40001):
        tr = Truth(t=k * dt, pos=f.x.pos.copy(), vel=np.zeros(3),
                   quat=np.array([0.0, 0.0, 0.0, 1.0]), omega=np.zeros(3),
                   accel_inertial=np.zeros(3),
                   gravity=np.array([0.0, 0.0, -1.6242]))
        omega_m, accel_m = imu.sample(tr, dt, rng)
        f.propagate(omega_m, accel_m, dt)
        if k % 10000 == 0:
            marks.append(f.rss("pos"))
    assert marks == sorted(marks), "position sigma must grow monotonically"
    assert marks[-1] > 2.0 * marks[0]


# ----------------------------------------------------------------------
# Closed-loop navigation in the 6-DOF descent
# ----------------------------------------------------------------------
def test_closed_loop_navigation_still_lands():
    """The controller flying on the ESTIMATE instead of truth. This is the
    only configuration in which sensor noise can affect the trajectory at
    all, so it is what makes the noise-tolerance question answerable."""
    from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig
    cfg = Descent6DOFConfig(initial_downrange=-173550.0, navigation=True,
                            nav_cant_deg=60.0, nav_seed=0)
    res = Descent6DOF(cfg).run()
    assert res.reason == "touchdown"
    assert res.nav_error is not None and len(res.nav_error) > 1000
    td = res.touchdown()
    assert abs(td["v_vertical"]) < 3.0
    assert td["tilt_deg"] < 5.0
    # navigation error is bounded, and the filter knows roughly how big it is
    assert res.nav_error[-1, 1] < 150.0


def test_zero_noise_navigation_tracks_truth():
    """With nav_noise_scale = 0 the filter should converge essentially onto
    truth, so the closed loop must reproduce the truth-fed trajectory."""
    from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig
    base = dict(initial_downrange=-173550.0, nav_cant_deg=60.0)
    truth = Descent6DOF(Descent6DOFConfig(navigation=False, **base)).run()
    clean = Descent6DOF(Descent6DOFConfig(navigation=True, nav_noise_scale=0.0,
                                          nav_seed=0, **base)).run()
    assert clean.reason == "touchdown"
    # The handover error is still drawn (the filter starts off truth), so this
    # is not bit-exact -- but the two must land in the same neighbourhood.
    assert abs(clean.touchdown()["miss_m"] - truth.touchdown()["miss_m"]) < 120.0
    assert clean.nav_error[-1, 1] < truth.touchdown()["miss_m"] + 100.0


def test_noise_scale_actually_scales_the_sensors():
    from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig
    a = Descent6DOF(Descent6DOFConfig(navigation=True, nav_noise_scale=1.0))
    b = Descent6DOF(Descent6DOFConfig(navigation=True, nav_noise_scale=4.0))
    assert b.suite.imu.arw == pytest.approx(4.0 * a.suite.imu.arw)
    assert b.suite.trn.sigma == pytest.approx(4.0 * a.suite.trn.sigma)
    assert b.suite.star.sigma_cross == pytest.approx(4.0 * a.suite.star.sigma_cross)
    assert b.suite.altimeters[0].sigma_floor == pytest.approx(
        4.0 * a.suite.altimeters[0].sigma_floor)


def test_controller_sees_the_estimate_not_truth():
    """Guard against the obvious implementation slip: if the estimate were
    not actually wired into the control path, noise would change nothing."""
    from phenei_gym.envs.descent6dof import Descent6DOF, Descent6DOFConfig
    base = dict(initial_downrange=-173550.0, navigation=True, nav_seed=1)
    quiet = Descent6DOF(Descent6DOFConfig(nav_noise_scale=0.2, **base)).run()
    loud = Descent6DOF(Descent6DOFConfig(nav_noise_scale=8.0, **base)).run()
    assert quiet.touchdown()["miss_m"] != loud.touchdown()["miss_m"]
