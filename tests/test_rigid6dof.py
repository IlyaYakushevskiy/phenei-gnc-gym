"""6-DOF must agree with the already-validated 3-DOF polar model, and the
attitude loop must actually hold an attitude."""
import math

import numpy as np
import pytest

from phenei_gym.actuators import ActuatorLayout
from phenei_gym.dynamics import PlanarVehicle, propagate
from phenei_gym.dynamics.rigid6dof import (
    I_MASS, I_OMEGA, I_POS, I_QUAT, I_VEL,
    make_state, quat_from_axis_angle, quat_to_matrix, rk4_step, tilt_from_vertical,
)
from phenei_gym.guidance import min_energy_arc
from phenei_gym.guidance.attitude import AttitudeController, attitude_error, quat_pointing
from phenei_gym.params import PARAMS

IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def veh():
    return PlanarVehicle.from_params(thrust_max=6000.0)


def polar_to_landing_frame(r, theta, rdot, thetadot, radius):
    """(r, theta) about the target -> landing-site frame position/velocity."""
    s, c = math.sin(theta), math.cos(theta)
    pos = np.array([r * s, 0.0, r * c - radius])
    vel = np.array([rdot * s + r * thetadot * c, 0.0, rdot * c - r * thetadot * s])
    return pos, vel


def test_6dof_coast_matches_3dof_polar(veh):
    """Zero torque, zero thrust: the two formulations must be the same physics."""
    arc = min_energy_arc()
    r0 = arc.radius + 60e3
    st = arc.state_at_altitude(60e3)
    theta0, rdot0 = -0.05, st["v_vertical"]
    thetadot0 = st["v_horizontal"] / r0

    x3 = np.array([r0, theta0, rdot0, thetadot0, 880.0])
    traj = propagate(x3, lambda t, x: (0.0, 0.0), veh, dt=0.05, t_max=200.0)

    pos, vel = polar_to_landing_frame(r0, theta0, rdot0, thetadot0, veh.radius)
    s6 = make_state(pos, vel, IDENTITY_Q, np.zeros(3), 880.0)
    inertia = np.array([200.0, 200.0, 300.0])
    n = int(200.0 / 0.05)
    for _ in range(n):
        s6 = rk4_step(s6, np.zeros(3), np.zeros(3), 0.0, inertia, veh.mu, veh.radius, 0.05)

    r_f, th_f = traj.x[-1, 0], traj.x[-1, 1]
    want, _ = polar_to_landing_frame(r_f, th_f, traj.x[-1, 2], traj.x[-1, 3], veh.radius)
    # agreement to <1 m after 200 s and ~100 km of travel
    assert np.linalg.norm(s6[I_POS] - want) < 1.0


def test_quaternion_stays_normalised(veh):
    s = make_state(np.array([0.0, 0.0, 1000.0]), np.zeros(3), IDENTITY_Q,
                   np.array([0.3, -0.2, 0.5]), 800.0)
    inertia = np.array([200.0, 210.0, 300.0])
    for _ in range(2000):
        s = rk4_step(s, np.zeros(3), np.zeros(3), 0.0, inertia, veh.mu, veh.radius, 0.01)
    assert np.linalg.norm(s[I_QUAT]) == pytest.approx(1.0, abs=1e-9)


def test_torque_free_rotation_conserves_angular_momentum(veh):
    """World-frame |I omega| is conserved for a torque-free rigid body."""
    inertia = np.array([128.0, 128.0, 226.0])
    s = make_state(np.array([0.0, 0.0, 1000.0]), np.zeros(3), IDENTITY_Q,
                   np.array([0.4, -0.3, 0.2]), 800.0)
    h0 = np.linalg.norm(quat_to_matrix(s[I_QUAT]) @ (inertia * s[I_OMEGA]))
    for _ in range(5000):
        s = rk4_step(s, np.zeros(3), np.zeros(3), 0.0, inertia, veh.mu, veh.radius, 0.004)
    h1 = np.linalg.norm(quat_to_matrix(s[I_QUAT]) @ (inertia * s[I_OMEGA]))
    assert h1 == pytest.approx(h0, rel=1e-8)


def test_free_fall_matches_surface_gravity(veh):
    s = make_state(np.zeros(3), np.zeros(3), IDENTITY_Q, np.zeros(3), 700.0)
    s2 = rk4_step(s, np.zeros(3), np.zeros(3), 0.0, np.ones(3) * 100, veh.mu, veh.radius, 1.0)
    assert s2[I_VEL][2] == pytest.approx(-veh.surface_gravity, rel=1e-6)


def test_tilt_matches_commanded_pointing():
    for deg in (0.0, 15.0, 65.0, 89.0):
        q = quat_from_axis_angle([0, 1, 0], math.radians(deg))
        assert math.degrees(tilt_from_vertical(q)) == pytest.approx(deg, abs=1e-6)


def test_quat_pointing_round_trip():
    target = np.array([0.6, -0.2, 0.77])
    target /= np.linalg.norm(target)
    body_z = quat_to_matrix(quat_pointing(target))[:, 2]
    assert np.allclose(body_z, target, atol=1e-9)


def test_attitude_controller_converges(veh):
    """Closed loop must null a 30 deg error and hold it."""
    layout = ActuatorLayout.from_params(6000.0)
    inertia = np.array([PARAMS.value("SIM-M05"), PARAMS.value("SIM-M05"), PARAMS.value("SIM-M06")])
    ctl = AttitudeController(layout, inertia, omega_n=1.5, zeta=0.9)
    q_cmd = IDENTITY_Q
    s = make_state(np.array([0.0, 0.0, 5000.0]), np.zeros(3),
                   quat_from_axis_angle([0, 1, 0], math.radians(30)), np.zeros(3), 700.0)
    dt = 0.01
    for _ in range(int(12.0 / dt)):
        u = ctl.command(s, q_cmd, thrust_cmd=700.0 * veh.surface_gravity)
        w = layout.wrench_of(u)
        force = np.array([0.0, 0.0, w[0]])
        s = rk4_step(s, force, w[1:], layout.mass_flow(u), inertia, veh.mu, veh.radius, dt)
    assert math.degrees(tilt_from_vertical(s[I_QUAT])) < 1.0
    assert np.linalg.norm(s[I_OMEGA]) < 0.02


def test_main_engines_alone_cannot_yaw():
    """Rank 3, not 4: four parallel engines produce no torque about body Z.
    This is why some RCS is structural to the design (TR-SIM-08)."""
    layout = ActuatorLayout.from_params(6000.0)
    mains = np.zeros(layout.n)
    mains[layout.main_idx] = 1.0
    assert layout.rank(mains) == 3
    lo, hi = layout.reachable(mains)["Mz"]
    assert abs(lo) < 1e-9 and abs(hi) < 1e-9


def test_rcs_yaw_is_bidirectional():
    """Regression: the first pod layout had an all-positive Mz row, so the
    vehicle could only yaw one way despite B being full rank."""
    layout = ActuatorLayout.from_params(6000.0)
    lo, hi = layout.reachable()["Mz"]
    assert lo < -1.0 and hi > 1.0
    assert abs(lo + hi) < 1e-6  # symmetric authority
