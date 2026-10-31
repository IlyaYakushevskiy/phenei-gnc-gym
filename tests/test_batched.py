"""The batched CUDA env must be the same physics as the validated reference.

This test is the contract between ``envs/batched.py`` and
``dynamics/rigid6dof.py``. If either changes, this is what catches a drift.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phenei_gym.dynamics.rigid6dof import (
    I_MASS, I_OMEGA, I_POS, I_QUAT, I_VEL, derivative, make_state, rk4_step,
)
from phenei_gym.envs.batched import BatchedConfig, BatchedDescent, quat_pointing, quat_to_matrix
from phenei_gym.params import PARAMS


@pytest.fixture(scope="module")
def env():
    return BatchedDescent(BatchedConfig(n_envs=64, device="cpu", dtype=torch.float64))


def test_quaternion_to_matrix_matches_numpy(env):
    from phenei_gym.dynamics.rigid6dof import quat_to_matrix as np_q2m
    rng = np.random.default_rng(0)
    q = rng.standard_normal((16, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    got = quat_to_matrix(torch.tensor(q)).numpy()
    for i in range(16):
        assert np.allclose(got[i], np_q2m(q[i]), atol=1e-12)


def test_quat_pointing_matches_numpy():
    from phenei_gym.guidance.attitude import quat_pointing as np_qp
    rng = np.random.default_rng(1)
    v = rng.standard_normal((16, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    got = quat_pointing(torch.tensor(v)).numpy()
    for i in range(16):
        want = np_qp(v[i])
        # quaternions q and -q are the same rotation
        assert np.allclose(got[i], want, atol=1e-10) or np.allclose(got[i], -want, atol=1e-10)


def test_batched_dynamics_match_reference(env):
    """Same state, same wrench, 500 RK4 steps: positions must agree to <1 mm."""
    rng = np.random.default_rng(2)
    n = 8
    pos = np.stack([rng.uniform(-200, 200, n), rng.uniform(-200, 200, n),
                    rng.uniform(5e3, 50e3, n)], axis=1)
    vel = np.stack([rng.uniform(-300, 300, n), rng.uniform(-50, 50, n),
                    rng.uniform(-400, -100, n)], axis=1)
    q = rng.standard_normal((n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    om = rng.uniform(-0.05, 0.05, (n, 3))
    mass = rng.uniform(600, 900, n)
    force_b = np.stack([np.zeros(n), np.zeros(n), rng.uniform(1000, 5000, n)], axis=1)
    torque_b = rng.uniform(-80, 80, (n, 3))
    mdot = rng.uniform(0.3, 1.6, n)

    e = BatchedDescent(BatchedConfig(n_envs=n, device="cpu", dtype=torch.float64))
    e.pos = torch.tensor(pos); e.vel = torch.tensor(vel)
    e.quat = torch.tensor(q); e.omega = torch.tensor(om); e.mass = torch.tensor(mass)
    fb, tb, md = torch.tensor(force_b), torch.tensor(torque_b), torch.tensor(mdot)

    dt, steps = 1.0 / 200.0, 500
    for _ in range(steps):
        s = (e.pos, e.vel, e.quat, e.omega, e.mass)
        k1 = e._derivative(*s, fb, tb, md)
        s2 = tuple(a + 0.5 * dt * b for a, b in zip(s, k1))
        k2 = e._derivative(*s2, fb, tb, md)
        s3 = tuple(a + 0.5 * dt * b for a, b in zip(s, k2))
        k3 = e._derivative(*s3, fb, tb, md)
        s4 = tuple(a + dt * b for a, b in zip(s, k3))
        k4 = e._derivative(*s4, fb, tb, md)
        e.pos, e.vel, e.quat, e.omega, e.mass = tuple(
            a + (dt / 6.0) * (b + 2 * c + 2 * d + f) for a, b, c, d, f in zip(s, k1, k2, k3, k4))
        e.quat = e.quat / e.quat.norm(dim=-1, keepdim=True)

    radius = PARAMS.value("SIM-E03") * 1e3
    mu = PARAMS.value("SIM-E02")
    for i in range(n):
        # the batched env scales inertia with mass; mirror that exactly
        I = np.array([PARAMS.value("SIM-M07"), PARAMS.value("SIM-M07"), PARAMS.value("SIM-M08")])
        x = make_state(pos[i], vel[i], q[i], om[i], mass[i])
        for _ in range(steps):
            Ii = I * (x[I_MASS] / PARAMS.value("SIM-M03"))
            x = rk4_step(x, force_b[i], torque_b[i], mdot[i], Ii, mu, radius, dt)
        assert np.linalg.norm(x[I_POS] - e.pos[i].numpy()) < 1e-3
        assert np.linalg.norm(x[I_VEL] - e.vel[i].numpy()) < 1e-5
        assert abs(x[I_MASS] - float(e.mass[i])) < 1e-9


def test_allocators_both_realise_the_commanded_wrench(env):
    """Fixed pseudo-inverse vs the reference iterative allocator.

    They do not agree with each other: the batched one trades exactness for a
    single matmul. Each is checked against the COMMAND, with separate force
    and torque tolerances rather than a norm over mixed units.

    Yaw commands are kept inside the +-52.9 N.m the layout can actually
    produce -- four parallel main engines make no yaw torque at all, so RCS
    alone sets that limit, and commanding past it is saturation, not error.
    Measured over 400 random commands: iterative is exact on force and
    pitch/roll; pinv is within 0.01 % on force and ~9 N.m on pitch/roll
    (<1 % of the ~1050 N.m available) but noticeably worse on yaw.
    """
    from phenei_gym.actuators import ActuatorLayout
    layout = ActuatorLayout.from_params(6000.0)
    B = layout.effectiveness()
    wt = np.array([1.0, 20.0, 20.0, 20.0]) / np.abs(B).sum(1)
    Bw = B * wt[:, None]
    pinv = Bw.T @ np.linalg.inv(Bw @ Bw.T + 1e-9 * np.eye(4))
    rng = np.random.default_rng(3)
    for _ in range(50):
        w = np.array([rng.uniform(1500, 3500), *rng.uniform(-50, 50, 2),
                      rng.uniform(-30, 30)])
        for name, u, tol in (("iterative", layout.allocate(w), 1.0),
                             ("pinv", np.clip(pinv @ (w * wt), 0.0, 1.0), 32.0)):
            got = layout.wrench_of(u)
            assert abs(got[0] - w[0]) < 0.02 * abs(w[0]), name
            assert np.abs(got[1:] - w[1:]).max() < tol, name


def test_yaw_authority_is_the_binding_actuation_limit():
    """Documented limit: yaw comes only from RCS, at ~1/20 of pitch/roll."""
    from phenei_gym.actuators import ActuatorLayout
    layout = ActuatorLayout.from_params(6000.0)
    reach = layout.reachable()
    assert reach["Mz"][1] < 0.06 * reach["Mx"][1]


def test_mass_never_goes_below_dry(env):
    e = BatchedDescent(BatchedConfig(n_envs=32, device="cpu", dtype=torch.float64,
                                     mass0=PARAMS.value("SIM-M01") + 2.0))
    e.pos[:, 2] = 1000.0
    d = torch.zeros(32, 3, dtype=torch.float64); d[:, 2] = 1.0
    e.set_command(torch.ones(32, dtype=torch.float64), d)
    for _ in range(4000):
        e.step()
    assert float(e.mass.min()) >= PARAMS.value("SIM-M01") - 1e-9
