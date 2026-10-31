"""6-DOF rigid-body dynamics for the hopper, in a landing-site inertial frame.

Frame: origin at the target landing site on the surface, ``+Z`` up (local
vertical there), ``+X`` downrange (the direction of travel), ``+Y`` completing
a right-handed set.  The Moon's centre sits at ``(0, 0, -R)``, so spherical
point-mass gravity is exact and the centrifugal/Coriolis terms of the polar
formulation appear automatically -- the frame is inertial, the Moon does not
rotate (``SIM-E05``).

This is deliberately the same physics as ``planar.py``, just without the
planar and point-mass restrictions, so a zero-torque 6-DOF run must reproduce
a 3-DOF polar run exactly.  ``tests/test_rigid6dof.py`` asserts that.

State (14,)::

    [0:3]   position      [m]      landing-site frame
    [3:6]   velocity      [m/s]    landing-site frame
    [6:10]  quaternion    [-]      body->world, SCALAR-LAST (x, y, z, w)
    [10:13] angular rate  [rad/s]  BODY frame
    [13]    mass          [kg]

What this adds over the 3-DOF model, and why it matters: attitude is now a
state rather than an instantaneous control, so thrust direction is rate
limited by the actual control authority.  MCR App. D explicitly assumes "no
attitude or throttle dynamics ... thrust magnitude and direction can change
instantaneously" and flags it as PDR work.  This is where that assumption gets
tested.
"""

from __future__ import annotations

import numpy as np

I_POS, I_VEL, I_QUAT, I_OMEGA, I_MASS = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), 13
STATE_DIM = 14


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-last convention."""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return np.array([
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
        pw * qw - px * qx - py * qy - pz * qz,
    ])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Body -> world rotation matrix. Scalar-last."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / max(np.linalg.norm(a), 1e-15)
    s = np.sin(0.5 * angle)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(0.5 * angle)])


def tilt_from_vertical(q: np.ndarray) -> float:
    """Angle [rad] between the body +Z axis and the local vertical (+Z world).

    This is the quantity SIM-A16 tracks: how far from upright the vehicle
    actually flies, and therefore how badly a body-fixed nadir sensor points.
    """
    body_z = quat_to_matrix(q)[:, 2]
    return float(np.arccos(np.clip(body_z[2], -1.0, 1.0)))


def local_up(position: np.ndarray, radius: float) -> np.ndarray:
    """Local vertical at a landing-site-frame position."""
    r_vec = np.array([position[0], position[1], position[2] + radius])
    return r_vec / np.linalg.norm(r_vec)


def tilt_from_local_vertical(q: np.ndarray, position: np.ndarray, radius: float) -> float:
    """Angle [rad] between body +Z and the LOCAL vertical at the vehicle.

    Use this, not ``tilt_from_vertical``. The landing-site frame's +Z is the
    vertical *at the target*; 150 km downrange the local vertical differs from
    it by 150/1737 rad = 5 deg, which is large enough to be mistaken for a
    control error. Measuring against the wrong vertical cost several debugging
    iterations before it was spotted.
    """
    body_z = quat_to_matrix(q)[:, 2]
    return float(np.arccos(np.clip(body_z @ local_up(position, radius), -1.0, 1.0)))


def gravity(position: np.ndarray, mu: float, radius: float) -> np.ndarray:
    """Point-mass lunar gravity at a landing-site-frame position."""
    r_vec = np.array([position[0], position[1], position[2] + radius])
    r = np.linalg.norm(r_vec)
    return -mu * r_vec / (r * r * r)


def derivative(
    state: np.ndarray,
    force_body: np.ndarray,
    torque_body: np.ndarray,
    mass_flow: float,
    inertia: np.ndarray,
    mu: float,
    radius: float,
) -> np.ndarray:
    """Time derivative of the 14-state. ``inertia`` is the body diagonal (3,)."""
    q = state[I_QUAT]
    omega = state[I_OMEGA]
    m = state[I_MASS]

    rot = quat_to_matrix(q)
    accel = rot @ force_body / m + gravity(state[I_POS], mu, radius)

    # q_dot = 0.5 * q (x) (omega, 0)   for a body->world, scalar-last quaternion
    q_dot = 0.5 * quat_mul(q, np.array([omega[0], omega[1], omega[2], 0.0]))

    omega_dot = (torque_body - np.cross(omega, inertia * omega)) / inertia

    d = np.zeros(STATE_DIM)
    d[I_POS] = state[I_VEL]
    d[I_VEL] = accel
    d[I_QUAT] = q_dot
    d[I_OMEGA] = omega_dot
    d[I_MASS] = -mass_flow
    return d


def rk4_step(
    state: np.ndarray,
    force_body: np.ndarray,
    torque_body: np.ndarray,
    mass_flow: float,
    inertia: np.ndarray,
    mu: float,
    radius: float,
    dt: float,
) -> np.ndarray:
    args = (force_body, torque_body, mass_flow, inertia, mu, radius)
    k1 = derivative(state, *args)
    k2 = derivative(state + 0.5 * dt * k1, *args)
    k3 = derivative(state + 0.5 * dt * k2, *args)
    k4 = derivative(state + dt * k3, *args)
    out = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    out[I_QUAT] /= np.linalg.norm(out[I_QUAT])  # drift control
    return out


def make_state(
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    omega: np.ndarray,
    mass: float,
) -> np.ndarray:
    s = np.zeros(STATE_DIM)
    s[I_POS] = position
    s[I_VEL] = velocity
    s[I_QUAT] = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    s[I_OMEGA] = omega
    s[I_MASS] = mass
    return s
