"""Inner-loop attitude controller: quaternion feedback + thruster allocation.

This is the 100 Hz loop of the hierarchical architecture (``TR-SIM-04``). It is
deliberately classical and deterministic, because (a) it is the layer that
makes thruster placement a studiable parameter rather than something a network
absorbs, and (b) an RL guidance layer on top only has to command attitude and
thrust, keeping episodes short.
"""

from __future__ import annotations

import numpy as np

from phenei_gym.actuators import ActuatorLayout
from phenei_gym.dynamics.rigid6dof import I_OMEGA, I_QUAT, quat_conj, quat_mul


def attitude_error(q: np.ndarray, q_cmd: np.ndarray) -> np.ndarray:
    """Vector part of the body-frame error quaternion, sign-corrected.

    Sign correction picks the short way round; without it the controller will
    happily rotate 350 deg instead of 10.
    """
    q_err = quat_mul(quat_conj(q_cmd), q)
    if q_err[3] < 0.0:
        q_err = -q_err
    return q_err[:3]


class AttitudeController:
    """PD on the error quaternion, with gyroscopic feed-forward."""

    def __init__(self, layout: ActuatorLayout, inertia: np.ndarray,
                 omega_n: float = 1.5, zeta: float = 0.9):
        self.layout = layout
        self.inertia = np.asarray(inertia, dtype=float)
        # Standard second-order placement: tau = -2 I wn^2 q_v - 2 I zeta wn w
        self.kp = 2.0 * self.inertia * omega_n**2
        self.kd = 2.0 * self.inertia * zeta * omega_n

    def torque(self, state: np.ndarray, q_cmd: np.ndarray) -> np.ndarray:
        q, omega = state[I_QUAT], state[I_OMEGA]
        q_err_v = attitude_error(q, q_cmd)
        gyroscopic = np.cross(omega, self.inertia * omega)
        return -self.kp * q_err_v - self.kd * omega + gyroscopic

    def command(
        self, state: np.ndarray, q_cmd: np.ndarray, thrust_cmd: float,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        """Throttle vector realising ``thrust_cmd`` along body +Z and the
        attitude correction, as closely as the allocator can."""
        tau = self.torque(state, q_cmd)
        wrench = np.array([thrust_cmd, tau[0], tau[1], tau[2]])
        return self.layout.allocate(wrench, active=active)


def quat_pointing(body_z_target_world: np.ndarray) -> np.ndarray:
    """Shortest-rotation quaternion putting body +Z along the given world vector."""
    v = np.asarray(body_z_target_world, dtype=float)
    v = v / max(np.linalg.norm(v), 1e-15)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0]) if v[2] > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    axis /= s
    angle = np.arctan2(s, float(np.dot(z, v)))
    sa = np.sin(0.5 * angle)
    return np.array([axis[0] * sa, axis[1] * sa, axis[2] * sa, np.cos(0.5 * angle)])
