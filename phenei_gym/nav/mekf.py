"""Multiplicative extended Kalman filter for the landing navigation problem.

A 15-state error-state filter: attitude error, velocity, position and the two
IMU bias triads.  IMU drives propagation; the star tracker, TRN, altimeters
and the velocity sensor correct.

Why multiplicative: attitude has three degrees of freedom but a quaternion has
four, so carrying the quaternion *in* the state vector gives a singular
covariance and forces normalisation hacks.  The MEKF instead keeps a
non-singular 3-vector attitude *error* about a nominal quaternion, corrects
the nominal multiplicatively and resets the error to zero.  This is standard
spacecraft practice (Lefferts/Markley/Shuster 1982) and is what every flown
star-tracker/gyro filter does.

The error is defined in the **world** frame::

    R_true = (I + [dtheta]x) R_nominal

which keeps the position and velocity Jacobians simple at the cost of a
rotation in the gyro-bias partial.

What this filter deliberately does *not* know:

- **the terrain.** Altimeter returns are predicted against a flat surface at
  ``z = 0``.  Real ground is sloped and rough (``SIM-E06``-``E09``), so the
  slope appears as an unmodelled measurement bias.  Quantifying that bias is
  a Phase 4 deliverable, and hiding the terrain from the filter is the only
  way to measure it.
- **the gravity gradient.** ``d(dv)/dt`` omits the ``dg/dp`` term.  Over a
  222 s burn at 1.6 m/s^2 this is negligible against the IMU bias terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from phenei_gym.dynamics.rigid6dof import gravity, quat_conj, quat_mul, quat_to_matrix
from phenei_gym.params import PARAMS

I_DTH, I_DV, I_DP, I_DBG, I_DBA = (slice(0, 3), slice(3, 6), slice(6, 9),
                                   slice(9, 12), slice(12, 15))
NX = 15

#: Smallest measurement variance the filter will accept, for numerical
#: safety only. It is NOT what keeps the filter honest -- see the Q floors.
R_FLOOR = 1e-12

# ----------------------------------------------------------------------
# Process-noise floors representing THIS FILTER'S OWN MODEL ERROR.
#
# These are not numerical fudge factors; they are the price of the two
# approximations documented in the module docstring, and without them the
# filter is provably broken rather than merely approximate:
#
#   with Q ~ 0 and a near-perfect measurement, P collapses to ~0. The
#   chi-square gate then rejects ANY subsequent measurement that disagrees by
#   even a nanometre -- and real measurements always disagree a little,
#   because the model error below is real. The filter goes blind while still
#   reporting a tiny covariance. Observed directly: after one exact position
#   fix, a fix 1 mm away was rejected as an outlier.
#
# A Kalman filter may not claim more certainty than its model deserves.
#
#: Unmodelled acceleration from the neglected gravity gradient. dg/dr =
#: 2*mu/r^3 = 1.87e-6 s^-2 at the lunar surface, so a 100 m position error
#: produces ~1.9e-4 m/s^2 of acceleration the filter does not model.
Q_ACCEL_MODEL = 2.0e-4          # m/s^2 / sqrt(s)

#: Attitude propagation error from the first-order state transition and the
#: small-angle quaternion increment.
Q_GYRO_MODEL = 1.0e-7           # rad/s / sqrt(s)

#: Keeps the position block from going exactly singular; position is driven
#: by velocity, so it needs no physical term of its own.
Q_POS_MODEL = 1.0e-6            # m / sqrt(s)

#: Bias states must keep a little freedom or they lock at their initial
#: value and never become observable again.
Q_BIAS_FLOOR = 1.0e-9


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def small_angle_quat(theta: np.ndarray) -> np.ndarray:
    """Scalar-last quaternion for a small rotation vector."""
    half = 0.5 * np.asarray(theta, dtype=float)
    q = np.array([half[0], half[1], half[2], 1.0])
    return q / np.linalg.norm(q)


@dataclass
class NavState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "NavState":
        return NavState(self.pos.copy(), self.vel.copy(), self.quat.copy(),
                        self.bias_gyro.copy(), self.bias_accel.copy())


@dataclass
class MEKFConfig:
    # initial 1-sigma uncertainties
    sigma0_att: float = math.radians(0.5)
    sigma0_vel: float = 5.0
    sigma0_pos: float = 200.0
    sigma0_bias_gyro: float = math.radians(0.5) / 3600.0
    sigma0_bias_accel: float = 0.75e-3 * 9.80665
    # process noise, filled from the IMU model
    arw: float = 0.0
    vrw: float = 0.0
    rrw_gyro: float = 0.0
    rrw_accel: float = 0.0
    # outlier gate, chi-square threshold in sigmas
    gate: float = 5.0
    mu: float = 0.0
    radius: float = 0.0

    @classmethod
    def from_imu(cls, imu, **kw) -> "MEKFConfig":
        return cls(
            arw=imu.arw, vrw=imu.vrw,
            rrw_gyro=imu.rrw_gyro, rrw_accel=imu.rrw_accel,
            sigma0_bias_gyro=max(imu.turn_on_gyro, 1e-12),
            sigma0_bias_accel=max(imu.turn_on_accel, 1e-12),
            mu=PARAMS.value("SIM-E02"),
            radius=PARAMS.value("SIM-E03") * 1e3,
            **kw,
        )


class MEKF:
    """15-state multiplicative EKF. One instance per vehicle."""

    def __init__(self, cfg: MEKFConfig, state: NavState | None = None):
        self.cfg = cfg
        self.x = state.copy() if state else NavState()
        self.P = np.diag(np.concatenate([
            np.full(3, cfg.sigma0_att ** 2),
            np.full(3, cfg.sigma0_vel ** 2),
            np.full(3, cfg.sigma0_pos ** 2),
            np.full(3, cfg.sigma0_bias_gyro ** 2),
            np.full(3, cfg.sigma0_bias_accel ** 2),
        ]))
        self.rejected = {"star": 0, "trn": 0, "alt": 0, "vel": 0}
        self.accepted = {"star": 0, "trn": 0, "alt": 0, "vel": 0}

    # ------------------------------------------------------------------
    @property
    def sigma(self) -> dict[str, np.ndarray]:
        d = np.sqrt(np.clip(np.diag(self.P), 0.0, None))
        return {"att": d[I_DTH], "vel": d[I_DV], "pos": d[I_DP],
                "bias_gyro": d[I_DBG], "bias_accel": d[I_DBA]}

    def rss(self, key: str) -> float:
        return float(np.linalg.norm(self.sigma[key]))

    # ------------------------------------------------------------------
    def propagate(self, omega_m: np.ndarray, accel_m: np.ndarray, dt: float) -> None:
        """One IMU step. ``accel_m`` is measured specific force, body frame."""
        c, x = self.cfg, self.x
        omega = omega_m - x.bias_gyro
        accel = accel_m - x.bias_accel
        rot = quat_to_matrix(x.quat)
        f_world = rot @ accel + gravity(x.pos, c.mu, c.radius)

        # --- nominal, in this order: position uses the pre-update velocity
        x.pos = x.pos + x.vel * dt + 0.5 * f_world * dt * dt
        x.vel = x.vel + f_world * dt
        dq = small_angle_quat(omega * dt)
        x.quat = quat_mul(x.quat, dq)            # body-frame increment, right
        x.quat /= np.linalg.norm(x.quat)

        # --- error covariance
        F = np.zeros((NX, NX))
        F[I_DTH, I_DBG] = -rot
        F[I_DV, I_DTH] = -skew(rot @ accel)
        F[I_DV, I_DBA] = -rot
        F[I_DP, I_DV] = np.eye(3)
        Phi = np.eye(NX) + F * dt + 0.5 * (F @ F) * dt * dt

        # Sensor noise PLUS the filter's own model error (see the Q_*_MODEL
        # constants). The model-error terms dominate when the sensors are
        # artificially quiet, which is exactly when they are needed.
        Q = np.zeros((NX, NX))
        Q[I_DTH, I_DTH] = np.eye(3) * (c.arw ** 2 + Q_GYRO_MODEL ** 2) * dt
        Q[I_DV, I_DV] = np.eye(3) * (c.vrw ** 2 + Q_ACCEL_MODEL ** 2) * dt
        Q[I_DP, I_DP] = np.eye(3) * Q_POS_MODEL ** 2 * dt
        Q[I_DBG, I_DBG] = np.eye(3) * (c.rrw_gyro ** 2 + Q_BIAS_FLOOR ** 2) * dt
        Q[I_DBA, I_DBA] = np.eye(3) * (c.rrw_accel ** 2 + Q_BIAS_FLOOR ** 2) * dt
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    def _update(self, H: np.ndarray, residual: np.ndarray, R: np.ndarray,
                tag: str) -> bool:
        """Joseph-form update with a Mahalanobis gate. True if accepted.

        ``R`` is floored at ``R_FLOOR`` for numerical safety. What actually
        keeps the gate from blinding the filter is the model-error floor on
        ``Q`` -- see the ``Q_*_MODEL`` constants.
        """
        R = R + np.eye(len(residual)) * R_FLOOR
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.rejected[tag] += 1
            return False
        maha = float(residual @ Sinv @ residual)
        if maha > self.cfg.gate ** 2 * len(residual):
            self.rejected[tag] += 1
            return False
        K = self.P @ H.T @ Sinv
        dx = K @ residual
        A = np.eye(NX) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject(dx)
        self.accepted[tag] += 1
        return True

    def _inject(self, dx: np.ndarray) -> None:
        x = self.x
        dtheta = dx[I_DTH]
        x.quat = quat_mul(small_angle_quat(dtheta), x.quat)   # world-frame, left
        x.quat /= np.linalg.norm(x.quat)
        x.vel = x.vel + dx[I_DV]
        x.pos = x.pos + dx[I_DP]
        x.bias_gyro = x.bias_gyro + dx[I_DBG]
        x.bias_accel = x.bias_accel + dx[I_DBA]

    # ------------------------------------------------------------------
    def update_star(self, quat_m: np.ndarray, cov_body: np.ndarray) -> bool:
        """Star tracker attitude fix. ``cov_body`` is 3x3 in boresight axes."""
        dq = quat_mul(quat_m, quat_conj(self.x.quat))
        if dq[3] < 0.0:
            dq = -dq
        residual = 2.0 * dq[:3]                 # world-frame small angle
        H = np.zeros((3, NX)); H[:, I_DTH] = np.eye(3)
        rot = quat_to_matrix(self.x.quat)
        return self._update(H, residual, rot @ cov_body @ rot.T, "star")

    def update_trn(self, fix_xy: np.ndarray, sigma: float) -> bool:
        residual = np.asarray(fix_xy, dtype=float) - self.x.pos[:2]
        H = np.zeros((2, NX)); H[0, 6] = 1.0; H[1, 7] = 1.0
        return self._update(H, residual, np.eye(2) * sigma ** 2, "trn")

    def update_altimeter(self, rng_m: float, boresight_body: np.ndarray,
                         sigma: float) -> bool:
        """Slant-range update, predicted against a FLAT surface at z = 0.

        The flat-surface prediction is the honest model of what the onboard
        filter knows.  Terrain slope under the beam therefore shows up as a
        residual the filter absorbs into position -- which is precisely the
        error source this is here to measure.
        """
        rot = quat_to_matrix(self.x.quat)
        b = rot @ (boresight_body / np.linalg.norm(boresight_body))
        c = -b[2]
        if c <= 1e-3:                      # beam not pointing down: unusable
            return False
        rho_hat = self.x.pos[2] / c
        if rho_hat <= 0.0:
            return False
        H = np.zeros((1, NX))
        H[0, 8] = 1.0 / c                                        # d rho / d p_z
        H[0, I_DTH] = -(rho_hat / c) * (skew(b)[2, :])           # d rho / d dtheta
        return self._update(H, np.array([rng_m - rho_hat]),
                            np.array([[sigma ** 2]]), "alt")

    def update_velocity(self, vel_m: np.ndarray, sigma: float,
                        horizontal_only: bool = False) -> bool:
        v = np.asarray(vel_m, dtype=float)
        if horizontal_only:
            residual = v[:2] - self.x.vel[:2]
            H = np.zeros((2, NX)); H[0, 3] = 1.0; H[1, 4] = 1.0
            R = np.eye(2) * sigma ** 2
        else:
            residual = v - self.x.vel
            H = np.zeros((3, NX)); H[:, I_DV] = np.eye(3)
            R = np.eye(3) * sigma ** 2
        return self._update(H, residual, R, "vel")
