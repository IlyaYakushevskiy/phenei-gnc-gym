"""3-DOF planar point-mass dynamics in Moon-centred inertial polar coordinates.

This reproduces the model that PHENEI mission analysis uses (MCR report
sec. 7.2 and App. D), so that results from this simulation are directly
comparable with their CasADi/IPOPT multiphase optimum.

State and control, exactly as in the report::

    x = (r, theta, rdot, thetadot, m)      u = (T, beta)

``beta`` is measured from the **local horizontal**: ``beta = 0`` points along
``+theta`` (the direction of travel) and ``beta = pi/2`` points radially
outward.  ``theta`` runs from -89 deg at the launch site to 0 at the target.

Equations of motion (MCR App. D eq. 2)::

    rdd   = (T/m) sin(beta) + r thetadot^2 - mu / r^2
    thdd  = ((T/m) cos(beta) - 2 rdot thetadot) / r
    mdot  = -T / (Isp g0)

Assumptions inherited from the report: non-rotating Moon, point-mass gravity,
planar motion, no third-body terms, spherical surface, constant Isp.  Every
constant comes from ``design_docs/params.csv`` -- see ``PlanarVehicle.from_params``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from phenei_gym.params import PARAMS

IDX_R, IDX_THETA, IDX_RDOT, IDX_THETADOT, IDX_M = range(5)

G0 = 9.80665  # standard gravity, defining constant of Isp


@dataclass(frozen=True)
class PlanarVehicle:
    """Constants the planar EOM need. Built from params.csv, never hard-coded."""

    mu: float          # lunar gravitational parameter       [m^3/s^2]  SIM-E02
    radius: float      # lunar mean radius                   [m]        SIM-E03
    isp: float         # main engine specific impulse        [s]        SIM-P03
    thrust_max: float  # total maximum thrust                [N]        SIM-P01
    mass_dry: float    # dry mass incl. payload              [kg]       SIM-M01

    @classmethod
    def from_params(cls, thrust_max: float | None = None) -> "PlanarVehicle":
        """Build from params.csv.

        ``SIM-P01`` is a range (3-6 kN, Mattia 31.10.2026), so ``thrust_max``
        must be chosen explicitly; omitting it takes the top of the range and
        is only meaningful for reproducing the MCR trajectory, which assumed
        6 kN.
        """
        if thrust_max is None:
            thrust_max = PARAMS.range("SIM-P01")[1]
        return cls(
            mu=PARAMS.value("SIM-E02"),
            radius=PARAMS.value("SIM-E03") * 1e3,  # params.csv is in km
            isp=PARAMS.value("SIM-P03"),
            thrust_max=float(thrust_max),
            mass_dry=PARAMS.value("SIM-M01"),
        )

    @property
    def exhaust_velocity(self) -> float:
        """Isp * g0 [m/s]."""
        return self.isp * G0

    @property
    def surface_gravity(self) -> float:
        """mu / R^2 [m/s^2]. Cross-checked against SIM-E01 in the tests."""
        return self.mu / self.radius**2


def eom(x: np.ndarray, thrust: float, beta: float, veh: PlanarVehicle) -> np.ndarray:
    """Time derivative of the planar state. MCR App. D eq. 2."""
    r, _, rdot, thetadot, m = x
    a_t = thrust / m
    return np.array(
        [
            rdot,
            thetadot,
            a_t * np.sin(beta) + r * thetadot**2 - veh.mu / r**2,
            (a_t * np.cos(beta) - 2.0 * rdot * thetadot) / r,
            -thrust / veh.exhaust_velocity,
        ]
    )


def rk4_step(x: np.ndarray, thrust: float, beta: float, dt: float, veh: PlanarVehicle) -> np.ndarray:
    """One classical RK4 step with the control held constant (zero-order hold).

    Matches the transcription of MCR App. D: one RK4 step per interval with
    piecewise-constant controls.
    """
    k1 = eom(x, thrust, beta, veh)
    k2 = eom(x + 0.5 * dt * k1, thrust, beta, veh)
    k3 = eom(x + 0.5 * dt * k2, thrust, beta, veh)
    k4 = eom(x + dt * k3, thrust, beta, veh)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def to_local(x: np.ndarray, veh: PlanarVehicle) -> dict[str, float]:
    """Polar state -> the local quantities guidance and telemetry speak in.

    ``downrange`` is signed ground distance to the target at ``theta = 0``
    (negative before it).  ``v_horizontal`` is positive in the direction of
    travel; ``v_vertical`` is positive upward.
    """
    r, theta, rdot, thetadot, m = x
    return {
        "altitude": r - veh.radius,
        "downrange": veh.radius * theta,
        "v_vertical": rdot,
        "v_horizontal": r * thetadot,
        "speed": float(np.hypot(rdot, r * thetadot)),
        "mass": m,
        # Gravity minus centrifugal relief: the vertical acceleration that
        # thrust must actually supply to hold altitude. At 39 km with
        # 1376 m/s of horizontal speed the relief is ~1.07 of ~1.55 m/s^2.
        "g_effective": veh.mu / r**2 - r * thetadot**2,
    }


@dataclass
class Trajectory:
    """Recorded history of a propagation. Arrays are (n_steps,) or (n_steps, 5)."""

    t: np.ndarray
    x: np.ndarray
    thrust: np.ndarray
    beta: np.ndarray
    reason: str

    def __len__(self) -> int:
        return len(self.t)

    @property
    def propellant_used(self) -> float:
        return float(self.x[0, IDX_M] - self.x[-1, IDX_M])

    @property
    def delta_v(self) -> float:
        """Ideal delta-v actually imparted, from the rocket equation."""
        m0, mf = self.x[0, IDX_M], self.x[-1, IDX_M]
        return 0.0 if mf >= m0 else float(np.log(m0 / mf))  # scaled by c in to_dv

    def local(self, veh: PlanarVehicle) -> dict[str, np.ndarray]:
        cols: dict[str, list[float]] = {}
        for xi in self.x:
            for k, v in to_local(xi, veh).items():
                cols.setdefault(k, []).append(v)
        return {k: np.asarray(v) for k, v in cols.items()}


ControlFn = Callable[[float, np.ndarray], tuple[float, float]]
StopFn = Callable[[float, np.ndarray], str | None]


def propagate(
    x0: np.ndarray,
    control: ControlFn,
    veh: PlanarVehicle,
    dt: float,
    t_max: float,
    stop: StopFn | None = None,
) -> Trajectory:
    """Integrate until ``stop`` returns a reason, ``t_max``, or the mass floor.

    ``control(t, x) -> (thrust, beta)`` is evaluated once per step and held.
    """
    x = np.asarray(x0, dtype=float).copy()
    ts, xs, Ts, Bs = [0.0], [x.copy()], [], []
    reason, t = "t_max reached", 0.0
    # floor, never ceil: stepping past t_max would silently extend the
    # integration whenever dt does not divide t_max exactly.
    n_max = int(np.floor(t_max / dt + 1e-9))

    for _ in range(n_max):
        thrust, beta = control(t, x)
        thrust = float(np.clip(thrust, 0.0, veh.thrust_max))
        # Never burn past the dry mass: clip thrust to the propellant left.
        max_by_mass = (x[IDX_M] - veh.mass_dry) * veh.exhaust_velocity / dt
        if max_by_mass <= 0.0:
            thrust = 0.0
        else:
            thrust = min(thrust, max_by_mass)

        x = rk4_step(x, thrust, beta, dt, veh)
        t += dt
        ts.append(t)
        xs.append(x.copy())
        Ts.append(thrust)
        Bs.append(beta)

        if stop is not None:
            why = stop(t, x)
            if why:
                reason = why
                break
    else:
        Ts.append(Ts[-1] if Ts else 0.0)
        Bs.append(Bs[-1] if Bs else 0.0)

    # pad controls to match states
    while len(Ts) < len(ts):
        Ts.append(Ts[-1] if Ts else 0.0)
        Bs.append(Bs[-1] if Bs else 0.0)

    return Trajectory(
        t=np.asarray(ts),
        x=np.asarray(xs),
        thrust=np.asarray(Ts),
        beta=np.asarray(Bs),
        reason=reason,
    )
