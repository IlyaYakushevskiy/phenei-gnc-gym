"""Reference (classical) descent guidance for the 3-DOF planar braking burn.

This is deliberately simple and explainable -- an altitude-priority retrograde
law, not an optimiser.  Its job is to fly the vehicle from braking-burn
ignition (SIM-A01, 39 km) to touchdown so that the simulation can be exercised
and visualised, and to be the thing the convex optimum (TR-SIM-11) and later
the RL policy are measured against.

Structure, in two channels:

*Vertical*  track a constant-deceleration descent profile
``v_z_des = -sqrt(2 a_dec h + v_td^2)``.  This profile is natural here: the
arc arrives at 39 km already descending at 557 m/s, which is exactly
``sqrt(2 * 3.98 * 39000)`` -- so a ~4 m/s^2 profile passes through the initial
condition.  Near the ground it flares automatically to ``v_td``.

*Horizontal*  spend whatever thrust is left after the vertical demand on
decelerating.  Because thrust magnitude is held at maximum during braking, the
steering angle alone trades between the two channels.

The vertical demand uses ``g_effective = mu/r^2 - r thetadot^2``, i.e. gravity
minus centrifugal relief.  That relief is large early in the burn (~1.07 of
~1.55 m/s^2 at 39 km with 1376 m/s of horizontal speed), so holding altitude
costs almost nothing until the horizontal velocity has been removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phenei_gym.dynamics import IDX_M, PlanarVehicle, to_local
from phenei_gym.params import PARAMS


@dataclass
class DescentGuidance:
    """Altitude-priority retrograde braking, then vertical terminal descent."""

    veh: PlanarVehicle
    a_dec: float             # vertical profile deceleration        [m/s^2] SIM-GD01
    k_vz: float              # descent-rate tracking gain           [1/s]   SIM-GD02
    k_vx: float              # horizontal-velocity tracking gain    [1/s]   SIM-GD03
    v_touchdown: float       # commanded descent rate at contact    [m/s]   SIM-GD04
    h_terminal: float        # switch to terminal descent below     [m]     SIM-A12
    throttle_min: float      # fraction of thrust_max               [-]     SIM-P04
    n_engines_terminal: int  # engines firing in terminal descent   [-]     TR-SIM-09

    @classmethod
    def from_params(cls, veh: PlanarVehicle, n_engines_terminal: int = 2) -> "DescentGuidance":
        return cls(
            veh=veh,
            a_dec=PARAMS.value("SIM-GD01"),
            k_vz=PARAMS.value("SIM-GD02"),
            k_vx=PARAMS.value("SIM-GD03"),
            v_touchdown=PARAMS.value("SIM-GD04"),
            h_terminal=PARAMS.value("SIM-A12"),
            throttle_min=PARAMS.value("SIM-P04") / 100.0,
            n_engines_terminal=n_engines_terminal,
        )

    # ------------------------------------------------------------------
    def desired_descent_rate(self, altitude: float) -> float:
        """Constant-deceleration profile, flaring to ``v_touchdown`` at h=0."""
        h = max(altitude, 0.0)
        return -math.sqrt(2.0 * self.a_dec * h + self.v_touchdown**2)

    def available_thrust(self, altitude: float) -> tuple[float, float]:
        """(min, max) thrust available, accounting for terminal engine shutdown.

        Below the safety descent point the vehicle cannot hover on all four
        engines at a 30 % floor (SIM-P12), so TR-SIM-09 option B shuts two
        down.  That halves both the ceiling and the floor.
        """
        t_max = self.veh.thrust_max
        if altitude <= self.h_terminal:
            frac = self.n_engines_terminal / 4.0
            return self.throttle_min * frac * t_max, frac * t_max
        return self.throttle_min * t_max, t_max

    def __call__(self, _t: float, x: np.ndarray) -> tuple[float, float]:
        loc = to_local(x, self.veh)
        h, v_z, v_x = loc["altitude"], loc["v_vertical"], loc["v_horizontal"]
        m = x[IDX_M]
        t_min, t_max = self.available_thrust(h)

        # vertical demand: hold the profile against effective gravity
        a_r_cmd = loc["g_effective"] + self.k_vz * (self.desired_descent_rate(h) - v_z)
        # horizontal demand: kill velocity in the direction of travel
        a_th_cmd = -self.k_vx * v_x

        a_mag = math.hypot(a_r_cmd, a_th_cmd)
        thrust = float(np.clip(m * a_mag, t_min, t_max))

        # Resolve the achievable direction. Radial gets priority; the tangential
        # component is whatever magnitude is left, always retrograde.
        a_avail = thrust / m
        sin_beta = float(np.clip(a_r_cmd / a_avail, -1.0, 1.0))
        beta = math.pi - math.asin(sin_beta) if a_th_cmd <= 0.0 else math.asin(sin_beta)
        return thrust, beta


def touchdown_stop(veh: PlanarVehicle, h_stop: float = 0.0):
    """Stop condition: surface contact, or departing upward from the surface."""
    def stop(_t: float, x: np.ndarray) -> str | None:
        if x[0] - veh.radius <= h_stop:
            return "touchdown"
        if x[0] - veh.radius > 60e3:
            return "altitude diverged above 60 km"
        if x[IDX_M] <= veh.mass_dry + 1e-6:
            return "propellant exhausted"
        return None
    return stop
