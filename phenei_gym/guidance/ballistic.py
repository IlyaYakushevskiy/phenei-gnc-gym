"""The minimum-energy two-impulse ballistic arc that MCR sec. 7.2 uses as its baseline.

Re-deriving this arc is the validation anchor for the whole simulation: it
reproduces two numbers the report states independently (total delta-v
3049.7 m/s and apoapsis altitude 360 km), which confirms both the orbital
mechanics and the lunar constants before any thrust is modelled.  The approach
state at braking-burn ignition (SIM-A05..A08) is then derived from it rather
than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from phenei_gym.params import PARAMS


@dataclass(frozen=True)
class BallisticArc:
    """A surface-to-surface Keplerian arc spanning ``range_angle`` radians."""

    range_angle: float      # central angle swept                  [rad]
    v_surface: float        # speed at launch = speed at impact    [m/s]
    fpa_surface: float      # flight-path angle above horizontal   [rad]
    semi_major_axis: float  # a                                    [m]
    eccentricity: float     # e                                    [-]
    apoapsis_altitude: float  # r_a - R                            [m]
    periapsis_altitude: float  # r_p - R (negative: inside the Moon) [m]
    time_of_flight: float   # launch to impact                     [s]
    mu: float
    radius: float

    @property
    def delta_v_impulsive(self) -> float:
        """Two impulses, one to launch and one to land."""
        return 2.0 * self.v_surface

    @property
    def angular_momentum(self) -> float:
        return self.radius * self.v_surface * math.cos(self.fpa_surface)

    def state_at_altitude(self, altitude: float, descending: bool = True) -> dict[str, float]:
        """Speed and flight-path angle where the arc crosses ``altitude``.

        This is how the simulation's initial condition is obtained: evaluating
        the arc at the 39 km braking-burn ignition altitude (SIM-A01).
        """
        r = self.radius + altitude
        v_sq = self.mu * (2.0 / r - 1.0 / self.semi_major_axis)
        if v_sq <= 0.0:
            raise ValueError(f"altitude {altitude} m is above the arc's apoapsis")
        v = math.sqrt(v_sq)
        cos_fpa = min(1.0, self.angular_momentum / (r * v))
        fpa = math.acos(cos_fpa)
        sign = -1.0 if descending else 1.0
        return {
            "radius": r,
            "speed": v,
            "fpa": sign * fpa,
            "v_vertical": sign * v * math.sin(fpa),
            "v_horizontal": v * cos_fpa,
        }


def min_energy_arc(
    range_angle: float | None = None,
    mu: float | None = None,
    radius: float | None = None,
) -> BallisticArc:
    """Minimum-energy surface-to-surface arc for a given central angle.

    Among all Keplerian arcs through both surface points, this is the one with
    the smallest total delta-v.  Defaults come from params.csv: the 89 deg hop
    range angle (SIM-A10) and the lunar constants (SIM-E02, SIM-E03).
    """
    if range_angle is None:
        range_angle = math.radians(PARAMS.value("SIM-A10"))
    if mu is None:
        mu = PARAMS.value("SIM-E02")
    if radius is None:
        radius = PARAMS.value("SIM-E03") * 1e3

    half = math.sin(range_angle / 2.0)
    v_sq = (mu / radius) * 2.0 * half / (1.0 + half)
    v = math.sqrt(v_sq)
    # Optimal launch elevation for a minimum-energy arc.
    fpa = math.pi / 4.0 - range_angle / 4.0

    energy = 0.5 * v_sq - mu / radius
    a = -mu / (2.0 * energy)
    h = radius * v * math.cos(fpa)
    p = h * h / mu
    e = math.sqrt(max(0.0, 1.0 - p / a))

    # True anomaly of the surface crossings; the arc runs through apoapsis.
    cos_nu = (p / radius - 1.0) / e
    nu1 = math.acos(max(-1.0, min(1.0, cos_nu)))
    tof = _kepler_tof(nu1, 2.0 * math.pi - nu1, a, e, mu)

    return BallisticArc(
        range_angle=range_angle,
        v_surface=v,
        fpa_surface=fpa,
        semi_major_axis=a,
        eccentricity=e,
        apoapsis_altitude=a * (1.0 + e) - radius,
        periapsis_altitude=a * (1.0 - e) - radius,
        time_of_flight=tof,
        mu=mu,
        radius=radius,
    )


def _kepler_tof(nu1: float, nu2: float, a: float, e: float, mu: float) -> float:
    """Time of flight between two true anomalies on an ellipse."""
    def mean_anomaly(nu: float) -> float:
        ecc_anom = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(nu / 2.0),
            math.sqrt(1.0 + e) * math.cos(nu / 2.0),
        )
        return ecc_anom - e * math.sin(ecc_anom)

    dm = mean_anomaly(nu2) - mean_anomaly(nu1)
    if dm < 0.0:
        dm += 2.0 * math.pi
    return dm * math.sqrt(a**3 / mu)
