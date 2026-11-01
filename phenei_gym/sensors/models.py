"""Sensor models: noise, bias, rate, latency, range gating and availability.

The point of this module is not the noise -- adding Gaussian noise is trivial.
It is **availability**.  The MCR's sensor suite has hard altitude gates
(``SIM-S28``-``S32``, ``SIM-S35``) and body-fixed boresights, and the hopper
flies 64-68 degrees from local vertical for 78 % of the braking burn
(``SIM-A16``).  A body-fixed "nadir" altimeter therefore spends most of the
burn pointing at the horizon and returns nothing.  Modelling that honestly is
what turns "we have three altimeters" into a navigation error budget.

Every model exposes the same contract::

    reading = sensor.sample(truth, t, rng)      # None when unavailable

so a suite is just a list and the filter decides what to do with gaps.

Conventions follow ``dynamics/rigid6dof.py``: landing-site frame, ``+Z`` up,
quaternions scalar-last body->world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from phenei_gym.dynamics.rigid6dof import quat_mul, quat_to_matrix
from phenei_gym.params import PARAMS

ARCSEC = math.pi / (180.0 * 3600.0)
G0 = 9.80665          # only to convert "mg" of accelerometer bias into m/s^2
DEG = math.pi / 180.0


# ----------------------------------------------------------------------
# Truth packet handed to every sensor
# ----------------------------------------------------------------------
@dataclass
class Truth:
    """Ground truth at one instant, in the landing-site frame."""

    t: float
    pos: np.ndarray            # (3,) m
    vel: np.ndarray            # (3,) m/s
    quat: np.ndarray           # (4,) body->world, scalar-last
    omega: np.ndarray          # (3,) rad/s, body frame
    accel_inertial: np.ndarray  # (3,) m/s^2, total inertial acceleration
    gravity: np.ndarray        # (3,) m/s^2
    terrain_height: float = 0.0   # surface height under the vehicle

    @property
    def altitude(self) -> float:
        return float(self.pos[2] - self.terrain_height)

    def specific_force_body(self) -> np.ndarray:
        """What an ideal accelerometer triad reads: non-gravitational accel."""
        rot = quat_to_matrix(self.quat)
        return rot.T @ (self.accel_inertial - self.gravity)


# ----------------------------------------------------------------------
# Ray casting against the terrain heightmap
# ----------------------------------------------------------------------
def raycast(
    origin: np.ndarray,
    direction: np.ndarray,
    terrain,
    max_range: float,
    coarse_step: float = 5.0,
    refine: int = 24,
) -> float | None:
    """Range from ``origin`` along unit ``direction`` to the heightmap.

    Marches coarsely for a sign change in ``ray_z - terrain_height`` and then
    bisects.  ``None`` if the ray never hits inside ``max_range`` -- which is
    the common case during the braking burn, and the whole reason this
    function returns an Optional rather than a float.

    ``terrain`` may be ``None``, meaning a flat surface at z = 0.
    """
    d = np.asarray(direction, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-12)

    def gap(s: float) -> float:
        p = origin + s * d
        h = 0.0 if terrain is None else float(terrain.height_at(p[0], p[1]))
        return p[2] - h

    if gap(0.0) <= 0.0:                      # already at or below the surface
        return 0.0
    if d[2] >= 0.0 and terrain is None:      # pointing up over a flat plane
        return None

    # Step size scaled to altitude: far out, a 5 m step is a waste of time.
    step = max(coarse_step, abs(gap(0.0)) / 200.0)
    s = 0.0
    g0 = gap(0.0)
    while s < max_range:
        s1 = min(s + step, max_range)
        g1 = gap(s1)
        if g1 <= 0.0:
            lo, hi = s, s1
            for _ in range(refine):
                mid = 0.5 * (lo + hi)
                if gap(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        s, g0 = s1, g1
    return None


# ----------------------------------------------------------------------
# IMU
# ----------------------------------------------------------------------
@dataclass
class IMU:
    """Safran STIM300 class MEMS IMU (``SIM-S01``..``S06``, ``S25``, ``S26``).

    Three units are flown with majority voting (``SIM-S01``); this models the
    *voted* output, so the effective white noise is reduced by sqrt(3) while
    the biases are not -- voting rejects a failed unit, it does not average
    away calibration error.  That asymmetry is the interesting part: for a
    222 s burn the error is bias-dominated, so triple redundancy buys
    reliability and almost no accuracy.
    """

    n_units: int = 3
    vote: bool = True
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # noise coefficients, SI, filled by from_params()
    arw: float = 0.0          # rad/sqrt(s)   angular random walk
    vrw: float = 0.0          # m/s/sqrt(s)   velocity random walk
    turn_on_gyro: float = 0.0  # rad/s      1 sigma
    turn_on_accel: float = 0.0  # m/s^2     1 sigma
    rrw_gyro: float = 0.0     # rad/s/sqrt(s)
    rrw_accel: float = 0.0    # m/s^2/sqrt(s)
    rate_hz: float = 2000.0

    @classmethod
    def from_params(cls, rng: np.random.Generator | None = None, **kw) -> "IMU":
        rng = rng or np.random.default_rng(0)
        # SIM-S03 0.15 deg/sqrt(h) -> rad/sqrt(s):  deg2rad / sqrt(3600)
        arw = PARAMS.value("SIM-S03") * DEG / 60.0
        # SIM-S06 0.07 m/s/sqrt(h) -> m/s/sqrt(s)
        vrw = PARAMS.value("SIM-S06") / 60.0
        turn_on_gyro = PARAMS.value("SIM-S04") * DEG / 3600.0      # deg/h -> rad/s
        turn_on_accel = PARAMS.value("SIM-S05") * 1e-3 * G0        # mg -> m/s^2
        rrw_gyro = PARAMS.value("SIM-S25") * DEG / 3600.0          # deg/h/sqrt(s)
        rrw_accel = PARAMS.value("SIM-S26") * 1e-3 * G0            # mg/sqrt(s)
        imu = cls(
            arw=arw, vrw=vrw,
            turn_on_gyro=turn_on_gyro, turn_on_accel=turn_on_accel,
            rrw_gyro=rrw_gyro, rrw_accel=rrw_accel,
            rate_hz=PARAMS.value("SIM-S02"), **kw,
        )
        imu.draw_biases(rng)
        return imu

    def draw_biases(self, rng: np.random.Generator) -> None:
        """Turn-on biases. Redrawn per Monte Carlo case, not per step."""
        self.bias_gyro = rng.normal(0.0, self.turn_on_gyro, 3)
        self.bias_accel = rng.normal(0.0, self.turn_on_accel, 3)

    def sample(self, truth: Truth, dt: float, rng: np.random.Generator):
        """Return ``(omega_measured, specific_force_measured)``, body frame."""
        k = math.sqrt(self.n_units) if self.vote else 1.0
        sg = self.arw / math.sqrt(dt) / k
        sa = self.vrw / math.sqrt(dt) / k
        self.bias_gyro += rng.normal(0.0, self.rrw_gyro * math.sqrt(dt), 3)
        self.bias_accel += rng.normal(0.0, self.rrw_accel * math.sqrt(dt), 3)
        omega_m = truth.omega + self.bias_gyro + rng.normal(0.0, sg, 3)
        accel_m = truth.specific_force_body() + self.bias_accel + rng.normal(0.0, sa, 3)
        return omega_m, accel_m

    def sigma_gyro(self, dt: float) -> float:
        k = math.sqrt(self.n_units) if self.vote else 1.0
        return self.arw / math.sqrt(dt) / k

    def sigma_accel(self, dt: float) -> float:
        k = math.sqrt(self.n_units) if self.vote else 1.0
        return self.vrw / math.sqrt(dt) / k


# ----------------------------------------------------------------------
# Star tracker
# ----------------------------------------------------------------------
@dataclass
class StarTracker:
    """Two units at 4 Hz (``SIM-S07``), off below ``SIM-S35``.

    The availability gate is the point of interest: with the trackers off
    below 2 km, attitude through the entire terminal descent is gyro
    propagation from the last fix, so gyro bias becomes unobservable exactly
    where thrust pointing error turns into horizontal touchdown velocity.
    """

    rate_hz: float = 4.0
    sigma_cross: float = 10.0 * ARCSEC
    sigma_roll: float = 60.0 * ARCSEC
    min_altitude: float = 2000.0
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    _last: float = -1e9

    @classmethod
    def from_params(cls, **kw) -> "StarTracker":
        return cls(
            rate_hz=PARAMS.value("SIM-S40"),
            sigma_cross=PARAMS.value("SIM-S08") * ARCSEC,
            sigma_roll=PARAMS.value("SIM-S27") * ARCSEC,
            min_altitude=PARAMS.value("SIM-S35"),
            **kw,
        )

    def available(self, truth: Truth) -> bool:
        return truth.altitude >= self.min_altitude

    def sample(self, truth: Truth, rng: np.random.Generator):
        """Measured quaternion, or ``None``."""
        if not self.available(truth):
            return None
        if truth.t - self._last < 1.0 / self.rate_hz - 1e-9:
            return None
        self._last = truth.t
        # Error is small-angle: two cross-boresight axes and one about it.
        b = self.boresight_body / np.linalg.norm(self.boresight_body)
        e1 = np.cross(b, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(b, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(b, e1)
        theta = (rng.normal(0.0, self.sigma_cross) * e1
                 + rng.normal(0.0, self.sigma_cross) * e2
                 + rng.normal(0.0, self.sigma_roll) * b)
        dq = np.array([0.5 * theta[0], 0.5 * theta[1], 0.5 * theta[2], 1.0])
        dq /= np.linalg.norm(dq)
        return quat_mul(truth.quat, dq)

    @property
    def covariance(self) -> np.ndarray:
        """3x3 attitude measurement covariance, body axes about the boresight."""
        return np.diag([self.sigma_cross ** 2, self.sigma_cross ** 2, self.sigma_roll ** 2])


# ----------------------------------------------------------------------
# Terrain-relative navigation
# ----------------------------------------------------------------------
@dataclass
class TRN:
    """Crater matching on the Argus cameras: a horizontal position fix.

    Deliberately **not** a rendered image (see PLAN Phase 4): the 2080 SUPER
    cannot render thousands of camera views per step, and the quantity the
    filter consumes is a position fix with a covariance.  What *is* modelled
    is the geometry that decides whether a fix exists at all -- the camera is
    body-fixed, so the vehicle's 64-68 degree flight attitude (``SIM-A16``)
    swings the boresight off the ground.
    """

    rate_hz: float = 1.0
    sigma: float = 100.0
    alt_min: float = 2000.0
    alt_max: float = 50000.0
    fov_deg: float = 60.0
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    require_ground_in_fov: bool = True
    max_incidence: float = 70.0 * DEG
    _last: float = -1e9

    @classmethod
    def from_params(cls, wide: bool = True, **kw) -> "TRN":
        lo, hi = PARAMS.range("SIM-S32")
        return cls(
            rate_hz=PARAMS.value("SIM-S14"),
            sigma=PARAMS.value("SIM-S13"),
            alt_min=lo, alt_max=hi,
            fov_deg=PARAMS.value("SIM-S11" if wide else "SIM-S10"),
            max_incidence=PARAMS.value("SIM-S43") * DEG,
            **kw,
        )

    def geometry(self, truth: Truth) -> tuple[bool, float]:
        """``(ground visible, incidence angle at the boresight)``.

        Not a nadir test.  From altitude ``h`` the surface fills a cone of
        half-angle ``arcsin(R/(R+h))`` about nadir -- 77.8 degrees at 40 km --
        so a camera pointed well off nadir still sees ground, just obliquely.
        Ground is visible when the *near* edge of the FOV cone falls inside
        the horizon cone.

        The returned incidence angle is what actually matters for crater
        matching: it stretches the ground footprint by ``1/cos(incidence)``
        and foreshortens crater rims along one axis, so the fix degrades
        smoothly with attitude instead of switching off.  Treating this as a
        binary gate was the first version of this model and it was wrong --
        it reported no TRN at all during the braking burn.
        """
        rot = quat_to_matrix(truth.quat)
        bore = rot @ (self.boresight_body / np.linalg.norm(self.boresight_body))
        off_nadir = math.acos(np.clip(-bore[2], -1.0, 1.0))
        r = PARAMS.value("SIM-E03") * 1e3
        horizon = math.asin(min(r / (r + max(truth.altitude, 1.0)), 1.0))
        half_fov = 0.5 * self.fov_deg * DEG
        near_edge = max(off_nadir - half_fov, 0.0)
        if near_edge >= horizon:
            return False, off_nadir
        # The boresight itself may miss the ground while the near edge does
        # not; the usable incidence is then the near edge.
        incidence = off_nadir if off_nadir < horizon else near_edge
        return True, incidence

    def sigma_at(self, truth: Truth) -> float:
        """Fix accuracy, degraded by footprint stretch at oblique incidence."""
        visible, incidence = self.geometry(truth)
        if not visible:
            return float("inf")
        stretch = 1.0 / max(math.cos(min(incidence, self.max_incidence)), 1e-3)
        return self.sigma * stretch

    def available(self, truth: Truth) -> bool:
        if not (self.alt_min <= truth.altitude <= self.alt_max):
            return False
        if not self.require_ground_in_fov:
            return True
        visible, incidence = self.geometry(truth)
        return visible and incidence <= self.max_incidence

    def sample(self, truth: Truth, rng: np.random.Generator):
        """``(fix_xy, sigma)`` or ``None``."""
        if not self.available(truth):
            return None
        if truth.t - self._last < 1.0 / self.rate_hz - 1e-9:
            return None
        self._last = truth.t
        sigma = self.sigma_at(truth)
        return truth.pos[:2] + rng.normal(0.0, sigma, 2), sigma


# ----------------------------------------------------------------------
# Altimeters
# ----------------------------------------------------------------------
@dataclass
class Altimeter:
    """Single-beam laser or radar rangefinder with a hard range gate.

    Measures **slant range along its own boresight**, not altitude.  Over
    sloped ground, and at any non-zero vehicle tilt, those differ, and the
    difference is a bias the filter cannot see unless it is told the terrain
    slope.  Quantifying that bias is one of the Phase 4 deliverables.
    """

    name: str = "DLEM30"
    param_id: str = "SIM-S28"
    rate_hz: float = 10.0
    sigma_floor: float = 1.0        # m
    sigma_frac: float = 0.001       # of range
    r_min: float = 200.0
    r_max: float = 8000.0
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    dust_altitude: float = 0.0      # blinded below this (0 = immune)
    _last: float = -1e9

    @classmethod
    def from_params(cls, name: str, param_id: str, dust: bool = True, **kw) -> "Altimeter":
        lo, hi = PARAMS.range(param_id)
        dust_lo, dust_hi = PARAMS.range("SIM-S24")
        return cls(
            name=name, param_id=param_id,
            rate_hz=PARAMS.value("SIM-S31"),
            sigma_floor=PARAMS.value("SIM-S36"),
            sigma_frac=PARAMS.value("SIM-S37"),
            r_min=lo, r_max=hi,
            dust_altitude=(dust_hi if dust else 0.0),
            **kw,
        )

    def sample(self, truth: Truth, rng: np.random.Generator, terrain=None):
        """Slant range, or ``None`` when gated out, off-target or dust-blind."""
        if truth.t - self._last < 1.0 / self.rate_hz - 1e-9:
            return None
        self._last = truth.t
        if truth.altitude <= self.dust_altitude:
            return None
        rot = quat_to_matrix(truth.quat)
        bore = rot @ (self.boresight_body / np.linalg.norm(self.boresight_body))
        rng_true = raycast(truth.pos, bore, terrain, self.r_max * 1.2)
        if rng_true is None or not (self.r_min <= rng_true <= self.r_max):
            return None
        sigma = self.sigma_floor + self.sigma_frac * rng_true
        return rng_true + rng.normal(0.0, sigma)

    def sigma_at(self, rng_m: float) -> float:
        return self.sigma_floor + self.sigma_frac * rng_m


# ----------------------------------------------------------------------
# Velocity sensing
# ----------------------------------------------------------------------
@dataclass
class DopplerLidar:
    """Config 2 (``SIM-S22``, ``SIM-S33``): direct 3-axis ground-relative velocity.

    Rejected in the MCR baseline on mass (+3 kg).  Modelled so the sim can
    price that decision against the touchdown velocity requirement
    (``SIM-T03`` = 0.5 m/s) instead of arguing about it.
    """

    rate_hz: float = 10.0
    sigma_floor: float = 0.01
    sigma_frac: float = 0.002
    max_altitude: float = 3000.0
    _last: float = -1e9

    @classmethod
    def from_params(cls, **kw) -> "DopplerLidar":
        return cls(sigma_floor=PARAMS.value("SIM-S38"),
                   sigma_frac=PARAMS.value("SIM-S39"),
                   max_altitude=PARAMS.value("SIM-S41"), **kw)

    def sample(self, truth: Truth, rng: np.random.Generator):
        if truth.t - self._last < 1.0 / self.rate_hz - 1e-9:
            return None
        self._last = truth.t
        if truth.altitude > self.max_altitude:
            return None
        speed = float(np.linalg.norm(truth.vel))
        sigma = self.sigma_floor + self.sigma_frac * speed
        return truth.vel + rng.normal(0.0, sigma, 3), sigma


@dataclass
class FeatureTrackVelocity:
    """Config 1 (``SIM-S21``, ``SIM-S34``): camera feature tracking + IMU.

    Horizontal only, and only once the ground is close enough to resolve
    features.  Accuracy ``SIM-S34`` is an ``Assumption (LLM)`` and is the
    single most load-bearing guess in the sensor suite -- see the note on
    that row.
    """

    rate_hz: float = 10.0
    sigma: float = 0.5
    max_altitude: float = 2000.0
    _last: float = -1e9

    @classmethod
    def from_params(cls, **kw) -> "FeatureTrackVelocity":
        return cls(sigma=PARAMS.value("SIM-S34"),
                   max_altitude=PARAMS.value("SIM-S42"), **kw)

    def sample(self, truth: Truth, rng: np.random.Generator):
        if truth.t - self._last < 1.0 / self.rate_hz - 1e-9:
            return None
        self._last = truth.t
        if truth.altitude > self.max_altitude:
            return None
        return truth.vel[:2] + rng.normal(0.0, self.sigma, 2), self.sigma


# ----------------------------------------------------------------------
# The suite
# ----------------------------------------------------------------------
@dataclass
class SensorSuite:
    """The MCR baseline (Config 1) or the Doppler-lidar variant (Config 2)."""

    imu: IMU
    star: StarTracker
    trn: TRN
    altimeters: list[Altimeter]
    velocity: DopplerLidar | FeatureTrackVelocity
    config: str = "1"

    @classmethod
    def baseline(cls, rng: np.random.Generator | None = None, config: str = "1",
                 cant_deg: float = 0.0) -> "SensorSuite":
        """``cant_deg`` tilts the downward-looking boresights aft.

        Zero cant is the MCR's implicit body-fixed nadir assumption.  A
        non-zero cant is the mitigation proposed in ``TR-SIM-14`` for the
        64-68 degree flight attitude: pitch the altimeters and the TRN camera
        so they look at the ground during the burn rather than past it.
        """
        rng = rng or np.random.default_rng(0)
        c = math.cos(cant_deg * DEG)
        s = math.sin(cant_deg * DEG)
        # Body frame: +Z up (along main-engine thrust), +X downrange.
        #
        # The cant is AFT, i.e. toward -X. The vehicle travels in +X and
        # brakes retrograde, so its thrust axis -- and therefore body +Z --
        # leans toward -X for the whole burn: measured on the reference
        # trajectory, body +Z is (-0.88, 0, +0.47) at ignition and still
        # (-0.86, 0, +0.52) at 5.5 km, rotating upright only in the last
        # 20 s. A body -Z boresight is therefore already swung toward +X and
        # canting it aft is what brings it back to nadir.
        #
        # Worth stating because the sign is easy to get wrong and a sign
        # error here is silent: canting the wrong way simply reports fewer
        # fixes, which looks like a sensor limitation rather than a bug. It
        # was verified against the flown quaternion, not a constructed one.
        down = np.array([-s, 0.0, -c])
        alts = [
            Altimeter.from_params("DLEM30", "SIM-S28", dust=True,
                                  boresight_body=down.copy()),
            Altimeter.from_params("DLEM17", "SIM-S29", dust=True,
                                  boresight_body=down.copy()),
            Altimeter.from_params("RADAR", "SIM-S30", dust=False,
                                  boresight_body=down.copy()),
        ]
        vel = (DopplerLidar.from_params() if config == "2"
               else FeatureTrackVelocity.from_params())
        return cls(
            imu=IMU.from_params(rng),
            star=StarTracker.from_params(),
            trn=TRN.from_params(wide=True, boresight_body=down.copy()),
            altimeters=alts,
            velocity=vel,
            config=config,
        )

    def availability(self, truth: Truth, terrain=None) -> dict[str, bool]:
        """Which sensors *could* produce a reading now, ignoring rate limits."""
        rot = quat_to_matrix(truth.quat)
        out = {"star": self.star.available(truth), "trn": self.trn.available(truth)}
        for a in self.altimeters:
            bore = rot @ (a.boresight_body / np.linalg.norm(a.boresight_body))
            r = raycast(truth.pos, bore, terrain, a.r_max * 1.2)
            out[a.name] = (r is not None and a.r_min <= r <= a.r_max
                           and truth.altitude > a.dust_altitude)
        out["velocity"] = truth.altitude <= self.velocity.max_altitude
        return out
