"""Sensor model tests.

The emphasis is on *geometry and availability*, not on noise: that a Gaussian
is Gaussian is uninteresting, whereas whether a body-fixed altimeter returns
anything at a 60 degree flight attitude decides the navigation error budget.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phenei_gym.sensors import (
    IMU,
    TRN,
    Altimeter,
    DopplerLidar,
    FeatureTrackVelocity,
    SensorSuite,
    StarTracker,
    Truth,
    raycast,
)
from phenei_gym.sensors.models import DEG

RNG = lambda s=0: np.random.default_rng(s)  # noqa: E731


def pitch_quat(deg: float) -> np.ndarray:
    """Rotation about +Y by ``deg``: takes body +Z to (sin, 0, cos)."""
    a = math.radians(deg) / 2.0
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)])


def truth_at(alt: float, tilt_deg: float = 0.0, t: float = 0.0,
             x: float = 0.0) -> Truth:
    return Truth(t=t, pos=np.array([x, 0.0, alt]), vel=np.array([0.0, 0.0, -30.0]),
                 quat=pitch_quat(tilt_deg), omega=np.zeros(3),
                 accel_inertial=np.zeros(3), gravity=np.array([0.0, 0.0, -1.6242]))


# ----------------------------------------------------------------------
# Ray casting
# ----------------------------------------------------------------------
def test_raycast_straight_down_flat():
    r = raycast(np.array([0.0, 0.0, 500.0]), np.array([0.0, 0.0, -1.0]), None, 2000.0)
    assert r == pytest.approx(500.0, abs=0.01)


@pytest.mark.parametrize("tilt", [0.0, 15.0, 45.0, 60.0, 70.0])
def test_raycast_slant_range_is_altitude_over_cosine(tilt):
    """A tilted beam measures slant range, not altitude. This is the whole
    reason the altimeter range gates bite earlier than the MCR implies."""
    a = math.radians(tilt)
    d = np.array([math.sin(a), 0.0, -math.cos(a)])
    r = raycast(np.array([0.0, 0.0, 300.0]), d, None, 5000.0)
    assert r == pytest.approx(300.0 / math.cos(a), rel=1e-4)


def test_raycast_upward_over_flat_ground_misses():
    assert raycast(np.array([0.0, 0.0, 100.0]), np.array([0.3, 0.0, 0.8]),
                   None, 1000.0) is None


def test_raycast_beyond_max_range_returns_none():
    assert raycast(np.array([0.0, 0.0, 9000.0]), np.array([0.0, 0.0, -1.0]),
                   None, 1000.0) is None


def test_raycast_below_surface_is_zero():
    assert raycast(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, -1.0]),
                   None, 100.0) == 0.0


def test_raycast_follows_terrain():
    from phenei_gym import terrain as tmod
    terr = tmod.generate(RNG(3))
    origin = np.array([0.0, 0.0, 400.0])
    r = raycast(origin, np.array([0.0, 0.0, -1.0]), terr, 2000.0)
    h = float(terr.height_at(0.0, 0.0))
    assert r == pytest.approx(400.0 - h, abs=0.05)


# ----------------------------------------------------------------------
# IMU
# ----------------------------------------------------------------------
def test_imu_white_noise_scales_with_rate_and_voting():
    """ARW is a density: the per-sample sigma must go as 1/sqrt(dt), and
    three voted units must cut it by sqrt(3)."""
    imu = IMU.from_params(RNG())
    s_fast, s_slow = imu.sigma_gyro(0.001), imu.sigma_gyro(0.004)
    assert s_fast / s_slow == pytest.approx(2.0, rel=1e-9)
    solo = IMU.from_params(RNG(), n_units=1, vote=False)
    assert solo.sigma_gyro(0.005) / imu.sigma_gyro(0.005) == pytest.approx(
        math.sqrt(3.0), rel=1e-9)


def test_imu_noise_matches_declared_sigma():
    imu = IMU.from_params(RNG(1))
    imu.bias_gyro = np.zeros(3)
    imu.bias_accel = np.zeros(3)
    imu.rrw_gyro = imu.rrw_accel = 0.0          # isolate the white noise
    rng, dt, n = RNG(2), 0.005, 20000
    tr = truth_at(1000.0)
    g = np.array([imu.sample(tr, dt, rng)[0] for _ in range(n)])
    assert g.mean() == pytest.approx(0.0, abs=4.0 * imu.sigma_gyro(dt) / math.sqrt(n))
    assert g.std() == pytest.approx(imu.sigma_gyro(dt), rel=0.05)


def test_imu_turn_on_bias_is_drawn_once_not_per_sample():
    imu = IMU.from_params(RNG(4))
    imu.rrw_gyro = imu.rrw_accel = 0.0
    b0 = imu.bias_gyro.copy()
    rng, tr = RNG(5), truth_at(1000.0)
    for _ in range(100):
        imu.sample(tr, 0.005, rng)
    assert np.allclose(imu.bias_gyro, b0)


def test_imu_specific_force_excludes_gravity():
    """An accelerometer in free fall reads zero; one being thrust upward at
    1 g against lunar gravity reads the thrust, not the net motion."""
    tr = Truth(t=0.0, pos=np.array([0.0, 0.0, 100.0]), vel=np.zeros(3),
               quat=np.array([0.0, 0.0, 0.0, 1.0]), omega=np.zeros(3),
               accel_inertial=np.array([0.0, 0.0, -1.6242]),
               gravity=np.array([0.0, 0.0, -1.6242]))
    assert np.allclose(tr.specific_force_body(), np.zeros(3), atol=1e-12)
    hover = Truth(t=0.0, pos=np.array([0.0, 0.0, 100.0]), vel=np.zeros(3),
                  quat=np.array([0.0, 0.0, 0.0, 1.0]), omega=np.zeros(3),
                  accel_inertial=np.zeros(3),
                  gravity=np.array([0.0, 0.0, -1.6242]))
    assert hover.specific_force_body()[2] == pytest.approx(1.6242)


# ----------------------------------------------------------------------
# Star tracker
# ----------------------------------------------------------------------
def test_star_tracker_off_in_terminal_descent():
    """SIM-S35: the MCR turns the trackers off below 2 km, so attitude
    through the whole terminal descent is gyro propagation."""
    st = StarTracker.from_params()
    assert st.available(truth_at(5000.0))
    assert not st.available(truth_at(1999.0))
    assert st.sample(truth_at(100.0), RNG()) is None


def test_star_tracker_respects_its_rate():
    st = StarTracker.from_params()
    rng = RNG()
    assert st.sample(truth_at(5000.0, t=0.0), rng) is not None
    assert st.sample(truth_at(5000.0, t=0.1), rng) is None      # 4 Hz
    assert st.sample(truth_at(5000.0, t=0.26), rng) is not None


def test_star_tracker_roll_error_exceeds_cross_boresight():
    """Roll about the optical axis is always the worst-determined axis."""
    st = StarTracker.from_params()
    assert st.sigma_roll > st.sigma_cross
    cov = st.covariance
    assert cov[2, 2] > cov[0, 0]


# ----------------------------------------------------------------------
# TRN
# ----------------------------------------------------------------------
def test_trn_sees_ground_obliquely_not_just_at_nadir():
    """The surface fills a cone of arcsin(R/(R+h)) about nadir, so a camera
    pointed well off nadir still sees ground. Treating this as a nadir test
    wrongly reported no TRN at all during the braking burn."""
    trn = TRN.from_params()
    visible, inc = trn.geometry(truth_at(10000.0, tilt_deg=-60.0))
    assert visible
    assert math.degrees(inc) == pytest.approx(60.0, abs=0.5)


def test_trn_accuracy_degrades_as_one_over_cos_incidence():
    trn = TRN.from_params()
    s0 = trn.sigma_at(truth_at(10000.0, 0.0))
    s60 = trn.sigma_at(truth_at(10000.0, -60.0))
    assert s0 == pytest.approx(trn.sigma, rel=1e-6)
    assert s60 == pytest.approx(trn.sigma / math.cos(math.radians(60.0)), rel=1e-3)


def test_trn_uses_the_frame_edge_when_the_boresight_misses_the_ground():
    """From 10 km the horizon is 83.9 degrees off nadir, so a boresight at
    85 degrees points at empty sky -- but the lower edge of a 60 degree field
    of view is at 55 degrees and still on the ground. The model reports the
    near-edge incidence rather than giving up, which is why availability has
    to be computed from the FOV cone and not from the boresight alone."""
    trn = TRN.from_params()
    visible, inc = trn.geometry(truth_at(10000.0, -85.0))
    assert visible
    assert math.degrees(inc) == pytest.approx(55.0, abs=0.5)
    assert trn.available(truth_at(10000.0, -85.0))


def test_trn_rejects_when_the_whole_cone_is_above_the_horizon():
    """Pointed up: nothing in the field of view is ground, so no fix."""
    trn = TRN.from_params()
    visible, _ = trn.geometry(truth_at(10000.0, -125.0))
    assert not visible
    assert not trn.available(truth_at(10000.0, -125.0))


def test_trn_rejects_an_incidence_too_oblique_to_match():
    """SIM-S43: beyond 70 degrees the footprint stretch exceeds 2.9x. Low
    down the horizon is far away, so a steeply tilted camera does see ground
    -- it just sees it too obliquely to match against a nadir-referenced map."""
    trn = TRN.from_params()
    tr = truth_at(2500.0, -75.0)
    visible, inc = trn.geometry(tr)
    assert visible and math.degrees(inc) > 70.0
    assert not trn.available(tr)


def test_trn_altitude_band():
    trn = TRN.from_params()
    assert trn.available(truth_at(10000.0))
    assert not trn.available(truth_at(1500.0))      # below SIM-S32 lower limit
    assert not trn.available(truth_at(60000.0))     # above the upper limit


def test_trn_sample_returns_fix_and_sigma():
    trn = TRN.from_params()
    out = trn.sample(truth_at(10000.0, -60.0, t=0.0), RNG())
    assert out is not None
    fix, sigma = out
    assert fix.shape == (2,)
    assert sigma > trn.sigma          # oblique, so inflated


# ----------------------------------------------------------------------
# Altimeters
# ----------------------------------------------------------------------
def test_altimeter_range_gate_is_on_slant_range_not_altitude():
    """At 60 degrees of tilt the slant range is 2x the altitude, so a unit
    gated at 200 m minimum stops returning at 100 m altitude -- and starts
    returning 2x higher up than the MCR's altitude-based numbers imply."""
    a = Altimeter.from_params("DLEM30", "SIM-S28")
    assert a.r_min == pytest.approx(200.0)
    rng = RNG()
    # 150 m altitude, upright: inside the blanking interval, no return
    a._last = -1e9
    assert a.sample(truth_at(150.0, 0.0, t=0.0), rng) is None
    # same altitude at 60 deg: slant range 300 m, which is in the gate
    a._last = -1e9
    r = a.sample(truth_at(150.0, -60.0, t=10.0), rng)
    assert r is not None and r == pytest.approx(300.0, rel=0.05)


def test_radar_altimeter_survives_dust_and_lasers_do_not():
    """SIM-S24: the radar is the only altitude sensor left at touchdown."""
    rng = RNG()
    radar = Altimeter.from_params("RADAR", "SIM-S30", dust=False)
    dlem = Altimeter.from_params("DLEM17", "SIM-S29", dust=True)
    low = truth_at(12.0, 0.0, t=0.0)
    assert radar.sample(low, rng) is not None
    assert dlem.dust_altitude > 12.0
    assert dlem.sample(low, rng) is None


def test_altimeter_sigma_grows_with_range():
    a = Altimeter.from_params("DLEM30", "SIM-S28")
    assert a.sigma_at(8000.0) > a.sigma_at(200.0)
    assert a.sigma_at(8000.0) == pytest.approx(1.0 + 0.001 * 8000.0, rel=1e-6)


# ----------------------------------------------------------------------
# Velocity sensing
# ----------------------------------------------------------------------
def test_doppler_beats_feature_tracking_at_the_touchdown_limit():
    """SIM-T03 demands 0.5 m/s horizontal at touchdown. If Config 1's
    accuracy really is SIM-S34 = 0.5 m/s, the requirement cannot even be
    verified with it -- which is the TR-SIM-10 argument in one assertion."""
    dop = DopplerLidar.from_params()
    cam = FeatureTrackVelocity.from_params()
    from phenei_gym.params import PARAMS
    limit = PARAMS.value("SIM-T03")
    at_touchdown = dop.sigma_floor + dop.sigma_frac * 0.5
    assert at_touchdown < 0.1 * limit
    assert cam.sigma >= limit


def test_feature_tracking_is_horizontal_only():
    cam = FeatureTrackVelocity.from_params()
    out = cam.sample(truth_at(500.0, t=0.0), RNG())
    assert out is not None and out[0].shape == (2,)


def test_velocity_sensors_have_an_altitude_ceiling():
    cam = FeatureTrackVelocity.from_params()
    assert cam.sample(truth_at(5000.0, t=0.0), RNG()) is None


# ----------------------------------------------------------------------
# The suite, and the cant mitigation
# ----------------------------------------------------------------------
def test_aft_cant_cancels_the_flown_attitude():
    """TR-SIM-14. The vehicle brakes retrograde travelling in +X, so body +Z
    leans toward -X and a body -Z boresight swings toward +X. Canting aft
    brings it back. The sign is easy to invert and inverting it is silent --
    it just reports fewer fixes."""
    lean = -60.0
    tr = truth_at(10000.0, lean)
    uncanted = SensorSuite.baseline(RNG(), cant_deg=0.0)
    canted = SensorSuite.baseline(RNG(), cant_deg=60.0)
    _, inc_un = uncanted.trn.geometry(tr)
    _, inc_ca = canted.trn.geometry(tr)
    assert math.degrees(inc_un) == pytest.approx(60.0, abs=0.5)
    assert math.degrees(inc_ca) == pytest.approx(0.0, abs=0.5)
    assert canted.trn.sigma_at(tr) < uncanted.trn.sigma_at(tr)


def test_cant_brings_altimeter_acquisition_higher():
    """The practical payoff: at the flown attitude an uncanted DLEM 30 is
    looking through 2x the slant range, so it acquires at half the altitude."""
    tr_hi = truth_at(6000.0, -60.0)
    un = SensorSuite.baseline(RNG(), cant_deg=0.0)
    ca = SensorSuite.baseline(RNG(), cant_deg=60.0)
    assert not un.availability(tr_hi)["DLEM30"]
    assert ca.availability(tr_hi)["DLEM30"]


def test_suite_config2_swaps_in_the_doppler_lidar():
    s1 = SensorSuite.baseline(RNG(), config="1")
    s2 = SensorSuite.baseline(RNG(), config="2")
    assert isinstance(s1.velocity, FeatureTrackVelocity)
    assert isinstance(s2.velocity, DopplerLidar)


def test_nothing_but_the_imu_in_the_altimeter_free_stretch():
    """The headline availability finding: from 60 km down to a few km the
    only things working are the IMU and a degraded TRN fix."""
    s = SensorSuite.baseline(RNG(), cant_deg=0.0)
    av = s.availability(truth_at(30000.0, -60.0))
    assert not av["DLEM30"] and not av["DLEM17"] and not av["RADAR"]
    assert not av["velocity"]
    assert av["star"] and av["trn"]
