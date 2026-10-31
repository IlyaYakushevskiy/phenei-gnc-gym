"""Dynamics must reproduce the MCR report before anything downstream is trusted."""
import math

import numpy as np
import pytest

from phenei_gym.dynamics import IDX_R, PlanarVehicle, eom, propagate, rk4_step, to_local
from phenei_gym.guidance import min_energy_arc
from phenei_gym.params import PARAMS


@pytest.fixture(scope="module")
def veh():
    return PlanarVehicle.from_params(thrust_max=6000.0)


@pytest.fixture(scope="module")
def arc():
    return min_energy_arc()


def test_surface_gravity_matches_params(veh):
    assert veh.surface_gravity == pytest.approx(PARAMS.value("SIM-E01"), abs=1e-4)


def test_mcr_published_numbers(arc):
    assert arc.delta_v_impulsive == pytest.approx(3049.7, rel=2e-4)
    assert arc.time_of_flight / 60.0 == pytest.approx(41.1, abs=0.1)
    assert arc.apoapsis_altitude / 1e3 == pytest.approx(360.0, abs=1.0)


def test_zero_thrust_hover_is_free_fall(veh):
    """With no thrust at rest, radial acceleration must equal -g."""
    x = np.array([veh.radius, 0.0, 0.0, 0.0, 700.0])
    assert eom(x, 0.0, 0.0, veh)[2] == pytest.approx(-veh.surface_gravity, rel=1e-9)


def test_vertical_thrust_equal_to_weight_holds_altitude(veh):
    """beta = pi/2 with T = m g must give zero radial acceleration."""
    m = 700.0
    x = np.array([veh.radius, 0.0, 0.0, 0.0, m])
    thrust = m * veh.surface_gravity
    assert eom(x, thrust, math.pi / 2.0, veh)[2] == pytest.approx(0.0, abs=1e-9)


def test_mass_flow_follows_isp(veh):
    x = np.array([veh.radius, 0.0, 0.0, 0.0, 700.0])
    assert eom(x, 3000.0, 0.0, veh)[4] == pytest.approx(-3000.0 / veh.exhaust_velocity, rel=1e-12)


def test_ballistic_arc_closes_at_the_target(veh, arc):
    r0 = arc.radius
    x0 = np.array([r0, -arc.range_angle,
                   arc.v_surface * math.sin(arc.fpa_surface),
                   arc.v_surface * math.cos(arc.fpa_surface) / r0, 900.0])
    traj = propagate(
        x0, lambda t, x: (0.0, 0.0), veh, dt=0.05, t_max=4000.0,
        stop=lambda t, x: "surface" if t > 60.0 and x[IDX_R] <= r0 else None,
    )
    assert math.degrees(abs(traj.x[-1, 1])) < 0.05
    assert traj.t[-1] == pytest.approx(arc.time_of_flight, abs=1.0)


def test_coast_conserves_energy_and_angular_momentum(veh, arc):
    r0 = arc.radius + 50e3
    x0 = np.array([r0, 0.0, -400.0, 1300.0 / r0, 900.0])
    traj = propagate(x0, lambda t, x: (0.0, 0.0), veh, dt=0.1, t_max=500.0)
    r, _, rd, thd, _ = traj.x.T
    energy = 0.5 * (rd**2 + (r * thd) ** 2) - veh.mu / r
    angmom = r**2 * thd
    assert np.ptp(energy) / abs(energy[0]) < 1e-9
    assert np.ptp(angmom) / abs(angmom[0]) < 1e-9


def test_propagate_never_steps_past_t_max(veh):
    """Regression: an earlier version used ceil() and overshot t_max."""
    x0 = np.array([veh.radius + 1e3, 0.0, 0.0, 0.0, 700.0])
    traj = propagate(x0, lambda t, x: (0.0, 0.0), veh, dt=7.0, t_max=100.0)
    assert traj.t[-1] <= 100.0 + 1e-9


def test_rk4_is_fourth_order(veh, arc):
    r0 = arc.radius + 100e3
    x0 = np.array([r0, 0.0, -400.0, 1300.0 / r0, 900.0])
    T = 600.0
    ref = x0.copy()
    for _ in range(int(T / 0.01)):
        ref = rk4_step(ref, 0.0, 0.0, 0.01, veh)

    def err(dt):
        tr = propagate(x0, lambda t, x: (0.0, 0.0), veh, dt=dt, t_max=T)
        return float(np.linalg.norm(tr.x[-1, :3] - ref[:3]))

    # step sizes divide T exactly and keep the error above roundoff
    order = math.log2(err(20.0) / err(10.0))
    assert 3.7 < order < 4.3


def test_centrifugal_relief_is_significant_at_ignition(veh, arc):
    """Holding altitude is nearly free early in the burn -- worth asserting,
    because it is why the optimal steering is almost horizontal."""
    st = arc.state_at_altitude(60e3)
    r = st["radius"]
    loc = to_local(np.array([r, 0.0, st["v_vertical"], st["v_horizontal"] / r, 880.0]), veh)
    assert 0.3 < loc["g_effective"] < 0.9     # vs 1.52 m/s^2 of gravity alone
