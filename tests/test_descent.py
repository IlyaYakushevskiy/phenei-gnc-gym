"""The reference descent must land inside the touchdown box, and the MCR's
39 km ignition altitude must be shown to be infeasible (SIM-A01 / TR-SIM-18)."""
import math

import numpy as np
import pytest

from phenei_gym.dynamics import IDX_M, PlanarVehicle, propagate, to_local
from phenei_gym.guidance import min_energy_arc
from phenei_gym.guidance.descent import DescentGuidance, touchdown_stop
from phenei_gym.params import PARAMS


def _fly(thrust_max: float, altitude_km: float, mass: float = 880.0, dt: float = 0.02):
    veh = PlanarVehicle.from_params(thrust_max=thrust_max)
    arc = min_energy_arc()
    h0 = altitude_km * 1e3
    st = arc.state_at_altitude(h0)
    g = DescentGuidance.from_params(veh, n_engines_terminal=2)
    g.a_dec = st["v_vertical"] ** 2 / (2.0 * h0)
    r0 = st["radius"]
    x0 = np.array([r0, 0.0, st["v_vertical"], st["v_horizontal"] / r0, mass])
    traj = propagate(x0, g, veh, dt=dt, t_max=900.0, stop=touchdown_stop(veh))
    return veh, traj, to_local(traj.x[-1], veh)


@pytest.fixture(scope="module")
def reference():
    return _fly(6000.0, PARAMS.value("SIM-A19"))


def test_reference_descent_touches_down(reference):
    _, traj, _ = reference
    assert traj.reason == "touchdown"


def test_reference_descent_meets_touchdown_limits(reference):
    _, _, final = reference
    assert abs(final["v_vertical"]) <= PARAMS.value("SIM-T02")
    assert abs(final["v_horizontal"]) <= PARAMS.value("SIM-T03")


def test_reference_descent_propellant_within_load(reference):
    _, traj, _ = reference
    used = 880.0 - traj.x[-1, IDX_M]
    assert used <= PARAMS.value("SIM-M04")
    assert used == pytest.approx(PARAMS.value("SIM-GD06"), rel=0.02)


def test_mcr_39km_ignition_is_infeasible():
    """TR-SIM-18: at 39 km the arc arrives descending at 557 m/s, leaving far
    too little time to remove 1376 m/s of horizontal velocity at 6 kN."""
    _, traj, final = _fly(6000.0, 39.0, dt=0.05)
    assert abs(final["v_horizontal"]) > 10.0 * PARAMS.value("SIM-T03")


def test_vehicle_flies_far_from_vertical_during_braking(reference):
    """SIM-A16: a body-fixed nadir sensor is mispointed for most of the burn."""
    _, traj, _ = reference
    tilt = np.abs(np.degrees(traj.beta - math.pi / 2.0))
    assert (tilt > 60.0).sum() * 0.02 > 0.5 * traj.t[-1]
