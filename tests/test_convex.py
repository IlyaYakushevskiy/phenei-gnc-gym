"""Convex powered-descent guidance (TR-SIM-11, SIM-CVX01-07).

These check the two things that make the formulation worth trusting: that the
returned plan really satisfies the dynamics and the thrust bounds it claims to
(so "feasible by construction" is not just a claim about the maths), and that
the lossless-convexification relaxation is actually tight at the optimum.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phenei_gym.guidance.convex import G0, ConvexDescentGuidance
from phenei_gym.params import PARAMS

# A representative braking-burn ignition state in the landing-site frame,
# taken from the 6 kN / 60 km baseline (see CLAUDE.md "Approach conditions").
R0 = np.array([-173550.0, 0.0, 60000.0])
V0 = np.array([1359.4, 0.0, -537.8])
M0 = 880.0


@pytest.fixture(scope="module")
def guid():
    return ConvexDescentGuidance.from_params(thrust_max=6000.0, thrust_cap=0.90)


@pytest.fixture(scope="module")
def plan(guid):
    p = guid.plan(R0, V0, M0, 240.0)
    assert p is not None, "the nominal ignition state must be feasible"
    return p


def test_plan_is_feasible_min_fuel(plan):
    assert plan.phase == "min-fuel"
    assert plan.miss < 1e-3


def test_terminal_constraints_are_met(plan, guid):
    assert np.allclose(plan.r[-1, :2], 0.0, atol=1e-3)
    assert plan.r[-1, 2] == pytest.approx(guid.h_terminal, abs=1e-3)
    assert np.allclose(plan.v[-1], [0.0, 0.0, -guid.v_gate], atol=1e-3)


def test_initial_conditions_are_respected(plan):
    assert np.allclose(plan.r[0], R0, atol=1e-6)
    assert np.allclose(plan.v[0], V0, atol=1e-6)
    assert plan.m[0] == pytest.approx(M0, rel=1e-9)


def test_thrust_stays_inside_the_engine_annulus(plan, guid):
    thrust = plan.m * np.linalg.norm(plan.u, axis=1)
    # The lower bound is the non-convex one -- the whole reason for the
    # change of variables -- so it is the interesting half of this check.
    assert thrust.min() >= guid.rho1 - 1.0
    assert thrust.max() <= guid.rho2 + 1.0


def test_relaxation_is_tight(guid):
    """sigma == ||u|| at the optimum: the convexification is lossless.

    If this ever fails the returned plan is NOT a solution of the original
    problem -- the solver would have found it cheaper to inflate the slack
    than to spend thrust, which means the objective no longer penalises
    ``sigma`` the way the proof requires.
    """
    n = guid.n_nodes
    tf = 240.0
    t = np.linspace(0.0, tf, n + 1)
    r_guess = guid._ballistic_guess(R0, V0, tf, n + 1)
    g_prof = guid._gravity(r_guess)
    z_ref = np.log(np.maximum(M0 - 0.65 * guid.rho2 / (guid.isp * G0) * t,
                              guid.mass_dry * 1.001))
    sol = guid._solve_once(R0, V0, M0, tf, g_prof, z_ref, "min-fuel")
    assert sol is not None
    _, _, _, u, sigma = sol
    assert np.allclose(sigma, np.linalg.norm(u, axis=1), atol=1e-4)


def test_plan_satisfies_its_own_dynamics(plan, guid):
    """Re-propagate the planned acceleration and compare with the plan.

    The solver enforces a discrete integration rule; this checks that rule is
    the one the vehicle is actually flown with, i.e. that the plan is not
    self-consistent nonsense. Gravity is recomputed from the planned positions
    rather than reused, so a sign or frame error in ``_gravity`` shows up here.
    """
    dt = plan.tf / guid.n_nodes
    a = plan.u + guid._gravity(plan.r)
    v = plan.v[:-1] + dt / 2.0 * (a[:-1] + a[1:])
    r = plan.r[:-1] + dt * plan.v[:-1] + dt**2 / 6.0 * (2.0 * a[:-1] + a[1:])
    assert np.abs(v - plan.v[1:]).max() < 1e-3
    assert np.abs(r - plan.r[1:]).max() < 1.0


def test_mass_depletion_matches_the_thrust_history(plan, guid):
    dt = plan.tf / guid.n_nodes
    sigma = np.linalg.norm(plan.u, axis=1)
    alpha = 1.0 / (guid.isp * G0)
    z = np.log(plan.m)
    pred = z[:-1] - alpha * dt / 2.0 * (sigma[:-1] + sigma[1:])
    assert np.abs(pred - z[1:]).max() < 1e-6


def test_trajectory_stays_above_the_surface_and_inside_the_glideslope(plan, guid):
    assert plan.r[:, 2].min() >= -1e-6
    tan_g = math.tan(math.radians(guid.glideslope_deg))
    assert (plan.r[:, 2] + 1e-6 >= tan_g * np.linalg.norm(plan.r[:, :2], axis=1)).all()


def test_propellant_is_consistent_with_the_mass_profile(plan):
    assert plan.propellant == pytest.approx(plan.m[0] - plan.m[-1], rel=1e-9)
    assert 300.0 < plan.propellant < PARAMS.value("SIM-M03") - PARAMS.value("SIM-M01")


def test_too_short_a_time_of_flight_is_infeasible(guid):
    """Not a defect: the vehicle genuinely cannot stop that fast.

    A guidance law that returned a plan here would be returning one that
    violates the thrust bound, which is exactly the failure TR-SIM-20 records.
    """
    assert guid.plan(R0, V0, M0, 120.0) is None


def test_an_unreachable_site_returns_nothing_rather_than_a_bad_plan(guid):
    """Current behaviour, and a known gap -- see the xfail below.

    Started low and fast the site cannot be reached inside the thrust bound,
    and no time of flight in the searched bracket is feasible even with the
    terminal position relaxed, so ``solve`` returns ``None``. That is safe --
    the caller falls back to the reference law and counts it -- but it is not
    the graceful degradation the two-phase formulation is supposed to give.
    """
    p = guid.solve(np.array([-30000.0, 0.0, 3000.0]),
                   np.array([900.0, 0.0, -60.0]), M0, tf_hint=90.0)
    assert p is None


@pytest.mark.xfail(reason="the min-error phase relaxes only terminal POSITION, so a state "
                          "that cannot make the soft-landing velocity in any searched time "
                          "of flight still comes back infeasible. Fixing it means relaxing "
                          "the terminal velocity too, and deciding what a guidance law should "
                          "promise when it cannot land softly at all -- a design question, "
                          "not a bug to patch quietly.",
                   strict=True)
def test_unreachable_site_should_degrade_to_minimum_landing_error(guid):
    p = guid.solve(np.array([-30000.0, 0.0, 3000.0]),
                   np.array([900.0, 0.0, -60.0]), M0, tf_hint=90.0)
    assert p is not None and p.phase == "min-error"


def test_time_of_flight_search_finds_the_feasibility_boundary(guid):
    """Minimum fuel sits at the short end of the feasible window.

    Worth asserting because it is the opposite of the usual intuition that a
    slower descent is cheaper: here a longer flight means more time spent
    holding the vehicle up against gravity, so propellant rises monotonically
    with time of flight and the optimum is pinned against infeasibility.
    """
    best = guid.solve(R0, V0, M0, tf_hint=222.0, n_tf=5)
    assert best is not None and best.phase == "min-fuel"
    longer = guid.plan(R0, V0, M0, best.tf + 40.0)
    assert longer is not None
    assert longer.propellant > best.propellant
