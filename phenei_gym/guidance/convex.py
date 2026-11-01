"""Convex powered-descent guidance (TR-SIM-11).

Why this exists
---------------
The reference law in :mod:`phenei_gym.guidance.descent` tracks *velocity
profiles* and never looks at where the landing site is.  That is fine for a
nominal reference trajectory and fatal under dispersion: an injection velocity
error integrates unattenuated over a 222 s burn, giving a 3251 m CEP against a
50 m requirement (``SIM-MC15``, ``TR-SIM-23``).  Two attempts to bolt position
feedback onto it both failed for the same structural reason (``TR-SIM-20``):
with the descent timeline fixed by a vertical profile, horizontal range and
horizontal velocity are not independently controllable, so the position demand
either exceeds the thrust bound or fights the vertical channel.

The fix is to stop fixing the timeline.  Here the time of flight, the thrust
magnitude history and the thrust direction history are all decision variables
of one optimisation whose *constraints* are the thrust bounds -- so a solution
is feasible by construction, and the landing site is a terminal constraint
rather than a hope.

Formulation
-----------
Lossless convexification, after Acikmese & Ploen (JGCD 2007) and Blackmore,
Acikmese & Scharf (JGCD 2010).  The non-convexity in powered descent is the
*lower* thrust bound ``||T|| >= rho1`` (an engine that cannot be throttled
below 30 %, ``SIM-P04``), which carves a hole out of the feasible set.  The
trick is to change variables to

    u = T / m        (thrust acceleration)        sigma >= ||u||

and optimise over ``(u, sigma)`` with ``sigma`` as an independent slack.  The
annulus ``rho1 <= ||T|| <= rho2`` becomes a pair of bounds on ``sigma``, which
is convex, and the relaxation is *lossless*: the optimum always has
``sigma = ||u||``, so the answer solves the original problem.  Mass enters
through ``z = ln m``, which makes ``zdot = -alpha*sigma`` linear, and the
``sigma`` bounds are expanded about a reference mass profile -- the lower one
to second order (still convex), the upper one to first order.

Successive convexification
--------------------------
One term is genuinely nonlinear here and is *not* in the textbook problem:
gravity.  The braking burn covers ~60 km of altitude and ~100 km of downrange
on a 1737 km body, and the centrifugal relief at 1359 m/s is 1.06 m/s^2
against 1.62 m/s^2 of gravity -- so a constant gravity vector is wrong by a
large fraction of the thing being modelled.  Instead gravity is held as a
*known time series* taken from the previous iterate's trajectory and the solve
is repeated (``SIM-CVX04`` iterations).  The frame is inertial, so there is no
centrifugal or Coriolis term to add: that relief appears by itself as the
curvature of the trajectory.

Two-phase fallback
------------------
Minimum fuel with a hard terminal position constraint is infeasible whenever
the vehicle cannot reach the site at all -- which is exactly the dispersed case
this guidance exists to handle.  So the solve is the standard two phases: try
minimum fuel to the site; if that is infeasible, minimise the terminal
position error instead and accept landing short.  A guidance law that returns
"as close as I can get" degrades; one that returns "infeasible" does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from phenei_gym.params import PARAMS

G0 = 9.80665


@dataclass
class ConvexPlan:
    """A solved trajectory, sampled on the solver's own time grid."""

    t: np.ndarray            # (N+1,)   time from the solve epoch   [s]
    r: np.ndarray            # (N+1, 3) position in the site frame  [m]
    v: np.ndarray            # (N+1, 3) velocity                    [m/s]
    u: np.ndarray            # (N+1, 3) thrust acceleration         [m/s^2]
    m: np.ndarray            # (N+1,)   mass                        [kg]
    tf: float                # time of flight                       [s]
    propellant: float        # propellant the plan spends           [kg]
    miss: float              # terminal horizontal error            [m]
    phase: str               # "min-fuel" or "min-error"

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(r, v, u)`` interpolated at time ``t`` from the solve epoch."""
        tc = float(np.clip(t, self.t[0], self.t[-1]))
        out = []
        for arr in (self.r, self.v, self.u):
            out.append(np.array([np.interp(tc, self.t, arr[:, i]) for i in range(3)]))
        return out[0], out[1], out[2]


@dataclass
class ConvexDescentGuidance:
    """Solve the powered-descent SOCP, and re-solve it in flight.

    ``radius``/``mu`` describe the body; ``rho1``/``rho2`` are the thrust
    bounds the *plan* is allowed to use, which are deliberately not the engine
    bounds: ``SIM-GD07`` reserves throttle headroom for differential-throttle
    attitude control, and a plan that commands 100 % leaves the attitude loop
    with no authority at all.
    """

    radius: float
    mu: float
    rho1: float                      # minimum plannable thrust   [N]
    rho2: float                      # maximum plannable thrust   [N]
    isp: float                       # [s]
    mass_dry: float                  # [kg]
    v_gate: float                    # descent rate the plan DELIVERS at the gate   [m/s]
    h_terminal: float                # plan terminates at this altitude             [m]
    n_nodes: int = 50                # SIM-CVX01
    scvx_iters: int = 3              # SIM-CVX04
    glideslope_deg: float = 4.0      # SIM-CVX05
    solver: str = "CLARABEL"
    _last: ConvexPlan | None = field(default=None, repr=False)

    @classmethod
    def from_params(cls, thrust_max: float = 6000.0, thrust_cap: float = 0.90,
                    v_gate: float | None = None, **kw) -> "ConvexDescentGuidance":
        """``v_gate`` defaults to the reference law's own profile at the gate.

        This matters more than it looks. The convex plan hands the vehicle to
        the terminal-descent law at ``SIM-A12`` (50 m), and the two laws have
        to agree about the state at that point. Delivering the vehicle there at
        the commanded touchdown rate instead (1 m/s) looks tidier and is a
        trap: the handover triggers only BELOW 50 m, so a vehicle arriving at
        50 m descending at 1 m/s never crosses, and the guidance -- whose
        terminal constraint it has by then satisfied -- holds it there. Measured
        cost of that bug: 15 s of hover at the gate, 13 kg of propellant, and
        the tanks dry 5 m above the ground.
        """
        if v_gate is None:
            a_dec = PARAMS.value("SIM-GD01")
            v_gate = math.sqrt(2.0 * a_dec * PARAMS.value("SIM-A12")
                               + PARAMS.value("SIM-GD04") ** 2)
        return cls(
            radius=PARAMS.value("SIM-E03") * 1e3,
            mu=PARAMS.value("SIM-E01") * (PARAMS.value("SIM-E03") * 1e3) ** 2,
            rho1=PARAMS.value("SIM-P04") / 100.0 * thrust_max,
            rho2=thrust_cap * thrust_max,
            isp=PARAMS.value("SIM-P03"),
            mass_dry=PARAMS.value("SIM-M01"),
            v_gate=v_gate,
            h_terminal=PARAMS.value("SIM-A12"),
            n_nodes=int(PARAMS.value("SIM-CVX01")),
            scvx_iters=int(PARAMS.value("SIM-CVX04")),
            glideslope_deg=PARAMS.value("SIM-CVX05"),
            **kw,
        )

    # ------------------------------------------------------------------
    def _gravity(self, r: np.ndarray) -> np.ndarray:
        """Spherical point-mass gravity in the site frame, same as the env."""
        rv = r + np.array([0.0, 0.0, self.radius])
        n = np.linalg.norm(rv, axis=-1, keepdims=True)
        return -self.mu * rv / n**3

    def _ballistic_guess(self, r0, v0, tf, n) -> np.ndarray:
        """Positions along a coasting arc, to seed the gravity profile."""
        t = np.linspace(0.0, tf, n)
        g0 = self._gravity(r0)
        return r0 + np.outer(t, v0) + 0.5 * np.outer(t**2, g0)

    # ------------------------------------------------------------------
    def _solve_once(self, r0, v0, m0, tf, g_prof, z_ref, phase, miss_cap=None):
        """One convex solve on a frozen gravity and mass reference."""
        n = self.n_nodes
        dt = tf / n
        alpha = 1.0 / (self.isp * G0)

        r = cp.Variable((n + 1, 3))
        v = cp.Variable((n + 1, 3))
        z = cp.Variable(n + 1)
        u = cp.Variable((n + 1, 3))
        sg = cp.Variable(n + 1)

        a = u + g_prof                                   # total acceleration
        cons = [r[0] == r0, v[0] == v0, z[0] == math.log(m0)]
        # Exact propagation for an acceleration that is linear between nodes.
        cons += [v[1:] == v[:-1] + dt / 2.0 * (a[:-1] + a[1:])]
        cons += [r[1:] == r[:-1] + dt * v[:-1]
                 + dt**2 / 6.0 * (2.0 * a[:-1] + a[1:])]
        cons += [z[1:] == z[:-1] - alpha * dt / 2.0 * (sg[:-1] + sg[1:])]

        # Lossless convexification of the thrust annulus.
        cons += [sg >= cp.norm(u, axis=1)]
        dz = z - z_ref
        mu1 = self.rho1 * np.exp(-z_ref)
        mu2 = self.rho2 * np.exp(-z_ref)
        cons += [sg >= cp.multiply(mu1, 1.0 - dz + 0.5 * cp.square(dz))]
        cons += [sg <= cp.multiply(mu2, 1.0 - dz)]
        cons += [z >= math.log(self.mass_dry)]

        # Stay above the surface, and inside a glideslope cone near the site so
        # the plan cannot approach horizontally through the ground.
        cons += [r[:, 2] >= 0.0]
        tan_g = math.tan(math.radians(self.glideslope_deg))
        cons += [r[:, 2] >= tan_g * cp.norm(r[:, :2], axis=1)]

        # Terminal gate: over the site, descending at v_touchdown.
        cons += [v[n] == np.array([0.0, 0.0, -self.v_gate])]
        cons += [r[n, 2] == self.h_terminal]
        if phase == "min-fuel":
            cons += [r[n, :2] == 0.0]
            obj = cp.Maximize(z[n])
        else:
            if miss_cap is not None:
                cons += [cp.norm(r[n, :2]) <= miss_cap]
                obj = cp.Maximize(z[n])
            else:
                obj = cp.Minimize(cp.norm(r[n, :2]))

        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            return None
        if r.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        return r.value, v.value, z.value, u.value, sg.value

    # ------------------------------------------------------------------
    def plan(self, r0, v0, m0, tf) -> ConvexPlan | None:
        """Solve for a given time of flight, iterating on gravity and mass."""
        n = self.n_nodes
        t = np.linspace(0.0, tf, n + 1)
        r_guess = self._ballistic_guess(np.asarray(r0, float), np.asarray(v0, float),
                                        tf, n + 1)
        g_prof = self._gravity(r_guess)
        # Mass reference: constant-rate expenditure at a mid-throttle guess.
        mdot = 0.65 * self.rho2 / (self.isp * G0)
        z_ref = np.log(np.maximum(m0 - mdot * t, self.mass_dry * 1.001))

        sol = None
        phase = "min-fuel"
        for _ in range(self.scvx_iters):
            s = self._solve_once(r0, v0, m0, tf, g_prof, z_ref, "min-fuel")
            if s is None:
                break
            sol, phase = s, "min-fuel"
            g_prof = self._gravity(s[0])
            z_ref = s[2]
        if sol is None:
            # Minimum landing error, then minimum fuel subject to that error.
            for _ in range(self.scvx_iters):
                s = self._solve_once(r0, v0, m0, tf, g_prof, z_ref, "min-error")
                if s is None:
                    return None
                sol, phase = s, "min-error"
                g_prof = self._gravity(s[0])
                z_ref = s[2]

        rr, vv, zz, uu, _ = sol
        m = np.exp(zz)
        return ConvexPlan(t=t, r=rr, v=vv, u=uu, m=m, tf=tf,
                          propellant=float(m[0] - m[-1]),
                          miss=float(np.linalg.norm(rr[n, :2])), phase=phase)

    def solve(self, r0, v0, m0, tf_hint: float | None = None,
              bracket: tuple[float, float] | None = None,
              n_tf: int = 7) -> ConvexPlan | None:
        """Plan, searching over the time of flight.

        Time of flight is the one parameter the convex problem cannot absorb --
        fixing it makes the problem convex, so it has to be searched outside.
        A coarse grid then one bisection refinement is enough: the fuel-vs-tf
        curve is smooth and shallow near its optimum.
        """
        r0 = np.asarray(r0, float); v0 = np.asarray(v0, float)
        if bracket is None:
            if tf_hint is not None:
                bracket = (max(8.0, 0.6 * tf_hint), 1.6 * tf_hint)
            else:
                # Rough scale: the time to null the speed at mean thrust.
                a = 0.75 * self.rho2 / m0
                tg = float(np.linalg.norm(v0)) / max(a, 0.1)
                bracket = (max(8.0, 0.5 * tg), 2.0 * tg)
        lo, hi = bracket
        best: ConvexPlan | None = None
        grid = np.linspace(lo, hi, n_tf)
        for tf in grid:
            p = self.plan(r0, v0, m0, float(tf))
            if p is None:
                continue
            if best is None or self._better(p, best):
                best = p
        if best is None:
            return None
        # One refinement pass around the winner.
        step = (hi - lo) / (n_tf - 1)
        for tf in (best.tf - step / 2.0, best.tf + step / 2.0):
            if tf <= 1.0:
                continue
            p = self.plan(r0, v0, m0, float(tf))
            if p is not None and self._better(p, best):
                best = p
        self._last = best
        return best

    @staticmethod
    def _better(a: ConvexPlan, b: ConvexPlan) -> bool:
        """Feasibility first, then landing error, then fuel."""
        if (a.phase == "min-fuel") != (b.phase == "min-fuel"):
            return a.phase == "min-fuel"
        if a.phase == "min-error" and abs(a.miss - b.miss) > 1.0:
            return a.miss < b.miss
        return a.propellant < b.propellant
