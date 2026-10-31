"""Thruster geometry and control allocation.

Everything here is built from the ``SIM-G*`` rows, so moving a thruster is a
CSV edit rather than a code change.  That is deliberate: the control
effectiveness matrix ``B`` is the object the Phase 6 placement studies sweep
(``TR-SIM-17``).

Body frame: ``+Z`` up, opposite the main-engine thrust.  Origin at the vehicle
centre of mass.

Layout as modelled (all ``Assumption (LLM)`` unless noted):

* 4 main engines (``SIM-G07``, given by Ilya) at radius ``SIM-G08``, azimuths
  0/90/180/270 deg, fixed, thrusting along body ``+Z``.
* 8 cold-gas RCS (``SIM-G14``) in 4 pods at azimuths 45/135/225/315 deg,
  radius ``SIM-G15``, on the upper deck, each pod carrying two thrusters
  canted ``SIM-G16`` from vertical in the vertical-tangential plane.

The resulting wrench space is rank 4 -- ``{Fz, Mx, My, Mz}``.  Lateral force
is *not* directly producible: like a quadrotor, the vehicle translates by
tilting.  Four parallel main engines alone are rank 3 (no yaw), which is why
some RCS is structural to the design rather than optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from phenei_gym.params import PARAMS

# Wrench rows actually spanned by this layout. Fx, Fy are unreachable by
# construction, so carrying them would make B rank-deficient by definition and
# hide genuine rank losses (e.g. an engine-out case).
WRENCH_LABELS = ("Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class Thruster:
    name: str
    position: np.ndarray   # m, body frame, relative to CoM
    direction: np.ndarray  # unit vector of the FORCE applied to the vehicle
    thrust_max: float      # N
    throttle_min: float    # fraction of thrust_max; 0.0 for on/off devices
    isp: float             # s
    kind: str              # "main" | "rcs"

    def wrench(self) -> np.ndarray:
        """Column of B: (Fz, Mx, My, Mz) produced at full thrust."""
        force = self.thrust_max * self.direction
        torque = np.cross(self.position, force)
        return np.array([force[2], torque[0], torque[1], torque[2]])


@dataclass
class ActuatorLayout:
    thrusters: list[Thruster] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_params(cls, thrust_max_total: float | None = None) -> "ActuatorLayout":
        if thrust_max_total is None:
            thrust_max_total = PARAMS.range("SIM-P01")[1]
        n_main = int(PARAMS.value("SIM-G07"))
        r_eng = PARAMS.value("SIM-G08")
        z_com = PARAMS.value("SIM-G20")
        z_deck_lower = PARAMS.value("SIM-G03")
        z_deck_upper = PARAMS.value("SIM-G04")
        n_rcs = int(PARAMS.value("SIM-G14"))
        r_pod = PARAMS.value("SIM-G15")
        cant = math.radians(PARAMS.value("SIM-G16"))
        f_rcs = PARAMS.value("SIM-P09")
        isp_main = PARAMS.value("SIM-P03")
        isp_rcs = PARAMS.value("SIM-P10")
        thr_min = PARAMS.value("SIM-P04") / 100.0

        th: list[Thruster] = []
        for i in range(n_main):
            a = 2.0 * math.pi * i / n_main
            th.append(Thruster(
                name=f"main{i}",
                position=np.array([r_eng * math.cos(a), r_eng * math.sin(a), z_deck_lower - z_com]),
                direction=np.array([0.0, 0.0, 1.0]),
                thrust_max=thrust_max_total / n_main,
                throttle_min=thr_min,
                isp=isp_main,
                kind="main",
            ))

        n_pods = n_rcs // 2
        for k in range(n_pods):
            a = math.pi / n_pods * (2 * k + 1) / 2.0 + math.pi / 4.0
            pos = np.array([r_pod * math.cos(a), r_pod * math.sin(a), z_deck_upper - z_com])
            # Alternate the tangential sense pod to pod. With every pod canted
            # the same way the Mz row of B is all-positive, i.e. the vehicle can
            # only yaw in ONE direction -- a real defect that rank alone hides,
            # because throttles are non-negative so the reachable set is a cone,
            # not a subspace. Alternating pods restores bidirectional yaw.
            yaw_sense = 1.0 if k % 2 == 0 else -1.0
            tangent = yaw_sense * np.array([-math.sin(a), math.cos(a), 0.0])
            for sign, tag in ((+1.0, "up"), (-1.0, "dn")):
                d = math.cos(cant) * sign * np.array([0.0, 0.0, 1.0]) + math.sin(cant) * tangent
                th.append(Thruster(
                    name=f"rcs{k}{tag}",
                    position=pos,
                    direction=d / np.linalg.norm(d),
                    thrust_max=f_rcs,
                    throttle_min=0.0,  # on/off device, modelled continuous
                    isp=isp_rcs,
                    kind="rcs",
                ))
        return cls(th)

    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.thrusters)

    @property
    def main_idx(self) -> np.ndarray:
        return np.array([i for i, t in enumerate(self.thrusters) if t.kind == "main"])

    @property
    def rcs_idx(self) -> np.ndarray:
        return np.array([i for i, t in enumerate(self.thrusters) if t.kind == "rcs"])

    def effectiveness(self, active: np.ndarray | None = None) -> np.ndarray:
        """B, shape (4, n): wrench produced per unit throttle of each thruster."""
        cols = [t.wrench() for t in self.thrusters]
        B = np.column_stack(cols)
        if active is not None:
            B = B * np.asarray(active, dtype=float)[None, :]
        return B

    def rank(self, active: np.ndarray | None = None, tol: float = 1e-6) -> int:
        return int(np.linalg.matrix_rank(self.effectiveness(active), tol=tol))

    def condition(self, active: np.ndarray | None = None) -> float:
        s = np.linalg.svd(self.effectiveness(active), compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")

    def reachable(self, active: np.ndarray | None = None) -> dict[str, tuple[float, float]]:
        """(min, max) of each wrench component over the throttle box.

        Throttles are non-negative, so the reachable wrench set is a convex
        CONE, not the span of B. Checking rank is therefore not enough: a
        layout can be full rank yet unable to produce negative yaw. This is
        the diagnostic that catches that.
        """
        B = self.effectiveness(active)
        return {
            lbl: (float(np.sum(np.clip(B[i], None, 0.0))), float(np.sum(np.clip(B[i], 0.0, None))))
            for i, lbl in enumerate(WRENCH_LABELS)
        }

    def authority(self, mass: float, inertia: np.ndarray) -> dict[str, float]:
        """Peak angular acceleration about each body axis [rad/s^2]."""
        B = self.effectiveness()
        out = {}
        for j, axis in enumerate(("roll", "pitch", "yaw")):
            # best achievable torque with all throttles in [0, 1]
            out[axis] = float(np.sum(np.clip(B[j + 1], 0.0, None)) / inertia[j])
        out["max_thrust"] = float(np.sum(np.clip(B[0], 0.0, None)))
        out["max_accel_z"] = out["max_thrust"] / mass
        return out

    # ------------------------------------------------------------------
    def allocate(
        self,
        wrench_des: np.ndarray,
        active: np.ndarray | None = None,
        iters: int = 12,
    ) -> np.ndarray:
        """Throttle commands in [0, 1] approximating ``wrench_des``.

        Damped least squares with iterative clipping: solve, clip to bounds,
        re-solve for the residual using only the thrusters still free.  Cheap,
        deterministic and good enough; a proper NNLS/LP is a later refinement
        if the placement studies need exactness.
        """
        B = self.effectiveness(active)
        n = B.shape[1]
        lo = np.zeros(n)
        hi = np.ones(n) if active is None else np.asarray(active, dtype=float)

        u = np.zeros(n)
        free = hi > 0.0
        residual = np.asarray(wrench_des, dtype=float).copy()
        for _ in range(iters):
            if not free.any():
                break
            Bf = B[:, free]
            # damped pseudo-inverse keeps it well behaved when B is rank-deficient
            du = Bf.T @ np.linalg.solve(Bf @ Bf.T + 1e-9 * np.eye(B.shape[0]), residual)
            trial = u.copy()
            trial[free] += du
            clipped = np.clip(trial, lo, hi)
            if np.allclose(clipped, trial, atol=1e-12):
                u = clipped
                break
            newly_fixed = free & (np.abs(clipped - trial) > 1e-12)
            u = clipped
            residual = np.asarray(wrench_des, dtype=float) - B @ u
            free = free & ~newly_fixed
        return np.clip(u, lo, hi)

    def wrench_of(self, u: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        return self.effectiveness(active) @ np.asarray(u, dtype=float)

    def mass_flow(self, u: np.ndarray) -> float:
        """kg/s at the given throttles, accounting for the two Isp classes."""
        g0 = 9.80665
        return float(sum(
            ui * t.thrust_max / (t.isp * g0) for ui, t in zip(u, self.thrusters)
        ))
