"""Rerun logging for planar descent trajectories.

Two spatial views, because the problem spans six orders of magnitude: the
trajectory is ~1000 km long and the vehicle is 2 m across.

* ``world/``   overview in kilometres -- trajectory, surface, vehicle marker,
  thrust vector.
* ``vehicle/`` the hopper itself in metres, built from the SIM-G* geometry
  rows and **tilted by the commanded thrust angle**.  In a 3-DOF model the
  thrust direction *is* the body axis, so this view shows directly how far
  from vertical the vehicle flies -- the SIM-A16 finding (~68 deg during the
  braking burn, rotating upright only at the end).

Recordings are written as ``.rrd`` files rather than streamed.  That is the
transport that survives an SSH/dev tunnel with no port forwarding, and it
makes a run replayable and archivable -- a specific Monte Carlo failure case
can be kept and reopened.  To watch live instead, see ``scripts/run_descent.py
--serve``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rerun as rr

from phenei_gym.dynamics import PlanarVehicle, Trajectory, to_local
from phenei_gym.params import PARAMS

TIMELINE = "flight_time"


def _hexagon(radius: float, z: float, n: int = 6) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n + 1, z)])


def hopper_wireframe() -> dict[str, np.ndarray]:
    """Wireframe of the hopper in body axes (+Z up), from the SIM-G* rows.

    Origin is at the surface contact plane, so z is height above the ground
    when the vehicle is upright.
    """
    r_lower = PARAMS.value("SIM-G01") / 2.0
    r_upper = PARAMS.value("SIM-G02") / 2.0
    z_lower = PARAMS.value("SIM-G03")
    z_upper = PARAMS.value("SIM-G04")
    r_eng = PARAMS.value("SIM-G08")
    d_noz = PARAMS.value("SIM-G13")
    r_foot = PARAMS.value("SIM-G18")
    n_eng = int(PARAMS.value("SIM-G07"))

    strips: dict[str, np.ndarray] = {
        "deck_lower": _hexagon(r_lower, z_lower),
        "deck_upper": _hexagon(r_upper, z_upper),
    }
    # truss between the decks
    for i, a in enumerate(np.linspace(0.0, 2.0 * np.pi, 7)[:6]):
        strips[f"truss_{i}"] = np.array(
            [[r_lower * math.cos(a), r_lower * math.sin(a), z_lower],
             [r_upper * math.cos(a), r_upper * math.sin(a), z_upper]]
        )
    # engine nozzles, hanging below the lower deck
    for i in range(n_eng):
        a = 2.0 * math.pi * i / n_eng
        cx, cy = r_eng * math.cos(a), r_eng * math.sin(a)
        strips[f"nozzle_{i}"] = np.array(
            [[cx, cy, z_lower], [cx, cy, z_lower - 0.40],
             [cx + d_noz / 2 * math.cos(a), cy + d_noz / 2 * math.sin(a), z_lower - 0.40]]
        )
    # landing legs
    for i in range(4):
        a = 2.0 * math.pi * i / 4 + math.pi / 4
        strips[f"leg_{i}"] = np.array(
            [[r_lower * 0.8 * math.cos(a), r_lower * 0.8 * math.sin(a), z_lower],
             [r_foot * math.cos(a), r_foot * math.sin(a), 0.0]]
        )
    return strips


def _rotation_about_y(angle: float) -> np.ndarray:
    """Tilt the body +Z axis away from local vertical by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def log_trajectory(
    traj: Trajectory,
    veh: PlanarVehicle,
    out_path: Path | str | None = None,
    *,
    app_id: str = "phenei_descent",
    stride: int = 1,
    serve: bool = False,
) -> Path | None:
    """Log one planar descent to a .rrd file (or serve it live)."""
    rr.init(app_id, spawn=False)
    if serve:
        uri = rr.serve_grpc()
        rr.serve_web_viewer(connect_to=uri, open_browser=False)
        print(f"  rerun gRPC: {uri}\n  web viewer on port 9090 "
              f"(forward it: ssh -L 9090:localhost:9090 ...)")

    loc = traj.local(veh)
    # Static scene: surface line and the landing-accuracy requirement.
    dr_km = loc["downrange"] / 1e3
    rr.log("world/surface",
           rr.LineStrips3D([np.array([[dr_km.min() - 5, 0.0, 0.0], [max(dr_km.max() + 5, 5.0), 0.0, 0.0]])],
                           colors=[[120, 110, 100]], radii=[0.4]),
           static=True)
    tol = PARAMS.value("SIM-T01") / 1e3  # 50 m requirement, in km
    rr.log("world/target",
           rr.LineStrips3D([np.array([[-tol, 0.0, 0.0], [tol, 0.0, 0.0]])],
                           colors=[[230, 170, 60]], radii=[0.8]),
           static=True)
    rr.log("world/path", rr.LineStrips3D(
        [np.column_stack([dr_km, np.zeros(len(dr_km)), loc["altitude"] / 1e3])],
        colors=[[90, 160, 230]], radii=[0.25]), static=True)
    wire = hopper_wireframe()
    rr.log("vehicle/body",
           rr.LineStrips3D(list(wire.values()), colors=[[200, 205, 215]] * len(wire), radii=[0.012]),
           static=True)
    rr.log("vehicle/ground",
           rr.LineStrips3D([np.array([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])],
                           colors=[[120, 110, 100]], radii=[0.02]),
           static=True)

    thrust_max = veh.thrust_max
    for i in range(0, len(traj), stride):
        rr.set_time(TIMELINE, duration=float(traj.t[i]))
        x, z = float(dr_km[i]), float(loc["altitude"][i] / 1e3)

        rr.log("world/hopper", rr.Points3D([[x, 0.0, z]], colors=[[240, 240, 240]], radii=[1.2]))

        beta = float(traj.beta[i])
        thrust = float(traj.thrust[i])
        # thrust direction in the local (downrange, up) plane
        tx, tz = math.cos(beta), math.sin(beta)
        scale = 30.0 * thrust / thrust_max
        rr.log("world/thrust", rr.Arrows3D(
            origins=[[x, 0.0, z]], vectors=[[-tx * scale, 0.0, -tz * scale]],
            colors=[[255, 120, 60]]))

        # Body axis is along +thrust; tilt from local vertical is (pi/2 - beta).
        tilt = beta - math.pi / 2.0
        rr.log("vehicle", rr.Transform3D(mat3x3=_rotation_about_y(tilt)))

        rr.log("telemetry/altitude_km", rr.Scalars(loc["altitude"][i] / 1e3))
        rr.log("telemetry/speed", rr.Scalars(loc["speed"][i]))
        rr.log("telemetry/v_vertical", rr.Scalars(loc["v_vertical"][i]))
        rr.log("telemetry/v_horizontal", rr.Scalars(loc["v_horizontal"][i]))
        rr.log("telemetry/thrust_N", rr.Scalars(thrust))
        rr.log("telemetry/throttle_pct", rr.Scalars(100.0 * thrust / thrust_max))
        rr.log("telemetry/mass_kg", rr.Scalars(loc["mass"][i]))
        rr.log("telemetry/tilt_from_vertical_deg", rr.Scalars(math.degrees(tilt)))
        rr.log("telemetry/beta_deg", rr.Scalars(math.degrees(beta)))

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rr.save(out)
        return out
    return None
