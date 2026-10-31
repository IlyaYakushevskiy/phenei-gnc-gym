"""Rerun logging for the 6-DOF descent: terrain, nozzles, and live plumes.

Everything is logged in **metres in one world frame** (unlike the 3-DOF
logger, which split km and m). The trajectory is ~130 km and the vehicle is
2 m, which float32 handles, and a single frame means the terrain, the vehicle
and the plumes all coexist so you can zoom from the whole descent down to the
landing site without switching views.

Plumes are driven by the actual per-engine throttle out of the control
allocator, so the visualisation *is* telemetry: during a pitch manoeuvre you
can watch one engine brighten as its opposite dims. Cold-gas RCS is drawn as
discrete puffs rather than a proportional jet, because ``SIM-P11`` gives a
20 ms minimum on-time against a 10 ms control period -- it physically cannot
modulate within a cycle.

Note MON-3/MMH burns almost invisibly in reality; a visible flame is a
deliberate readability choice and should be stated in the video.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rerun as rr

from phenei_gym.dynamics.rigid6dof import I_MASS, I_OMEGA, I_POS, I_QUAT, I_VEL, quat_to_matrix
from phenei_gym.params import PARAMS
from phenei_gym.terrain import Terrain
from phenei_gym.viz.rerun_log import hopper_wireframe

TIMELINE = "flight_time"


def _cone(apex: np.ndarray, axis: np.ndarray, length: float, radius: float, segments: int = 14):
    """Cone mesh with its apex at ``apex``, opening along ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    base = apex + axis * length
    ring = [base + radius * (math.cos(t) * u + math.sin(t) * v)
            for t in np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)]
    verts = np.vstack([apex, np.array(ring)])
    tris = np.array([[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)], dtype=np.uint32)
    return verts, tris


def _plume_mesh(apex, axis, length, radius, hot, cool):
    """Cone with a hot core at the nozzle fading to ``cool`` at the tip."""
    verts, tris = _cone(apex, axis, length, radius)
    cols = np.vstack([np.array(hot, dtype=np.uint8),
                      np.tile(np.array(cool, dtype=np.uint8), (len(verts) - 1, 1))])
    return verts, tris, cols


def log_descent6dof(
    result,
    terrain: Terrain,
    out_path: Path | str,
    *,
    app_id: str = "phenei_descent_6dof",
    stride: int | None = None,
    serve: bool = False,
) -> Path:
    rr.init(app_id, spawn=False)
    if serve:
        uri = rr.serve_grpc()
        rr.serve_web_viewer(connect_to=uri, open_browser=False)

    layout = result.layout
    state, thr = result.state, result.throttles
    if stride is None:
        stride = max(1, len(result.t) // 1200)

    # ---------------- static scene ----------------
    verts, tris, cols = terrain.mesh()
    rr.log("world/terrain", rr.Mesh3D(vertex_positions=verts, triangle_indices=tris,
                                      vertex_colors=cols), static=True)
    rr.log("world/trajectory", rr.LineStrips3D([state[:, I_POS]], colors=[[95, 165, 235]],
                                               radii=[18.0]), static=True)
    ring_r = PARAMS.value("SIM-T01")
    a = np.linspace(0, 2 * np.pi, 97)
    ring = np.column_stack([ring_r * np.cos(a), ring_r * np.sin(a),
                            terrain.height_at(ring_r * np.cos(a), ring_r * np.sin(a)) + 0.3])
    rr.log("world/target", rr.LineStrips3D([ring], colors=[[245, 180, 60]], radii=[0.6]),
           static=True)

    wire = hopper_wireframe()
    rr.log("vehicle/body", rr.LineStrips3D(list(wire.values()),
                                           colors=[[205, 210, 220]] * len(wire), radii=[0.015]),
           static=True)

    # nozzle bells and RCS pods as real geometry, from the SIM-G* rows
    d_noz = PARAMS.value("SIM-G13")
    z_low = PARAMS.value("SIM-G03")
    z_com = PARAMS.value("SIM-G20")
    mains = [layout.thrusters[i] for i in layout.main_idx]
    rcs = [layout.thrusters[i] for i in layout.rcs_idx]
    for k, t in enumerate(mains):
        apex = np.array([t.position[0], t.position[1], z_low])
        v, f = _cone(apex, np.array([0.0, 0.0, -1.0]), 0.40, d_noz / 2.0)
        rr.log(f"vehicle/nozzle{k}", rr.Mesh3D(vertex_positions=v, triangle_indices=f,
               vertex_colors=np.tile(np.array([120, 125, 135], np.uint8), (len(v), 1))),
               static=True)
    rr.log("vehicle/rcs_pods",
           rr.Points3D([t.position + np.array([0, 0, z_com]) for t in rcs],
                       colors=[[160, 170, 185]] * len(rcs), radii=[0.06]), static=True)

    # ---------------- per-frame ----------------
    radius = PARAMS.value("SIM-E03") * 1e3
    t_main_max = mains[0].thrust_max
    for i in range(0, len(result.t), stride):
        rr.set_time(TIMELINE, duration=float(result.t[i]))
        s = state[i]
        rot = quat_to_matrix(s[I_QUAT])
        rr.log("vehicle", rr.Transform3D(translation=s[I_POS], mat3x3=rot))

        u = thr[i]
        # main-engine plumes: length and width follow that engine's throttle
        for k, ti in enumerate(layout.main_idx):
            frac = float(u[ti])
            if frac <= 1e-3:
                rr.log(f"vehicle/plume{k}", rr.Clear(recursive=False))
                continue
            t = layout.thrusters[ti]
            apex = np.array([t.position[0], t.position[1], z_low - 0.40])
            v, f, c = _plume_mesh(apex, np.array([0.0, 0.0, -1.0]),
                                  0.6 + 5.0 * frac, (d_noz / 2.0) * (0.8 + 0.7 * frac),
                                  hot=[255, 250, 225], cool=[255, 120, 35])
            rr.log(f"vehicle/plume{k}", rr.Mesh3D(vertex_positions=v, triangle_indices=f,
                                                  vertex_colors=c))
        # cold-gas puffs: on/off, never proportional (SIM-P11)
        for k, ti in enumerate(layout.rcs_idx):
            on = float(u[ti]) > 0.05
            name = f"vehicle/rcs{k}"
            if not on:
                rr.log(name, rr.Clear(recursive=False))
                continue
            t = layout.thrusters[ti]
            apex = t.position + np.array([0.0, 0.0, z_com])
            v, f, c = _plume_mesh(apex, -t.direction, 0.55, 0.07,
                                  hot=[235, 245, 255], cool=[150, 190, 235])
            rr.log(name, rr.Mesh3D(vertex_positions=v, triangle_indices=f, vertex_colors=c))

        r_vec = s[I_POS] + np.array([0.0, 0.0, radius])
        up = r_vec / np.linalg.norm(r_vec)
        v_up = float(s[I_VEL] @ up)
        rr.log("telemetry/altitude_m", rr.Scalars(float(result.altitude[i])))
        rr.log("telemetry/v_vertical", rr.Scalars(v_up))
        rr.log("telemetry/v_horizontal", rr.Scalars(float(np.linalg.norm(s[I_VEL] - v_up * up))))
        rr.log("telemetry/mass_kg", rr.Scalars(float(s[I_MASS])))
        rr.log("telemetry/tilt_deg", rr.Scalars(math.degrees(float(result.tilt[i]))))
        rr.log("telemetry/tilt_cmd_deg", rr.Scalars(math.degrees(float(result.tilt_cmd[i]))))
        rr.log("telemetry/body_rate_deg_s",
               rr.Scalars(math.degrees(float(np.linalg.norm(s[I_OMEGA])))))
        for k, ti in enumerate(layout.main_idx):
            rr.log(f"telemetry/throttle_main{k}", rr.Scalars(100.0 * float(u[ti])))
        rr.log("telemetry/rcs_active", rr.Scalars(float((u[layout.rcs_idx] > 0.05).sum())))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(out)
    return out
