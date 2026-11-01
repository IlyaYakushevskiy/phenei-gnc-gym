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


def _blueprint():
    """Three panes, so the recording opens on something worth looking at.

    Rerun's automatic layout puts every entity in one 3D view and every scalar
    in one plot, which for this recording means a 175 km trajectory with an
    invisible vehicle on it and fourteen unrelated signals stacked on one
    axis. The panes below are the three questions actually being asked of the
    simulation: where is the vehicle going, what does the flight look like
    from alongside it, and what are the sensors and actuators doing.
    """
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Descent overview", origin="/",
                contents=["world/**", "marker/**", "chase_cam/**", "vehicle/**"]),
            rrb.Vertical(
                # Origin "chase_anchor" (translation only, no rotation) makes
                # this view ride with the vehicle while keeping world
                # orientation, so attitude changes are visible as the body
                # rotating rather than as the world swinging around it.
                #
                # Contents are deliberately ONLY the vehicle subtree. A 3D view
                # auto-fits its bounds to its contents, so including the world
                # would fit a 175 km trajectory and shrink the vehicle back to
                # a pixel -- the exact problem this pane exists to solve.
                rrb.Spatial3DView(name="Vehicle close-up", origin="chase_anchor",
                                  contents=["vehicle/**", "chase_anchor/**"]),
                rrb.TimeSeriesView(name="Flight state",
                                   origin="/",
                                   contents=["telemetry/altitude_m",
                                             "telemetry/v_vertical",
                                             "telemetry/v_horizontal",
                                             "telemetry/tilt_deg",
                                             "telemetry/tilt_cmd_deg"]),
                rrb.TimeSeriesView(name="Actuators", origin="telemetry",
                                   contents=["telemetry/throttle_main*",
                                             "telemetry/rcs_active"]),
            ),
            column_shares=[2.0, 1.0],
        ),
        collapse_panels=False,
    )


def _far_field_surface(radius: float, extent: float, n: int = 81):
    """Coarse spherical cap under the whole trajectory.

    The detailed terrain patch is 300 m across (``terrain.generate`` default)
    while the braking burn covers ~175 km, so at overview zoom there was no
    ground in the scene at all -- the vehicle appeared to fly over nothing.

    The cap is not flat, and that is the point: in the landing-site frame the
    surface falls away as ``sqrt(R^2 - d^2) - R``, which is **8.6 km below the
    site plane at 173 km downrange**. Drawing the far field at z = 0 would put
    the horizon kilometres out of place and make the descent look much
    shallower than it is.
    """
    g = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(g, g)
    d2 = X**2 + Y**2
    Z = np.sqrt(np.maximum(radius**2 - d2, 0.0)) - radius
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    idx = np.arange(n * n).reshape(n, n)
    tris = np.concatenate([
        np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:]], axis=-1).reshape(-1, 3),
        np.stack([idx[:-1, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 3),
    ]).astype(np.uint32)
    # Flat mid-grey, a little darker than the detailed patch so the landing
    # site reads as a distinct area rather than blending into the far field.
    cols = np.tile(np.array([58, 56, 54], np.uint8), (len(verts), 1))
    return verts, tris, cols


def _look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Camera rotation whose child frame is Rerun's RDF: +X right, +Y down, +Z forward.

    Returned as a world-from-camera matrix, i.e. its columns are the camera's
    right / down / forward axes expressed in world coordinates -- which is what
    ``rr.Transform3D(mat3x3=...)`` on the pinhole's parent entity wants.
    """
    fwd = target - eye
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    right = np.cross(up_hint, fwd)
    n = np.linalg.norm(right)
    if n < 1e-6:                                    # looking straight along up_hint
        right = np.cross(np.array([1.0, 0.0, 0.0]), fwd)
        n = np.linalg.norm(right)
    right /= n
    down = np.cross(fwd, right)
    return np.column_stack([right, down, fwd])


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
    own_recording: bool = True,
    marker_scale: float = 400.0,
    chase_distance: float = 60.0,
    chase_height: float = 22.0,
    blueprint: bool = True,
) -> Path:
    """``marker_scale``/``chase_*`` exist because of a scale problem.

    The trajectory is ~175 km long and the vehicle is 1.2 m tall, so at any
    zoom that shows the flight the vehicle is far below one pixel -- and the
    trajectory line, at its original 18 m radius, was itself 30x wider than
    the thing it was meant to lead the eye to. Two additions fix it without
    touching the real-scale geometry, which stays available for close-ups:

    * ``marker/`` draws the same hopper wireframe at ``marker_scale`` x with a
      thrust-axis arrow that tracks total throttle, so there is something
      visible flying the trajectory at overview zoom.
    * ``chase_cam`` is a real pinhole camera that trails the vehicle, so the
      viewer offers a flight view in which the geometry is at true scale.

    These are readability choices, not physical quantities, so they are
    arguments here rather than ``params.csv`` rows -- nothing in the
    simulation reads them.
    """
    # own_recording=False lets a caller (scripts/make_video.py) init the
    # recording, add its own entities, and save once -- so the descent, the
    # navigation telemetry and the Monte Carlo fan land in ONE .rrd instead of
    # three that have to be opened side by side.
    if own_recording:
        rr.init(app_id, spawn=False)
        if blueprint:
            rr.send_blueprint(_blueprint())
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
    moon_r = PARAMS.value("SIM-E03") * 1e3
    span = 1.3 * float(np.abs(state[:, I_POS][:, 0]).max()) + 5e3
    fv, ft, fc = _far_field_surface(moon_r, span)
    rr.log("world/surface", rr.Mesh3D(vertex_positions=fv, triangle_indices=ft,
                                      vertex_colors=fc), static=True)
    rr.log("world/trajectory", rr.LineStrips3D([state[:, I_POS]], colors=[[95, 165, 235]],
                                               radii=[4.0]), static=True)
    # Ground track: the trajectory projected onto the surface. Without it the
    # altitude of a point on a 175 km arc is impossible to judge by eye.
    gx, gy = state[:, I_POS][:, 0], state[:, I_POS][:, 1]
    # On the sphere, not on the z = 0 plane -- see _far_field_surface.
    gz = np.sqrt(np.maximum(moon_r**2 - gx**2 - gy**2, 0.0)) - moon_r
    near = (gx**2 + gy**2) < (0.45 * 300.0) ** 2
    gz = np.where(near, terrain.height_at(gx, gy), gz) + 2.0
    track = np.column_stack([gx, gy, gz])
    rr.log("world/ground_track", rr.LineStrips3D([track], colors=[[70, 80, 95]],
                                                 radii=[3.0]), static=True)
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
    # Oversized proxy so the vehicle is visible at trajectory zoom.
    rr.log("marker/body", rr.LineStrips3D(list(wire.values()),
                                          colors=[[255, 225, 140]] * len(wire),
                                          radii=[0.02]), static=True)
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
        rr.log("marker", rr.Transform3D(translation=s[I_POS], mat3x3=rot * marker_scale))

        # Thrust axis, in world metres so it reads at overview zoom. Length
        # tracks total main-engine throttle, so the marker shows WHEN the
        # vehicle is burning, not just where it is.
        duty = float(np.mean(thr[i][layout.main_idx]))
        axis_w = rot @ np.array([0.0, 0.0, -1.0])
        rr.log("world/thrust_axis", rr.Arrows3D(
            origins=[s[I_POS]], vectors=[axis_w * (600.0 + 5400.0 * duty)],
            colors=[[255, 150, 60]], radii=[60.0]))

        # Chase camera: trailing the velocity, raised, looking at the vehicle.
        rv = s[I_POS] + np.array([0.0, 0.0, radius])
        up = rv / np.linalg.norm(rv)
        vel = s[I_VEL]
        v_h = vel - float(vel @ up) * up
        nh = np.linalg.norm(v_h)
        back = -v_h / nh if nh > 1.0 else -np.array([1.0, 0.0, 0.0])
        eye = s[I_POS] + back * chase_distance + up * chase_height
        rr.log("chase_cam", rr.Transform3D(
            translation=eye, mat3x3=_look_at(eye, s[I_POS], up)))
        rr.log("chase_cam", rr.Pinhole(
            resolution=[1280, 720], focal_length=620.0,
            camera_xyz=rr.ViewCoordinates.RDF))
        # Follow anchor for the close-up view: the vehicle's POSITION with no
        # rotation. The camera above draws a clickable frustum in the overview
        # (select it and press the camera button to look through it); this
        # anchor is what the close-up pane is actually centred on, because a
        # 2D view on a pinhole can only show entities logged beneath that
        # pinhole, and the world is a sibling of it, not a child.
        rr.log("chase_anchor", rr.Transform3D(translation=s[I_POS]))

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
    if own_recording:
        rr.save(out)
    return out
