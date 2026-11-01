"""Rerun logging for the navigation chain: sensors, estimate and covariance.

This is the "sensor telemetry" half of the video deliverable.  The useful
thing to show is not the noise -- it is **when each sensor is and is not
working**, because the whole navigation story of this vehicle is a long
altimeter-free coast followed by a short, dense burst of measurements.

Three groups of entities:

``nav/`` scalars
    Estimate error against truth, and the filter's own 1-sigma next to it.
    Plotted together they show whether the filter is honest; the error line
    staying inside the sigma line is the claim being made.

``sensors/`` scalars
    One trace per sensor, 1 when it produced an accepted measurement in that
    frame and 0 otherwise.  Rendered as a plot this reads as an availability
    timeline, which is the single most informative picture of the problem.

``world/nav_*`` 3-D
    The estimated position next to the true one, and TRN fixes as points
    where they land.  Shows the estimate converging as fixes arrive.
"""

from __future__ import annotations

import math

import numpy as np
import rerun as rr

from phenei_gym.params import PARAMS

RADIUS = PARAMS.value("SIM-E03") * 1e3

SENSOR_COLORS = {
    "star": [255, 214, 102],
    "trn": [130, 200, 255],
    "DLEM30": [120, 230, 160],
    "DLEM17": [90, 200, 130],
    "RADAR": [255, 140, 120],
    "velocity": [200, 160, 255],
}


def log_static_nav_setup(sensor_names: list[str]) -> None:
    """Annotations that never change: requirement lines and sensor colours."""
    for name in sensor_names:
        rr.log(f"sensors/{name}",
               rr.SeriesLines(colors=[SENSOR_COLORS.get(name, [180, 180, 180])],
                              names=[name], widths=[2.0]),
               static=True)
    rr.log("nav/pos_error_m", rr.SeriesLines(colors=[[255, 120, 120]],
                                             names=["error"]), static=True)
    rr.log("nav/pos_sigma_m", rr.SeriesLines(colors=[[120, 120, 255]],
                                             names=["filter 1-sigma"]), static=True)
    rr.log("nav/vel_error_ms", rr.SeriesLines(colors=[[255, 120, 120]],
                                              names=["error"]), static=True)
    rr.log("nav/vel_sigma_ms", rr.SeriesLines(colors=[[120, 120, 255]],
                                              names=["filter 1-sigma"]), static=True)
    rr.log("nav/att_error_deg", rr.SeriesLines(colors=[[255, 120, 120]],
                                               names=["error"]), static=True)
    rr.log("nav/att_sigma_deg", rr.SeriesLines(colors=[[120, 120, 255]],
                                               names=["filter 1-sigma"]), static=True)


def log_nav_frame(
    t: float,
    truth_pos: np.ndarray,
    est_pos: np.ndarray,
    pos_err: float,
    pos_sigma: float,
    vel_err: float,
    vel_sigma: float,
    att_err_deg: float,
    att_sigma_deg: float,
    fired: dict[str, bool],
    trn_fix: np.ndarray | None = None,
) -> None:
    """One frame of navigation telemetry. ``fired`` is sensor -> accepted."""
    rr.set_time("time", duration=t)

    rr.log("nav/pos_error_m", rr.Scalars(pos_err))
    rr.log("nav/pos_sigma_m", rr.Scalars(pos_sigma))
    rr.log("nav/vel_error_ms", rr.Scalars(vel_err))
    rr.log("nav/vel_sigma_ms", rr.Scalars(vel_sigma))
    rr.log("nav/att_error_deg", rr.Scalars(att_err_deg))
    rr.log("nav/att_sigma_deg", rr.Scalars(att_sigma_deg))

    for name, on in fired.items():
        rr.log(f"sensors/{name}", rr.Scalars(1.0 if on else 0.0))

    rr.log("world/nav_estimate",
           rr.Points3D([est_pos], colors=[[255, 140, 140]], radii=[12.0]))
    if trn_fix is not None:
        rr.log("world/trn_fix",
               rr.Points3D([[trn_fix[0], trn_fix[1], float(truth_pos[2])]],
                           colors=[[130, 200, 255]], radii=[25.0]))


def log_availability_summary(counts: dict[str, int], total: int) -> None:
    """A static text panel: what fraction of the burn each sensor worked."""
    lines = ["| sensor | accepted fixes | % of burn |", "|---|---|---|"]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} | {100.0 * v / max(total, 1):.1f} % |")
    rr.log("notes/availability",
           rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN),
           static=True)
