"""Monte Carlo dispersions, shared by the campaign and the video builder.

Lives in the package rather than in ``scripts/mc_campaign.py`` so both the
campaign and ``scripts/make_video.py`` read the same table. Importing it
across scripts by file path does not work: ``@dataclass`` needs the module
registered in ``sys.modules``, and a path-loaded module is not.

Every figure here is an ``Assumption (LLM)`` pending TR-08 (injection
dispersion, which mission analysis has not run, so ``SIM-A15`` is unknown)
and the propulsion team's engine acceptance tolerances. They are documented
as ``SIM-MC01``-``MC09`` and are swept by a scale factor rather than trusted
as single values.
"""

from __future__ import annotations

import numpy as np

#: 1-sigma additive dispersions on the handover state, SI units.
SIGMA = {
    "d_altitude": 500.0,        # m      ignition altitude knowledge/execution
    "d_v_horizontal": 10.0,     # m/s    injection velocity error, in-plane
    "d_v_vertical": 5.0,        # m/s
    "d_crossrange": 1000.0,     # m      out-of-plane injection error
    "d_v_crossrange": 5.0,      # m/s
    "d_mass": 10.0,             # kg     propellant-load and dry-mass knowledge
}

#: 1-sigma multiplicative dispersions on engine performance.
SIGMA_SCALE = {
    "thrust_scale": 0.02,       # 2 % delivered thrust
    "isp_scale": 0.01,          # 1 % Isp
}

#: SIM-MC09. Truncation keeps the 95th and max statistics of a few-hundred
#: case campaign from being set by individual extreme draws.
TRUNCATE_SIGMA = 3.0


def truncated_normal(rng: np.random.Generator, sigma: float,
                     cut: float = TRUNCATE_SIGMA) -> float:
    """Gaussian truncated at ``cut`` sigma."""
    if sigma == 0.0:
        return 0.0
    while True:
        v = rng.normal(0.0, sigma)
        if abs(v) <= cut * sigma:
            return float(v)


def draw(rng: np.random.Generator, inject_scale: float = 1.0,
         perf_scale: float = 1.0) -> dict[str, float]:
    """One dispersed case, as Descent6DOFConfig keyword arguments."""
    d = {k: truncated_normal(rng, v * inject_scale) for k, v in SIGMA.items()}
    d["thrust_scale"] = 1.0 + truncated_normal(
        rng, SIGMA_SCALE["thrust_scale"] * perf_scale)
    d["isp_scale"] = 1.0 + truncated_normal(
        rng, SIGMA_SCALE["isp_scale"] * perf_scale)
    return d
