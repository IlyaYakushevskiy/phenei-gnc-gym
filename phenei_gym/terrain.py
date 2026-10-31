"""Procedural lunar landing-surface model.

Built from the environment rows so the Monte Carlo can randomise it:
slope ``SIM-E06``, RMS roughness ``SIM-E07``, hazard areal density
``SIM-E08``, regolith albedo ``SIM-E09``.

This is deliberately not demo-only scenery. The same heightmap is what the
laser and radar altimeters will raycast against in Phase 4, and what the
touchdown tip-over check needs, so it lives in the library rather than in the
visualisation layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phenei_gym.params import PARAMS

# Sun low on the horizon: GNC-14 requires a Sun elevation constraint (value
# TBD), and low grazing light is both the realistic lunar case and what makes
# relief legible.
SUN_ELEVATION_DEG = 15.0
SUN_AZIMUTH_DEG = 135.0


@dataclass
class Terrain:
    """Regular heightmap on a square patch centred on the landing site."""

    x: np.ndarray          # (nx,) metres, landing-site frame
    y: np.ndarray          # (ny,)
    height: np.ndarray     # (ny, nx) metres above the reference sphere
    albedo: float
    slope_deg: float
    roughness: float
    hazards: np.ndarray    # (k, 4) rows of (cx, cy, radius, height)

    # ------------------------------------------------------------------
    def height_at(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Bilinear sample. Points outside the patch clamp to the edge."""
        px = np.clip(np.asarray(px, dtype=float), self.x[0], self.x[-1])
        py = np.clip(np.asarray(py, dtype=float), self.y[0], self.y[-1])
        fx = (px - self.x[0]) / (self.x[1] - self.x[0])
        fy = (py - self.y[0]) / (self.y[1] - self.y[0])
        i0 = np.clip(np.floor(fx).astype(int), 0, len(self.x) - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, len(self.y) - 2)
        tx, ty = fx - i0, fy - j0
        h = self.height
        return (h[j0, i0] * (1 - tx) * (1 - ty) + h[j0, i0 + 1] * tx * (1 - ty)
                + h[j0 + 1, i0] * (1 - tx) * ty + h[j0 + 1, i0 + 1] * tx * ty)

    def normals(self) -> np.ndarray:
        """Unit surface normals, (ny, nx, 3)."""
        dy, dx = np.gradient(self.height, self.y[1] - self.y[0], self.x[1] - self.x[0])
        n = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def shaded_colors(self) -> np.ndarray:
        """Per-vertex RGB (uint8) from Lambertian shading under a low sun."""
        el, az = math.radians(SUN_ELEVATION_DEG), math.radians(SUN_AZIMUTH_DEG)
        sun = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        lam = np.clip(self.normals() @ sun, 0.0, 1.0)
        # ambient floor so shadowed faces read as dark grey, not black
        v = self.albedo * (0.12 + 0.88 * lam)
        # regolith is slightly warm-grey; scale into a visible range
        base = np.stack([v * 1.00, v * 0.96, v * 0.90], axis=-1)
        return np.clip(base / max(self.albedo, 1e-6) * 215.0, 10, 255).astype(np.uint8)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(vertices (N,3), triangles (M,3), vertex colours (N,3))."""
        xx, yy = np.meshgrid(self.x, self.y)
        verts = np.stack([xx.ravel(), yy.ravel(), self.height.ravel()], axis=-1)
        cols = self.shaded_colors().reshape(-1, 3)
        nx, ny = len(self.x), len(self.y)
        j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
        v00 = (j * nx + i).ravel()
        v10, v01, v11 = v00 + 1, v00 + nx, v00 + nx + 1
        tris = np.concatenate([np.stack([v00, v10, v11], -1), np.stack([v00, v11, v01], -1)])
        return verts, tris.astype(np.uint32), cols


def _value_noise(rng: np.random.Generator, shape: tuple[int, int], octaves: int = 5) -> np.ndarray:
    """Cheap fractal noise: summed, smoothed random grids at halving amplitude."""
    out = np.zeros(shape)
    amp, res = 1.0, 4
    for _ in range(octaves):
        coarse = rng.standard_normal((res, res))
        # bilinear upsample to the full grid
        yi = np.linspace(0, res - 1, shape[0])
        xi = np.linspace(0, res - 1, shape[1])
        y0 = np.clip(np.floor(yi).astype(int), 0, res - 2)
        x0 = np.clip(np.floor(xi).astype(int), 0, res - 2)
        ty, tx = (yi - y0)[:, None], (xi - x0)[None, :]
        up = (coarse[np.ix_(y0, x0)] * (1 - ty) * (1 - tx)
              + coarse[np.ix_(y0, x0 + 1)] * (1 - ty) * tx
              + coarse[np.ix_(y0 + 1, x0)] * ty * (1 - tx)
              + coarse[np.ix_(y0 + 1, x0 + 1)] * ty * tx)
        out += amp * up
        amp *= 0.5
        res *= 2
    return out


def generate(
    seed: int = 0,
    extent: float = 300.0,
    n: int = 193,
    slope_deg: float | None = None,
    roughness: float | None = None,
    hazard_density: float | None = None,
    albedo: float | None = None,
) -> Terrain:
    """Random landing surface. Defaults sample the SIM-E0x ranges."""
    rng = np.random.default_rng(seed)
    lo, hi = PARAMS.range("SIM-E06")
    slope_deg = rng.uniform(lo, hi) if slope_deg is None else slope_deg
    lo, hi = PARAMS.range("SIM-E07")
    roughness = rng.uniform(lo, hi) if roughness is None else roughness
    lo, hi = PARAMS.range("SIM-E08")
    hazard_density = rng.uniform(lo, hi) / 100.0 if hazard_density is None else hazard_density
    lo, hi = PARAMS.range("SIM-E09")
    albedo = rng.uniform(lo, hi) if albedo is None else albedo

    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(x, y)

    # regional slope, tilted about an arbitrary azimuth
    az = rng.uniform(0, 2 * math.pi)
    h = math.tan(math.radians(slope_deg)) * (xx * math.cos(az) + yy * math.sin(az))

    # fractal roughness, normalised to the requested RMS
    noise = _value_noise(rng, (n, n))
    if noise.std() > 0:
        h = h + roughness * noise / noise.std()

    # craters: depression with a raised rim
    hazards = []
    area = (2 * extent) ** 2
    n_craters = int(hazard_density * area / 400.0)
    for _ in range(max(0, n_craters)):
        cx, cy = rng.uniform(-extent, extent, 2)
        r = rng.uniform(3.0, 18.0)
        d = np.hypot(xx - cx, yy - cy) / r
        bowl = np.where(d < 1.0, -0.18 * r * (1.0 - d**2), 0.0)
        rim = np.where((d >= 0.85) & (d < 1.45), 0.05 * r * np.exp(-((d - 1.05) / 0.18) ** 2), 0.0)
        h = h + bowl + rim
        hazards.append((cx, cy, r, -0.18 * r))

    # rocks: small bumps, the thing hazard detection actually has to see
    n_rocks = int(hazard_density * area / 60.0)
    for _ in range(max(0, n_rocks)):
        cx, cy = rng.uniform(-extent, extent, 2)
        r = rng.uniform(0.4, 2.2)
        hr = rng.uniform(0.3, 1.0) * r
        h = h + hr * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.5 * r) ** 2)))
        hazards.append((cx, cy, r, hr))

    # the landing site itself is the datum
    h = h - float(h[n // 2, n // 2])
    return Terrain(x=x, y=y, height=h, albedo=albedo, slope_deg=slope_deg,
                   roughness=roughness, hazards=np.array(hazards) if hazards else np.zeros((0, 4)))
