"""Batched 6-DOF descent dynamics in PyTorch, for RL throughput.

The single-environment implementation in ``descent6dof.py`` is the reference:
readable, validated against the 3-DOF polar model, and ~25 s of wall time per
episode in pure Python. PPO needs on the order of 1e6 episodes-worth of steps,
so training needs this instead -- the same physics, batched over N
environments as CUDA tensors with no Python loop over the batch.

``tests/test_batched.py`` asserts parity with the reference implementation;
that test is the contract, and it must keep passing if either side changes.

Design notes
------------
* Allocation uses a **fixed** damped pseudo-inverse. The reference allocator
  iterates clip-and-resolve, which does not batch cleanly; the geometry is
  constant, so ``pinv(B)`` can be precomputed and applied as one matmul
  followed by a clamp. Slightly worse under saturation, far faster, and the
  parity test quantifies the difference.
* The RL action is the guidance command, not raw throttles: thrust fraction
  plus a thrust-direction tilt/azimuth. The 100 Hz attitude loop and the
  allocator stay classical (``TR-SIM-04``), which is what keeps episodes to a
  few thousand steps and keeps thruster placement a studiable parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from phenei_gym.actuators import ActuatorLayout
from phenei_gym.params import PARAMS

G0 = 9.80665


def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Hamilton product, scalar-last, batched over the leading dimension."""
    px, py, pz, pw = p.unbind(-1)
    qx, qy, qz, qw = q.unbind(-1)
    return torch.stack([
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
        pw * qw - px * qx - py * qy - pz * qz,
    ], dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(N,4) scalar-last -> (N,3,3) body->world."""
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_pointing(v: torch.Tensor) -> torch.Tensor:
    """Shortest rotation taking body +Z onto each unit vector in ``v`` (N,3)."""
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    z = torch.zeros_like(v)
    z[..., 2] = 1.0
    axis = torch.cross(z, v, dim=-1)
    s = axis.norm(dim=-1, keepdim=True)
    angle = torch.atan2(s.squeeze(-1), (z * v).sum(-1)).unsqueeze(-1)
    axis = torch.where(s > 1e-12, axis / s.clamp_min(1e-12), torch.tensor([1.0, 0.0, 0.0],
                       device=v.device, dtype=v.dtype).expand_as(v))
    half = 0.5 * angle
    return torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)


@dataclass
class BatchedConfig:
    n_envs: int = 4096
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    thrust_max: float = 6000.0
    thrust_cap: float = 0.90
    dt: float = 1.0 / 200.0
    attitude_substeps: int = 2      # 100 Hz attitude loop at dt = 1/200
    omega_n: float = 1.5
    zeta: float = 0.9
    mass0: float = 950.0
    ignition_altitude: float = 75e3


class BatchedDescent:
    """N independent 6-DOF descents stepped together on one device."""

    def __init__(self, cfg: BatchedConfig | None = None):
        self.cfg = cfg or BatchedConfig()
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            self.cfg.device = "cpu"
        dev, dt_ = self.cfg.device, self.cfg.dtype
        self.device = torch.device(dev)

        self.mu = PARAMS.value("SIM-E02")
        self.radius = PARAMS.value("SIM-E03") * 1e3
        self.mass_dry = PARAMS.value("SIM-M01")
        self.c_main = PARAMS.value("SIM-P03") * G0

        layout = ActuatorLayout.from_params(self.cfg.thrust_max)
        self.layout = layout
        B = torch.tensor(layout.effectiveness(), device=dev, dtype=dt_)      # (4, n_thr)
        scale = B.abs().sum(dim=1).clamp_min(1e-9)
        w = torch.tensor([1.0, 20.0, 20.0, 20.0], device=dev, dtype=dt_) / scale
        Bw = B * w[:, None]
        self.B = B
        self.w = w
        # constant geometry -> constant damped pseudo-inverse
        self.pinv = Bw.T @ torch.linalg.inv(Bw @ Bw.T + 1e-9 * torch.eye(4, device=dev, dtype=dt_))
        self.isp_vec = torch.tensor([t.thrust_max / (t.isp * G0) for t in layout.thrusters],
                                    device=dev, dtype=dt_)

        I_ref = np.array([PARAMS.value("SIM-M07"), PARAMS.value("SIM-M07"),
                          PARAMS.value("SIM-M08")])
        self.I_ref = torch.tensor(I_ref, device=dev, dtype=dt_)
        self.m_ref = PARAMS.value("SIM-M03")

        n = self.cfg.n_envs
        self.pos = torch.zeros(n, 3, device=dev, dtype=dt_)
        self.vel = torch.zeros(n, 3, device=dev, dtype=dt_)
        self.quat = torch.zeros(n, 4, device=dev, dtype=dt_)
        self.quat[:, 3] = 1.0
        self.omega = torch.zeros(n, 3, device=dev, dtype=dt_)
        self.mass = torch.full((n,), self.cfg.mass0, device=dev, dtype=dt_)
        self.q_cmd = self.quat.clone()
        self.thrust_cmd = torch.zeros(n, device=dev, dtype=dt_)

    # ------------------------------------------------------------------
    def inertia(self) -> torch.Tensor:
        return self.I_ref[None, :] * (self.mass / self.m_ref)[:, None]

    def local_up(self) -> torch.Tensor:
        r = self.pos.clone()
        r[:, 2] += self.radius
        return r / r.norm(dim=-1, keepdim=True)

    def altitude(self) -> torch.Tensor:
        r = self.pos.clone()
        r[:, 2] += self.radius
        return r.norm(dim=-1) - self.radius

    def gravity(self) -> torch.Tensor:
        r = self.pos.clone()
        r[:, 2] += self.radius
        n = r.norm(dim=-1, keepdim=True)
        return -self.mu * r / n.pow(3)

    def tilt(self) -> torch.Tensor:
        body_z = quat_to_matrix(self.quat)[:, :, 2]
        return torch.arccos(((body_z * self.local_up()).sum(-1)).clamp(-1.0, 1.0))

    # ------------------------------------------------------------------
    def set_command(self, thrust_frac: torch.Tensor, direction: torch.Tensor) -> None:
        """Guidance command: thrust fraction in [0,1] and a world thrust direction."""
        cap = self.cfg.thrust_cap * self.cfg.thrust_max
        self.thrust_cmd = thrust_frac.clamp(0.0, 1.0) * cap
        self.q_cmd = quat_pointing(direction)

    def _allocate(self) -> torch.Tensor:
        """Throttles (N, n_thr) in [0, 1] for the current command and attitude."""
        I = self.inertia()
        kp = 2.0 * I * self.cfg.omega_n**2
        kd = 2.0 * I * self.cfg.zeta * self.cfg.omega_n
        q_err = quat_mul(quat_conj(self.q_cmd), self.quat)
        q_err = torch.where((q_err[:, 3:4] < 0.0), -q_err, q_err)
        gyro = torch.cross(self.omega, I * self.omega, dim=-1)
        tau = -kp * q_err[:, :3] - kd * self.omega + gyro
        wrench = torch.cat([self.thrust_cmd[:, None], tau], dim=-1) * self.w[None, :]
        return (wrench @ self.pinv.T).clamp(0.0, 1.0)

    def _derivative(self, pos, vel, quat, omega, mass, force_b, torque_b, mdot):
        rot = quat_to_matrix(quat)
        r = pos.clone(); r[:, 2] += self.radius
        nrm = r.norm(dim=-1, keepdim=True)
        acc = torch.einsum("nij,nj->ni", rot, force_b) / mass[:, None] - self.mu * r / nrm.pow(3)
        wq = torch.cat([omega, torch.zeros_like(omega[:, :1])], dim=-1)
        dq = 0.5 * quat_mul(quat, wq)
        I = self.inertia()
        dom = (torque_b - torch.cross(omega, I * omega, dim=-1)) / I
        return vel, acc, dq, dom, -mdot

    def step(self) -> None:
        """One dt of physics, with the attitude loop at ``attitude_substeps`` rate."""
        dt = self.cfg.dt
        u = self._allocate()
        live = (self.mass > self.mass_dry).to(u.dtype)[:, None]
        u = u * live
        wrench = u @ self.B.T                      # (N, 4)
        force_b = torch.zeros_like(self.pos)
        force_b[:, 2] = wrench[:, 0]
        torque_b = wrench[:, 1:]
        mdot = (u * self.isp_vec[None, :]).sum(-1)

        s = (self.pos, self.vel, self.quat, self.omega, self.mass)
        k1 = self._derivative(*s, force_b, torque_b, mdot)
        s2 = tuple(a + 0.5 * dt * b for a, b in zip(s, k1))
        k2 = self._derivative(*s2, force_b, torque_b, mdot)
        s3 = tuple(a + 0.5 * dt * b for a, b in zip(s, k2))
        k3 = self._derivative(*s3, force_b, torque_b, mdot)
        s4 = tuple(a + dt * b for a, b in zip(s, k3))
        k4 = self._derivative(*s4, force_b, torque_b, mdot)

        self.pos, self.vel, self.quat, self.omega, self.mass = tuple(
            a + (dt / 6.0) * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4))
        self.quat = self.quat / self.quat.norm(dim=-1, keepdim=True)
        self.mass = self.mass.clamp_min(self.mass_dry)
