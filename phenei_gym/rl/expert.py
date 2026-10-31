"""Classical expert for behaviour cloning the landing policy.

Four successive reward structures failed to produce a soft landing in ~2.5e9
environment steps, each in a different way (loiter exploit, dive exploit,
hover-bias bug, and a descent corridor that only bit near the ground). The
common cause is exploration: a soft landing requires sustained, well-timed
high thrust near the surface, and zero-mean Gaussian noise around hover
almost never produces it, so the policy never sees the reward it is supposed
to climb toward.

Rather than keep tuning scalars, this seeds the policy from the classical
guidance that already lands the vehicle (21 m, 0.98 m/s -- TR-SIM-20). PPO
then fine-tunes for robustness, which is what RL is actually good for here.
Standard practice, and it turns the exploration problem into a supervised one.

The expert writes directly in the policy's action space, so cloning is a
plain regression.
"""

from __future__ import annotations

import torch

from phenei_gym.params import PARAMS


def expert_action(task, a_dec: float = 4.0, k_vz: float = 0.5,
                  k_pos: float = 0.035, k_vh: float = 0.9) -> torch.Tensor:
    """(N,3) action in [-1,1] reproducing the classical descent law."""
    s = task.sim
    up = s.local_up()
    alt = s.altitude().clamp_min(0.0)
    v_up = (s.vel * up).sum(-1)
    v_h_vec = s.vel - v_up.unsqueeze(-1) * up
    g_eff = s.mu / (s.pos[:, 2] + s.radius).pow(2)

    # vertical: constant-deceleration profile with feedforward (the omission of
    # that feedforward is what made the 6-DOF commanded attitude bang-bang)
    v_z_des = -torch.sqrt(2.0 * a_dec * alt + task.t02**2)
    a_up = g_eff + a_dec + k_vz * (v_z_des - v_up)

    # horizontal: PD onto the landing site, bounded so it cannot dominate
    horiz = s.pos.clone(); horiz[:, 2] = 0.0
    a_h_vec = -k_pos * horiz - k_vh * v_h_vec
    a_h_vec = a_h_vec - (a_h_vec * up).sum(-1, keepdim=True) * up
    lim = 5.0
    n = a_h_vec.norm(dim=-1, keepdim=True)
    a_h_vec = torch.where(n > lim, a_h_vec * lim / n.clamp_min(1e-9), a_h_vec)

    a_vec = a_up.unsqueeze(-1) * up + a_h_vec
    a_mag = a_vec.norm(dim=-1).clamp_min(1e-6)

    # --- back out the action parameterisation ---
    cap = task.cfg.batch.thrust_cap * task.cfg.batch.thrust_max
    hover = ((s.mass * g_eff) / cap).clamp(0.0, 1.0)
    want = ((s.mass * a_mag) / cap).clamp(0.0, 1.0)
    # invert the env's asymmetric hover-anchored thrust mapping
    a0 = torch.where(want >= hover,
                     (want - hover) / (1.0 - hover).clamp_min(1e-6),
                     want / hover.clamp_min(1e-9) - 1.0)

    d = a_vec / a_mag.unsqueeze(-1)
    east = torch.tensor([1.0, 0.0, 0.0], device=s.pos.device, dtype=s.pos.dtype).expand_as(up)
    east = east - (east * up).sum(-1, keepdim=True) * up
    east = east / east.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    north = torch.cross(up, east, dim=-1)
    du = (d * up).sum(-1).clamp_min(1e-3)
    a1 = torch.atan((d * east).sum(-1) / du) / task.cfg.max_tilt_cmd
    a2 = torch.atan((d * north).sum(-1) / du) / task.cfg.max_tilt_cmd
    return torch.stack([a0, a1, a2], dim=-1).clamp(-1.0, 1.0)
