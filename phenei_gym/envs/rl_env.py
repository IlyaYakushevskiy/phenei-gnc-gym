"""Vectorised RL environment for the powered-descent landing problem.

Sits on ``BatchedDescent`` and adds the five things an RL loop needs:
observation, action, reward, termination and reset -- all batched, all on
device, no host round-trip.

Architecture is hierarchical (``TR-SIM-04``): the policy acts at 10 Hz and
commands *guidance* (thrust fraction and thrust direction), while the 100 Hz
attitude PD and the allocator stay classical. That keeps episodes to hundreds
of steps instead of tens of thousands, keeps the inner loop verifiable, and
keeps thruster placement a studiable parameter rather than something the
network silently absorbs.

Observations are ground truth for now. Sensor models and the navigation
filter are Phase 4; swapping them in changes only ``_observe``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from phenei_gym.envs.batched import BatchedConfig, BatchedDescent, quat_to_matrix
from phenei_gym.params import PARAMS

OBS_DIM = 15
ACT_DIM = 3


@dataclass
class LandingTaskConfig:
    batch: BatchedConfig = field(default_factory=BatchedConfig)
    start_altitude: float = 400.0        # curriculum knob (TR-SIM-06); start low,
                                         # extend once soft landings are reliable
    start_alt_jitter: float = 120.0
    start_speed: float = 40.0            # horizontal speed at episode start
    start_speed_jitter: float = 20.0
    start_offset: float = 250.0          # downrange offset from the target
    start_offset_jitter: float = 120.0
    max_tilt_cmd: float = 0.70           # rad, limit on commanded thrust tilt
    policy_hz: float = 10.0
    max_seconds: float = 45.0
    corridor_accel: float = 4.0          # m/s^2, the descent-rate corridor
    corridor_weight: float = 0.25
    mass0: float = 620.0                 # dry 500 + 120 kg for the last phase


class LandingTask:
    """N parallel landing episodes. Tensors in, tensors out, device-resident."""

    def __init__(self, cfg: LandingTaskConfig | None = None):
        self.cfg = cfg or LandingTaskConfig()
        self.cfg.batch.mass0 = self.cfg.mass0
        self.sim = BatchedDescent(self.cfg.batch)
        self.device = self.sim.device
        self.dtype = self.cfg.batch.dtype
        self.n = self.cfg.batch.n_envs
        self.substeps = max(1, int(round(1.0 / (self.cfg.policy_hz * self.cfg.batch.dt))))
        self.max_steps = int(self.cfg.max_seconds * self.cfg.policy_hz)

        self.t01 = PARAMS.value("SIM-T01")       # 50 m
        self.t02 = PARAMS.value("SIM-T02")       # 2.0 m/s
        self.t03 = PARAMS.value("SIM-T03")       # 0.5 m/s
        self.t04 = PARAMS.value("SIM-T04")       # 5 deg
        self.mass_dry = PARAMS.value("SIM-M01")
        self.step_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.prev_shaping = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.reset(torch.ones(self.n, dtype=torch.bool, device=self.device))

    # ------------------------------------------------------------------
    def reset(self, mask: torch.Tensor) -> None:
        c, s, n = self.cfg, self.sim, int(mask.sum())
        if n == 0:
            return
        dev, dt = self.device, self.dtype
        u = lambda a, b: torch.rand(n, device=dev, dtype=dt) * (b - a) + a

        alt = u(c.start_altitude - c.start_alt_jitter, c.start_altitude + c.start_alt_jitter)
        ang = u(0.0, 6.283185)
        off = u(c.start_offset - c.start_offset_jitter, c.start_offset + c.start_offset_jitter)
        s.pos[mask] = torch.stack([off * torch.cos(ang), off * torch.sin(ang), alt], dim=-1)

        spd = u(c.start_speed - c.start_speed_jitter, c.start_speed + c.start_speed_jitter)
        # roughly heading at the site, with scatter
        toward = -s.pos[mask][:, :2]
        toward = toward / toward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        jitter = torch.randn(n, 2, device=dev, dtype=dt) * 0.25
        hv = (toward + jitter) * spd[:, None]
        s.vel[mask] = torch.stack([hv[:, 0], hv[:, 1], u(-70.0, -30.0)], dim=-1)

        q = torch.zeros(n, 4, device=dev, dtype=dt); q[:, 3] = 1.0
        s.quat[mask] = q
        s.omega[mask] = torch.randn(n, 3, device=dev, dtype=dt) * 0.01
        s.mass[mask] = c.mass0
        self.step_count[mask] = 0
        self.prev_shaping[mask] = self._shaping()[mask]

    # ------------------------------------------------------------------
    def _shaping(self) -> torch.Tensor:
        s = self.sim
        horiz = s.pos[:, :2].norm(dim=-1)
        alt = s.altitude().clamp_min(0.0)
        speed = s.vel.norm(dim=-1)
        # Shaping gradient has to be strong enough that descending and
        # closing on the site beats loitering. With weak shaping the policy
        # discovers that never touching down avoids the crash penalty.
        return -(0.02 * horiz + 0.03 * alt + 0.30 * speed + 5.0 * s.tilt())

    def _observe(self) -> torch.Tensor:
        s = self.sim
        up = s.local_up()
        v_up = (s.vel * up).sum(-1, keepdim=True)
        v_h = s.vel - v_up * up
        body_z = quat_to_matrix(s.quat)[:, :, 2]
        alt = s.altitude()
        return torch.cat([
            s.pos[:, :2] / 500.0,
            (alt / 1000.0).unsqueeze(-1),
            v_h / 50.0,
            v_up / 30.0,
            body_z,
            s.omega / 0.5,
            ((s.mass - self.mass_dry) / 120.0).unsqueeze(-1),
            (self.step_count.to(self.dtype) / self.max_steps).unsqueeze(-1),
        ], dim=-1)

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """action: (N,3) in [-1,1] -> thrust fraction and thrust-direction tilt."""
        s, c = self.sim, self.cfg
        a = action.clamp(-1.0, 1.0)
        # Thrust is commanded RELATIVE TO HOVER, not as a raw fraction. With a
        # raw fraction a zero-mean Gaussian policy starts at 0.5 -> 2700 N on a
        # 620 kg vehicle -> 2.7 m/s^2 NET UPWARD, so the vehicle climbs away and
        # every episode times out without ever touching the ground. Anchoring
        # the action at hover makes a = 0 neutral and the problem learnable.
        # Anchored at hover but spanning the FULL range: a=0 hovers, a=+1 is
        # the whole thrust cap, a=-1 is engines off. The earlier symmetric form
        # hover*(1 + 1.2a) capped thrust at 2.2x hover = 2215 N of the 5400 N
        # available -- only 1.95 m/s^2 of net braking, which needs 332 m to
        # arrest 36 m/s in a 400 m episode. The task was unachievable, and the
        # four "reward failures" before this were all that one bug.
        g_eff = (self.sim.mu / (s.pos[:, 2] + self.sim.radius).pow(2))
        hover = ((s.mass * g_eff) / (c.batch.thrust_cap * c.batch.thrust_max)).clamp(0.0, 1.0)
        a0 = a[:, 0]
        thrust_frac = torch.where(a0 >= 0.0,
                                  hover + a0 * (1.0 - hover),
                                  hover * (1.0 + a0)).clamp(0.0, 1.0)
        up = s.local_up()
        east = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype).expand(self.n, 3)
        east = east - (east * up).sum(-1, keepdim=True) * up
        east = east / east.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        north = torch.cross(up, east, dim=-1)
        tilt = c.max_tilt_cmd
        direction = up + torch.tan(tilt * a[:, 1:2]) * east + torch.tan(tilt * a[:, 2:3]) * north
        s.set_command(thrust_frac, direction)

        for _ in range(self.substeps):
            s.step()
        self.step_count += 1

        alt = s.altitude()
        up = s.local_up()
        v_up = (s.vel * up).sum(-1)
        v_h = (s.vel - v_up.unsqueeze(-1) * up).norm(dim=-1)
        horiz = s.pos[:, :2].norm(dim=-1)
        tilt_now = s.tilt()

        landed = alt <= 0.0
        out_of_fuel = s.mass <= self.mass_dry + 1e-6
        timeout = self.step_count >= self.max_steps
        diverged = (alt > 2.0 * c.start_altitude) | (horiz > 6.0 * c.start_offset)
        done = landed | out_of_fuel | timeout | diverged

        shaping = self._shaping()
        reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward = reward - 0.05                                   # time/propellant pressure

        # DENSE descent-rate corridor. The terminal crash penalty alone is too
        # sparse to shape against: over 1e9 steps it only brought impact speed
        # from 47 to 37 m/s. This gives a signal at every step, and it is not
        # an arbitrary hack -- it is the same sqrt(2 a h) profile the classical
        # guidance flies (SIM-GD01), so the policy is being told to stay inside
        # a physically achievable braking envelope.
        v_limit = torch.sqrt(2.0 * c.corridor_accel * alt.clamp_min(0.0)) + self.t02
        over = (-v_up - v_limit).clamp_min(0.0) + (v_h - v_limit).clamp_min(0.0)
        reward = reward - c.corridor_weight * over

        good = (landed & (horiz <= self.t01) & (v_up.abs() <= self.t02)
                & (v_h <= self.t03) & (torch.rad2deg(tilt_now) <= self.t04))
        soft = landed & (v_up.abs() <= 4.0) & (v_h <= 3.0)
        hard = landed & ~soft
        # Timing out must be WORSE than attempting a landing and failing, or
        # the optimal policy is to loiter: a -60 crash against a -15 timeout
        # taught exactly that, and the land rate collapsed to 1e-4.
        reward = reward + good.to(self.dtype) * 300.0
        reward = reward + (soft & ~good).to(self.dtype) * 80.0
        # Crash penalty MUST scale with impact speed. A flat -40 is cheaper
        # than the +60 of altitude shaping earned by descending 2000 m, so the
        # policy learns to dive: over 100 iterations the miss distance improved
        # but impact speed went 40 -> 72 m/s. Scaling with speed makes slamming
        # in unambiguously worse than a controlled descent.
        impact = (v_up.abs() + v_h).clamp_max(120.0)
        reward = reward - hard.to(self.dtype) * (20.0 + 4.0 * impact)
        reward = reward - out_of_fuel.to(self.dtype) * 80.0
        reward = reward - diverged.to(self.dtype) * 80.0
        reward = reward - timeout.to(self.dtype) * 80.0
        reward = reward + soft.to(self.dtype) * (80.0 / (1.0 + horiz / 25.0))

        info = {"landed": landed, "good": good, "soft": soft, "diverged": diverged,
                "out_of_fuel": out_of_fuel, "timeout": timeout,
                "miss": horiz, "v_up": v_up, "v_h": v_h, "tilt_deg": torch.rad2deg(tilt_now)}
        if done.any():
            self.reset(done)
        # observe AFTER the reset: the returned obs must be the state the
        # policy will actually act on next, not the terminal state.
        return self._observe(), reward, done, info

    def observe(self) -> torch.Tensor:
        return self._observe()
