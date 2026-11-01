"""Compact PPO for the batched landing task.

Written against the batched tensor env directly rather than a Gym vector
wrapper: the env never leaves the GPU, so neither should the rollout buffer.
This is the ``rsl_rl`` pattern (which is the ETH RSL workflow) in a form small
enough to read in one sitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    rollout: int = 32
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.004
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    init_log_std: float = -0.7


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 init_log_std: float = -0.7):
        super().__init__()
        def trunk():
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU())
        self.pi = nn.Sequential(trunk(), nn.Linear(hidden, act_dim))
        self.vf = nn.Sequential(trunk(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.vf(obs).squeeze(-1)

    def evaluate(self, obs, act):
        d = self.dist(obs)
        return d.log_prob(act).sum(-1), d.entropy().sum(-1), self.vf(obs).squeeze(-1)


class PPO:
    def __init__(self, policy: ActorCritic, cfg: PPOConfig, device):
        self.p, self.cfg, self.device = policy.to(device), cfg, device
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def collect(self, env, obs):
        c, n = self.cfg, obs.shape[0]
        O = torch.zeros(c.rollout, n, obs.shape[1], device=self.device)
        A = torch.zeros(c.rollout, n, self.p.log_std.numel(), device=self.device)
        LP = torch.zeros(c.rollout, n, device=self.device)
        V = torch.zeros(c.rollout, n, device=self.device)
        R = torch.zeros(c.rollout, n, device=self.device)
        D = torch.zeros(c.rollout, n, device=self.device)
        stats = {}
        for t in range(c.rollout):
            a, lp, v = self.p.act(obs)
            O[t], A[t], LP[t], V[t] = obs, a, lp, v
            obs, r, d, info = env.step(a)
            R[t], D[t] = r, d.to(R.dtype)
            for k in ("landed", "good", "soft", "out_of_fuel"):
                stats[k] = stats.get(k, 0.0) + float(info[k].sum())
            if info["landed"].any():
                m = info["landed"]
                stats["miss_sum"] = stats.get("miss_sum", 0.0) + float(info["miss"][m].sum())
                stats["vup_sum"] = stats.get("vup_sum", 0.0) + float(info["v_up"][m].abs().sum())
        last_v = self.p.vf(obs).squeeze(-1)

        adv = torch.zeros_like(R)
        gae = torch.zeros(n, device=self.device)
        for t in reversed(range(c.rollout)):
            nv = last_v if t == c.rollout - 1 else V[t + 1]
            nonterm = 1.0 - D[t]
            delta = R[t] + c.gamma * nv * nonterm - V[t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
        return obs, (O, A, LP, V, adv, adv + V), stats

    def update(self, batch, anchor=None, anchor_coef: float = 0.0):
        """One PPO update.

        ``anchor`` is an optional frozen reference policy (the behaviour-cloned
        one).  Adding ``anchor_coef * ||mu - mu_anchor||^2`` to the loss keeps
        the fine-tune inside a trust region around the cloned policy, which
        PPO's own ratio clip does NOT do: the clip bounds each step relative
        to the policy as it was one iteration ago, so a slow drift of many
        small in-clip steps walks arbitrarily far from the start. That is
        exactly the failure seen in studies/bc_v2: 76.1% soft at iteration 25
        decaying to 56.6% by 100 with every single step inside the clip.
        """
        c = self.cfg
        O, A, LP, V, adv, ret = (x.flatten(0, 1) for x in batch)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        total = O.shape[0]
        mb = total // c.minibatches
        last = {}
        for _ in range(c.epochs):
            idx = torch.randperm(total, device=self.device)
            for k in range(c.minibatches):
                j = idx[k * mb:(k + 1) * mb]
                lp, ent, v = self.p.evaluate(O[j], A[j])
                ratio = (lp - LP[j]).exp()
                l1 = ratio * adv[j]
                l2 = ratio.clamp(1 - c.clip, 1 + c.clip) * adv[j]
                pl = -torch.min(l1, l2).mean()
                vl = (v - ret[j]).pow(2).mean()
                loss = pl + c.value_coef * vl - c.entropy_coef * ent.mean()
                al = torch.zeros((), device=self.device)
                if anchor is not None and anchor_coef > 0.0:
                    with torch.no_grad():
                        mu_a = anchor.pi(O[j])
                    al = (self.p.pi(O[j]) - mu_a).pow(2).sum(-1).mean()
                    loss = loss + anchor_coef * al
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.p.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    last = {"policy_loss": float(pl), "value_loss": float(vl),
                            "entropy": float(ent.mean()),
                            "kl": float((LP[j] - lp).mean()),
                            "std": float(self.p.log_std.exp().mean()),
                            "anchor": float(al)}
            if last.get("kl", 0.0) > c.target_kl:
                break
        return last
