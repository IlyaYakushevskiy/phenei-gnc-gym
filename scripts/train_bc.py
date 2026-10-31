#!/usr/bin/env python3
"""Behaviour-clone the classical expert, then fine-tune with PPO.

Pure PPO from scratch never found a soft landing: a landing needs sustained,
well-timed high thrust near the surface and Gaussian noise around hover
almost never produces it. Cloning the classical law (which lands) removes the
exploration problem; PPO then improves on it, which is what RL is actually
for here.
"""
from __future__ import annotations

import argparse, csv, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.envs.batched import BatchedConfig
from phenei_gym.envs.rl_env import ACT_DIM, OBS_DIM, LandingTask, LandingTaskConfig
from phenei_gym.rl import ActorCritic, PPO, PPOConfig
from phenei_gym.rl.expert import expert_action

EXPERT = dict(a_dec=1.5, k_vz=0.5, k_pos=0.02, k_vh=1.2)


def evaluate(task, policy, steps=700, deterministic=True):
    tot = {k: 0 for k in ("landed", "good", "soft", "diverged", "timeout", "out_of_fuel")}
    miss = vup = vh = 0.0
    obs = task.observe()
    with torch.no_grad():
        for _ in range(steps):
            a = policy.pi(obs) if deterministic else policy.act(obs)[0]
            obs, r, d, i = task.step(a)
            for k in tot: tot[k] += int(i[k].sum())
            m = i["landed"]
            if m.any():
                miss += float(i["miss"][m].sum()); vup += float(i["v_up"][m].abs().sum())
                vh += float(i["v_h"][m].sum())
    L = max(tot["landed"], 1)
    return tot, miss / L, vup / L, vh / L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16384)
    ap.add_argument("--bc-iters", type=int, default=400)
    ap.add_argument("--ppo-iters", type=int, default=1500)
    ap.add_argument("--out", default="studies/bc")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    task = LandingTask(LandingTaskConfig(batch=BatchedConfig(n_envs=a.envs), start_altitude=400.0))
    pol = ActorCritic(OBS_DIM, ACT_DIM).to(task.device)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    obs = task.observe()
    t0 = time.time()

    # ---- phase 1: behaviour cloning (DAgger-style, on the policy's own states)
    print("=== behaviour cloning ===", flush=True)
    for it in range(1, a.bc_iters + 1):
        losses = []
        for _ in range(16):
            with torch.no_grad():
                tgt = expert_action(task, **EXPERT)
            pred = pol.pi(obs)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss))
            # roll forward on a MIX of expert and policy actions so the policy
            # sees the states it will actually visit, not just the expert's
            with torch.no_grad():
                mix = torch.rand(obs.shape[0], 1, device=task.device) < min(0.9, it / 100)
                act = torch.where(mix, pred.detach(), tgt)
            obs, _, _, _ = task.step(act)
        if it % 50 == 0 or it == 1:
            print(f"  bc {it:4d} | mse {sum(losses)/len(losses):.5f} | "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    torch.save({"policy": pol.state_dict()}, out / "bc.pt")

    ev = LandingTask(LandingTaskConfig(batch=BatchedConfig(n_envs=4096), start_altitude=400.0))
    tot, mi, vu, vh = evaluate(ev, pol)
    L = max(tot["landed"], 1)
    print(f"  AFTER BC: land {tot['landed']} soft {100*tot['soft']/L:.1f}% good {tot['good']} "
          f"div {tot['diverged']} | miss {mi:.1f} vup {vu:.2f} vh {vh:.2f}", flush=True)

    # ---- phase 2: PPO fine-tune
    print("=== PPO fine-tune ===", flush=True)
    algo = PPO(pol, PPOConfig(rollout=32, lr=1e-4), task.device)
    fh = (out / "progress.csv").open("w", newline=""); wr = csv.writer(fh)
    wr.writerow(["iter", "minutes", "land_rate", "good_rate", "soft_rate", "miss", "vup", "vh"])
    obs = task.observe(); best = -1.0
    for it in range(1, a.ppo_iters + 1):
        obs, batch, st = algo.collect(task, obs)
        algo.update(batch)
        if it % 25 == 0 or it == 1:
            tot, mi, vu, vh = evaluate(ev, pol)
            L = max(tot["landed"], 1)
            g = tot["good"] / L
            wr.writerow([it, round((time.time()-t0)/60, 1), tot["landed"], round(g, 4),
                         round(tot["soft"]/L, 4), round(mi, 1), round(vu, 2), round(vh, 2)])
            fh.flush()
            print(f"  ppo {it:4d} | {(time.time()-t0)/60:6.1f} min | land {tot['landed']:6d} "
                  f"soft {100*tot['soft']/L:5.1f}% good {100*g:5.2f}% div {tot['diverged']:5d} | "
                  f"miss {mi:7.1f} vup {vu:5.2f} vh {vh:5.2f}", flush=True)
            if g > best:
                best = g
                torch.save({"policy": pol.state_dict(), "good_rate": g, "iter": it},
                           out / "best.pt")
    fh.close()
    print(f"done. best good_rate {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
