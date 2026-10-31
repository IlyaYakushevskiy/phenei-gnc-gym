#!/usr/bin/env python3
"""Train a landing guidance policy with PPO on the batched CUDA env."""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenei_gym.envs.batched import BatchedConfig
from phenei_gym.envs.rl_env import ACT_DIM, OBS_DIM, LandingTask, LandingTaskConfig
from phenei_gym.rl import ActorCritic, PPO, PPOConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16384)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--start-altitude", type=float, default=2000.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default="studies/train")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    task = LandingTask(LandingTaskConfig(batch=BatchedConfig(n_envs=a.envs),
                                         start_altitude=a.start_altitude))
    pol = ActorCritic(OBS_DIM, ACT_DIM)
    algo = PPO(pol, PPOConfig(rollout=a.rollout, lr=a.lr), task.device)
    obs = task.observe()

    csv_path = out / "progress.csv"
    fh = csv_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["iter", "env_steps", "minutes", "reward", "land_rate", "good_rate",
                 "soft_rate", "fuel_rate", "miss_mean", "vup_mean", "entropy", "std", "kl"])

    best = -1.0
    t0 = time.time()
    steps = 0
    print(f"device={task.device} envs={a.envs} rollout={a.rollout} "
          f"substeps={task.substeps} obs={OBS_DIM} act={ACT_DIM}", flush=True)
    for it in range(1, a.iters + 1):
        obs, batch, st = algo.collect(task, obs)
        m = algo.update(batch)
        steps += a.rollout * a.envs * task.substeps
        landed = max(st.get("landed", 0.0), 1.0)
        rew = float(batch[4].mean() + batch[3].mean())
        row = dict(
            iter=it, env_steps=steps, minutes=round((time.time() - t0) / 60, 2),
            reward=round(float(batch[5].mean()), 3),
            land_rate=round(st.get("landed", 0.0) / (a.rollout * a.envs), 5),
            good_rate=round(st.get("good", 0.0) / landed, 4),
            soft_rate=round(st.get("soft", 0.0) / landed, 4),
            fuel_rate=round(st.get("out_of_fuel", 0.0) / (a.rollout * a.envs), 5),
            miss_mean=round(st.get("miss_sum", 0.0) / landed, 1),
            vup_mean=round(st.get("vup_sum", 0.0) / landed, 2),
            entropy=round(m.get("entropy", 0.0), 3), std=round(m.get("std", 0.0), 3),
            kl=round(m.get("kl", 0.0), 5))
        wr.writerow(list(row.values())); fh.flush()
        if it % a.log_every == 0 or it == 1:
            print(f"it {it:5d} | {steps/1e6:7.1f} M steps | {row['minutes']:6.1f} min | "
                  f"ret {row['reward']:8.2f} | land {row['land_rate']:.4f} "
                  f"good {row['good_rate']:.3f} soft {row['soft_rate']:.3f} "
                  f"fuel {row['fuel_rate']:.4f} | miss {row['miss_mean']:7.1f} m "
                  f"vup {row['vup_mean']:6.2f} | std {row['std']:.3f}", flush=True)
        score = row["good_rate"]
        if score > best and it > 20:
            best = score
            torch.save({"policy": pol.state_dict(), "iter": it, "good_rate": score,
                        "obs_dim": OBS_DIM, "act_dim": ACT_DIM}, out / "best.pt")
        if it % 100 == 0:
            torch.save({"policy": pol.state_dict(), "iter": it}, out / "last.pt")
    fh.close()
    json.dump({"iters": a.iters, "best_good_rate": best, "minutes": (time.time()-t0)/60},
              (out / "summary.json").open("w"), indent=2)
    print(f"done: best good_rate {best:.4f} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
