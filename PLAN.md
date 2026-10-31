# PHENEI GNC Gym — build plan

Status: draft 1, 31.10.2026. Derived from Ilya's brief in `README.md` and the MCR report.
Parameters referenced as `SIM-xx` live in `design_docs/params.csv`; decisions as `TR-SIM-xx` in
`design_docs/trades.csv`.

---

## 1. Answers to the open questions

### 2D or 3D → **3D (6-DOF)**, with a 3-DOF scaffold used only to debug the RL loop

Ilya's instinct is right that the marginal complexity is modest — the dynamics are a free rigid
body, and the real added work is the thruster-allocation layer, which is itself one of the
deliverables. More decisively: **a 2D sim cannot answer either primary engineering question.**
Thruster placement is about torque arms and yaw authority; sensor placement is about boresights,
fields of view and occlusion. Both vanish in the plane.

So: build the 3-DOF planar env first (Phase 1) because it reproduces the MCR App. D optimum exactly
and therefore validates the integrator, the reward and the PPO plumbing against a known answer —
then retire it. See `TR-SIM-01`.

### Framework → **custom batched 6-DOF env in PyTorch + `rsl_rl` PPO**, MuJoCo for touchdown, Rerun for telemetry

The recommendation, concretely:

| Layer | Choice | Why |
|---|---|---|
| Env dynamics | **own 6-DOF rigid-body integrator, batched PyTorch tensors on CUDA**, RK4 at 400 Hz | Mass drops 835 → 500 kg (−40 %), which physics engines handle badly; every term is ours so every assumption gets a `params.csv` row; thousands of parallel envs; stays on-GPU with the policy; and writing the EOM *is* the aerospace norm (POST2, DSENDS) |
| Env API | **`rsl_rl` `VecEnv` contract** for training + a thin **`gymnasium.Env`** single-env wrapper | The `VecEnv` contract is exactly what Isaac Lab exposes, so the code stays portable to Isaac later; the Gymnasium wrapper gives the "OpenAI-style gym" you asked for, plus SB3 cross-checks |
| RL | **`rsl_rl`** (ETH RSL's PPO) primary, **`stable-baselines3`** as debug cross-check | `rsl_rl` *is* the RSL way and is ~2k readable lines, so studying it is part of goal 2. If SB3 and `rsl_rl` disagree on the same env, the bug is ours |
| Touchdown / legs | **MuJoCo** (separate low-env-count study) | Honeycomb crush and slope tip-over come nearly free; not worth hand-coding |
| Classical baseline | **`cvxpy`** convex powered-descent (lossless convexification) + closed-form ZEM/ZEV | A verified optimal bound is what makes an RL result credible, and it cross-validates our dynamics against MCR App. D |
| Telemetry + 3D | **Rerun** (`rerun-sdk`) | Purpose-built for this deliverable: synchronised 3D scene + time-series panels + camera images on one scrubbable timeline. Current de-facto standard in robotics |
| Scene / camera render | **MuJoCo offscreen renderer** → mp4 via `imageio-ffmpeg` | Gives RGB *and* depth, so it doubles as the camera/lidar renderer for validation frames |
| Analysis plots | **matplotlib** | MC scatter, landing ellipse / CEP, dispersion histograms |
| Env & deps | **`uv`**, Python 3.11 | Fast, lockfile committed (`uv.lock` is deliberately not gitignored) |

**On OpenCV**: not the industry standard for this, and not the right tool — it has no scene graph,
camera model or timeline. Keep it only for frame compositing if `ffmpeg` is not enough. The current
standards for robotics telemetry visualisation are **Rerun** and **Foxglove**; Rerun is the better
fit here because it needs no ROS.

**On Isaac Lab / the ETH RSL way**: rejected as the backend, *not* as the method. The 2080 Super is
Turing with 8 GB — at or below the Isaac Sim floor — and on WSL2 the install is fragile. More
importantly Isaac's strength is contact-rich locomotion, and this vehicle is in free flight until
the last 10 ms. But the RSL *method* — thousands of parallel envs, aggressive domain randomisation,
PPO via `rsl_rl`, a fast low-level controller tracking learned targets — transfers completely to a
batched PyTorch env. You get the workflow without the install. See `TR-SIM-02`, `TR-SIM-03`.

**On the control architecture** (the most consequential decision, `TR-SIM-04`): **hierarchical**.
RL guidance at 10 Hz commands thrust magnitude and attitude; a classical quaternion-feedback
controller at 100 Hz plus a thruster-allocation layer tracks it. This keeps episodes to a few
thousand steps instead of ~20 k, keeps the inner loop deterministic and verifiable, and — critically
— makes thruster placement a *studiable parameter* in the allocation matrix rather than something
the network silently absorbs. It is also the standard architecture in the RL-landing literature
(Gaudet & Furfaro, 6-DOF planetary landing) and the RSL pattern.

### Starting point, and how the Monte Carlo fits

Build in this order; each stage is useless without the previous one:

```
Phase 1  3-DOF planar env  →  validate against MCR App. D optimum
Phase 2  6-DOF dynamics + allocation + attitude controller  →  classical guidance lands it
Phase 3  RL guidance on ground-truth state  →  policy lands it, beats ZEM/ZEV, approaches convex bound
Phase 4  sensor models + EKF in the loop  →  policy lands on estimated state
Phase 5  domain randomisation + MC campaign  →  landing-accuracy statistics vs requirement
Phase 6  placement / architecture studies  →  the engineering deliverables
Phase 7  video
```

The Monte Carlo is **not** only at the end. Domain randomisation during training (Phase 5) and the
MC verification campaign are the same machinery with different purposes — randomise during training
so the policy is robust, then run a frozen 2000-case campaign (`SIM-N06`) for the numbers that go in
the report. Build the dispersion sampler once, in Phase 5, and use it for both.

### What to fix in stone now

Fixed (do not renegotiate without updating `params.csv`):

- **Conventions**: SI; MCI inertial / landing-site ENU / body +Z up; quaternion scalar-last
  (`SIM-N04`, `SIM-N05`).
- **Rates**: 400 Hz integrator, 100 Hz attitude loop, 10 Hz guidance (`SIM-N01`–`N03`).
- **Episode**: starts at braking-burn ignition, 39 km, 1484 m/s, −22.0° FPA, 835 kg
  (`SIM-A01`, `SIM-A05`–`A08`, `SIM-M03`). Ends at touchdown. No ballistic hop.
- **Geometry and mass**: `SIM-G*`, `SIM-M*` as listed.
- **Success criteria**: 50 m radius, `|v_z| ≤ 2.0 m/s`, `|v_xy| ≤ 0.5 m/s`, tilt ≤ 5°, rates ≤ 2°/s
  (`SIM-T01`–`T05`). Four of those five are LLM assumptions because GNC-08 is TBD — they are the
  first thing to get real values for.

Deliberately left open: engine minimum throttle (`SIM-P04`, swept not fixed), the velocity-sensing
config (`TR-SIM-10` — the sim exists to decide it), and the deep-throttle resolution (`TR-SIM-09`).

### How many thrusters does classical control need?

Minimum actuation, from the rank of the control-effectiveness matrix:

- **4 main engines at the 4 sides, all thrusting along body −Z**, produce only `{F_z, M_x, M_y}` —
  **rank 3** of the 6 generalised forces. The fourth throttle combination (+,−,+,−) is in the null
  space: for symmetric placement it produces no net force and **no yaw torque at all**.
- Lateral translation comes from tilting the vehicle (underactuated, exactly like a quadrotor).
  That is fine and is what every lander does.
- **Yaw is not controllable by the main engines.** Some RCS is therefore not optional. Minimum for
  ±yaw as a pure couple is 4 thrusters; for independent 3-axis attitude authority with the mains
  off, the textbook minimum is 12 (6 couples × 2).
- Baseline adopted: **4 mains + 8 cold-gas RCS in 4 canted pods** on the upper deck (`SIM-G14`–`G16`,
  `TR-SIM-08`). Authority check at dry mass: differential throttle gives 210 N·m at 150 N
  differential → 1.6 rad/s² in pitch/roll; an RCS pure couple of 2 × 22 N at 0.85 m gives 37 N·m →
  0.29 rad/s² in yaw. Pitch/roll is ample; yaw is adequate but should be swept.

**The finding that matters most, though, is not the count — it is the throttle floor.** Hovering at
touchdown mass needs 812 N, i.e. **13.5 % of the 6 kN `T_max`**. Four engines at a realistic 30 %
floor produce 1800 N. **The baseline cannot hover on four engines** (`SIM-P12`, `TR-SIM-09`). The
three ways out — deep-throttle engines, shutting down two of four, or letting a bipropellant RCS
tier fly the terminal descent (which is what MCR §5.2.3 already implies) — are a genuine question
for Mattia, and the sim should quantify all three rather than assume one.

---

## 2. Phase plan

### Phase 0 — scaffolding
- `uv` project, Python 3.11, CUDA PyTorch; verify GPU visible under WSL2.
- `phenei_gym/params.py`: load `design_docs/params.csv`, expose typed access by ID, **fail loudly on
  a missing ID**, and emit a provenance report (how many values are `Assumption (LLM)`).
- Repo layout:
  ```
  phenei_gym/{params,dynamics,actuators,sensors,nav,guidance,envs,viz}/
  scripts/{train,evaluate,montecarlo,make_video}.py
  studies/            # one subdir per Phase 6 study, each writing a .csv result
  tests/
  ```
- `pytest` from the start: energy/momentum conservation in free flight, RK4 convergence order,
  frame round-trips, quaternion normalisation.

### Phase 1 — 3-DOF planar env *(validation scaffold)*
- Implement MCR App. D dynamics exactly: state `(r, θ, ṙ, θ̇, m)`, control `(T, β)`.
- Reproduce the published numbers: Δv 3049.7 m/s impulsive, 3240.6 m/s multiphase, apoapsis 360 km.
  **If these do not reproduce, the dynamics are wrong — stop and fix.**
- Convex powered-descent baseline in `cvxpy`; compare fuel against the multiphase optimum.
- Gymnasium env + SB3 PPO. Goal: a policy that lands within a few percent of the convex optimum.
  This exists to prove the reward and the plumbing, nothing more.

### Phase 2 — 6-DOF dynamics and the inner loop  *(COMPLETE, 31.10.2026)*

**Done**: rigid-body EOM with quaternion kinematics (`dynamics/rigid6dof.py`), verified against the
3-DOF polar model to <1 m over 200 s; actuator layout and control-effectiveness matrix built from
the `SIM-G*` rows (`actuators/layout.py`); weighted control allocation; quaternion-feedback
attitude controller (`guidance/attitude.py`). 31 tests pass.

**Result**: the 6-DOF descent lands. From 75 km at 950 kg with a 90 % throttle cap: 409 kg
propellant, touchdown at **0.99 m/s vertical, 0.00 m/s horizontal, 0.00 deg/s, 5.05° tilt**. Four
of the five touchdown criteria pass with margin; tilt sits exactly on the `SIM-T04` limit, which is
itself an LLM assumption since GNC-08 is TBD.

Peak tilt is 64–66°, confirming `SIM-A16` from the independent 6-DOF path.

- Rigid-body EOM, quaternion kinematics, batched over `N` envs on GPU.
- Variable mass and inertia (`SIM-M09`), thrust misalignment and CoM offset as per-env constants.
- **Actuator model**: 4 throttles with floor, slew limit and ignition transient; 8 RCS with minimum
  on-time; build the control-effectiveness matrix `B` from the geometry rows so moving a thruster is
  a parameter change, not a code change.
- **Allocation**: pseudo-inverse with a non-negative least-squares / LP fallback respecting the
  throttle floor. This is the layer the placement studies interrogate.
- Quaternion-feedback attitude controller at 100 Hz; verify step responses and limit cycles.
- Acceptance: the Phase 1 convex trajectory, flown open-loop in 6-DOF with the inner loop closed,
  lands within the touchdown box on a flat surface.

### Phase 3 — RL guidance on ground-truth state
- Observation: altitude, position and velocity in landing-site ENU, attitude error, rates, mass,
  propellant remaining, time-to-go estimate.
- Action: thrust magnitude + commanded attitude (2 or 3 DOF). Continuous, `tanh`-squashed.
- Reward: dense shaping on tracking a convex/ZEM-ZEV reference, plus terminal terms on the five
  touchdown criteria and a propellant penalty. Start from the literature's shaping, not from scratch.
- `rsl_rl` PPO, 4096 envs. Curriculum per `TR-SIM-06`: 50 m → 2 km → 8 km → 39 km.
- Acceptance: beats ZEM/ZEV on fuel, within ~5 % of the convex bound, 100 % landing in nominal
  conditions.

### Phase 4 — sensors and navigation in the loop
- Sensor models with range gating, rates, bias, noise, latency and dropout, per `SIM-S*`. Crucially,
  **model availability**: no altimeter above 6–8 km, no star tracker in terminal descent, TRN fixes
  at 1 Hz with 100 m 1σ.
- Terrain as a randomised heightmap (`TR-SIM-12`); altimeters by GPU raycast, including the
  slope-induced bias.
- TRN as a synthetic position fix, **not** a rendered image — the 2080 Super cannot render 4096
  camera views per step. Render real images only for the video and a small validation set.
- MEKF fusing IMU / star tracker / TRN / altimeters. Report navigation error vs the `GNC-05`,
  `GNC-06` TBD budgets — **this quantifies requirements the team has not yet been able to set**.
- Retrain the policy on the filter estimate plus its covariance.
- Acceptance: lands on estimated state; the altimeter-free 39 → 8 km stretch is characterised.

### Phase 5 — domain randomisation and the MC campaign
- One dispersion sampler driving: initial state (`SIM-A15`, placeholder until TR-08 lands), mass and
  inertia, CoM offset, thrust magnitude and misalignment, all sensor error terms, terrain slope,
  roughness, hazard density, albedo.
- Train with randomisation on; then freeze the policy and run 2000 cases.
- Outputs: landing-radius CDF and 99th percentile vs the 50 m requirement, touchdown-limit violation
  rates, propellant 99th percentile, failure taxonomy. Written to `studies/*.csv`.

### Phase 6 — the engineering studies *(the actual deliverables)*
Each is a sweep over `params.csv` rows, each writes a CSV:
1. **Velocity sensing** — Config 1 vs Config 2 (`TR-SIM-10`). Does direct Doppler velocity justify
   3 kg? This closes an open MCR decision.
2. **Throttle floor** — sweep `SIM-P04` from 10 % to 40 % across the three `TR-SIM-09` options.
   Feeds straight back to propulsion as a requirement on P-13.
3. **Thruster placement** — sweep main-engine radius `SIM-G08`, RCS count, pod radius and cant.
   Report the attitude authority and propellant cost of each.
4. **Sensor placement** — altimeter cant angles and camera FOV split (`SIM-S10`/`S11`) vs navigation
   error and landing accuracy.
5. **Touchdown limits** — invert the question: given the achievable control, what `GNC-08` values are
   actually feasible? The structures team needs this and it is currently TBD.
6. **Noise tolerance** — scale all sensor noise until the requirement breaks; report the margin.
7. **HDA trigger altitude** — at what altitude must hazards be detected and committed, given divert
   capability and perception latency (`TR-SIM-13`)? Output: a required trigger altitude and divert
   propellant reserve, or the finding that orbital pre-screening is cheaper.
8. **Sensor cant angle** — sweep descent-sensor boresight cant against the ≈68° pitched attitude the
   vehicle actually flies (`TR-SIM-14`). Output: required cant, and the real altimeter acquisition
   altitude.
9. *(stretch)* end-to-end policy from raw sensors vs the EKF architecture (`TR-SIM-07`).

### Phase 2.5 — presentable simulation visuals  *(ASAP, requested by Ilya 31.10.2026)*

Pulled forward ahead of the RL work: the demo needs to look like a lunar landing, not a wireframe
on a blank grid. Everything here is `viz/rerun_log.py` plus a small terrain module, and none of it
depends on Phase 2 being finished, so it can proceed in parallel.

1. **Lunar surface at the landing site.** Replace the single grey line with a textured terrain
   mesh. Build the heightmap from `SIM-E06`/`SIM-E07` (slope, RMS roughness) plus craters and
   rocks at `SIM-E08` density, log it as `rr.Mesh3D` with per-vertex colours shaded by local
   normal and `SIM-E09` regolith albedo. This is the same heightmap the altimeter raycasts will
   use in Phase 4, so it is not throwaway demo work — build it as `phenei_gym/terrain.py` and have
   the viz read it. Optional upgrade: drape real LRO NAC imagery (0.5 m/px, cited in MCR §5.3.5)
   over the mesh for the hero shot.
2. **Draw the thrusters properly.** The 4 main nozzles and 8 RCS pods already exist in
   `hopper_wireframe()` as line strips. Promote them to real geometry: nozzle bells as cones of
   `SIM-G13` exit diameter, RCS pods as small boxes at `SIM-G15`/`SIM-G16`.
3. **Main-engine plumes that fire with the actual commanded throttle.** One cone per engine,
   logged per frame, length and width scaling with that engine's throttle from the allocation
   vector, so the plume *is* the telemetry — you can see differential throttle working during a
   pitch manoeuvre. Hot white-yellow core grading to orange, alpha falling off with distance. Zero
   throttle must render nothing, not a stub. Note MON-3/MMH burns nearly invisibly in reality;
   this is deliberately a readability choice, worth stating in the video.
4. **Cold-gas RCS puffs.** Same mechanism, different look: short, narrow, pale blue-white, and
   **on/off rather than proportional** — `SIM-P11` gives a 20 ms minimum on-time against a 10 ms
   control period, so the RCS physically cannot modulate within a cycle. Rendering them as
   discrete puffs is both prettier and more honest than a continuous jet.
5. **Scale handling.** The existing split stays (`world/` in km, `vehicle/` in m) because the
   trajectory is ~1000 km and the vehicle is 2 m. Add a third `closeup/` entity: vehicle plus the
   local terrain patch in metres, which is the view the plumes and touchdown actually read in.

Acceptance: a single `.rrd` that can be scrubbed from ignition to touchdown showing the hopper over
textured terrain, pitched ~65° with four plumes lit, rotating upright as it descends, RCS puffing
during attitude corrections, and the plumes visibly throttling down at the flare.

### Phase 7 — video
Rerun timeline with the 3D vehicle, plume, terrain and landing ellipse; sensor telemetry panels
(altimeter returns with range gating visible, TRN fixes popping in, EKF covariance shrinking);
throttle and RCS activity; a small MC fan of 50 trajectories converging on the target. MuJoCo
offscreen for the vehicle and camera views. Blender only if a hero shot is wanted.

---

## 3. Immediate risks

| Risk | Mitigation |
|---|---|
| `SIM-P04` throttle floor makes the baseline unable to hover (`TR-SIM-09`) | Raised to Mattia now; sim sweeps it rather than assuming |
| 52 of the 107 `SIM-*` rows are `Assumption (LLM)` | Every one is marked and greppable; prioritise review of `SIM-T02`–`T05` (touchdown limits), `SIM-S03`/`S04`/`S06` (IMU noise — check the Safran datasheet) and `SIM-P04` |
| Dispersion analysis TR-08 is not done, so `SIM-A15` is unknown | Parameterise initial dispersion; report sensitivity to it instead of a single answer |
| 8 GB GPU | Batched custom env with MLP policies, no rendering in the training loop |
| Reward shaping eats the schedule | Phase 1 exists precisely to settle shaping against a known optimum before 6-DOF |
| Deck-separation interpretation (`SIM-G05`) and nozzle/ground-clearance interference (`SIM-G13`) | Both flagged to Lorenzo (STR) |

## 4. Open items for the team

- **Mattia (PRP)**: P-13 minimum throttle — the single most consequential unknown. Also: is the RCS
  tier bipropellant and ~800 N capable, as §5.2.3 implies?
- **Lorenzo (STR)**: P-20 touchdown loads; confirm `SIM-G05`; rescale `Assembly_STR.x_t`; nozzle
  envelope vs the 0.6 m ground clearance.
- **Nicola (GNC)**: Safran STIM300 datasheet noise figures; Argus camera FOV and rate; Config 1
  velocity-accuracy estimate.
- **Freddy (MA)**: confirm the 39 km ignition altitude and resolve the 245 km vs 360 km apex
  discrepancy; provide the TR-08 injection dispersion when available.
- **Ilya (SE)**: resolve the GNC-01 vs S1.1.3/P-08 landing-accuracy conflict (50 m vs 100 m radius).
