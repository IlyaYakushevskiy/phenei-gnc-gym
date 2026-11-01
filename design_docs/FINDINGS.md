# PHENEI GNC simulation — findings for the subsystem teams

Status: Phases 0–5 of `PLAN.md`. Every number below is traceable to a row in
`params.csv` or `trades.csv`; the IDs are given so you can check the provenance
and the confidence marker yourself. Anything marked `Assumption (LLM)` in those
files is a placeholder I invented to make the simulation run and **needs a human
number** — those are listed in §7.

Validation gate: `scripts/validate_mcr.py` is 15/15. It re-derives the MCR's own
published Δv (3049.7 → 3050.1 m/s), flight time (41.1 → 41.07 min) and apoapsis
(360 → 359.8 km) from the dynamics, so the trajectory model agrees with mission
analysis before any of the following is claimed.

---

## 1. The braking-burn ignition altitude is infeasible as written

**`SIM-A01`, `SIM-A20`, `SIM-A21`, `TR-SIM-18`** — for Freddy (MA) and Mattia (PRP).

MCR §5.3.5 puts braking ignition at **39 km**. The minimum-energy arc arrives
there descending at 557 m/s with 1376 m/s of horizontal velocity to remove.
That leaves ~140 s before ground contact against the ~200 s the burn needs.

Required thrust at 39 km is **9.36 kN** — 1.56× the top of the 3–6 kN range.

Ignition has to move up, and it scales roughly as 1/thrust:

| Thrust | Ignition altitude |
|---|---|
| 3 kN | 138 km |
| 4 kN | 96 km |
| 5 kN | 75 km |
| 6 kN | 62 km |

**Two of the MCR's own numbers already agree with this**: the stated 202 s burn
duration implies ≈55 km, and 6 kN implies ≈62 km. Only §5.3.5's 39 km disagrees,
and that figure is marked TBC.

The simulation uses **60 km** (`SIM-A19`) and lands inside the touchdown box
using 368 kg of propellant in 222 s.

---

## 2. The vehicle flies nearly sideways, and the sensor suite assumes it doesn't

**`SIM-A16`–`A18`, `SIM-S43`, `SIM-S44`, `TR-SIM-14`** — for Nicola (GNC).

Because the burn is overwhelmingly a horizontal-velocity kill, the vehicle holds
**58–68° from local vertical for 78 % of the burn**, spiking to 86° at ignition
and dropping below 45° only at ~2.5 km. This is simulated, not estimated —
measured on the flown quaternion.

Consequences for every body-fixed downward-looking sensor:

- **Altimeters measure slant range, not altitude.** At 60° the slant path is 2×
  the altitude, so a nadir-mounted DLEM 30 first acquires at **4188 m**, not the
  6–8 km of §5.3.5. The altimeter-free stretch is therefore **60 km → 4.2 km**,
  far worse than the 39 → 8 km the report lists as an open risk.
- **TRN degrades rather than failing.** The surface fills a cone of
  arcsin(R/(R+h)) about nadir, so an oblique camera still gets a fix — but the
  ground footprint stretches by 1/cos(incidence). At the flown attitude the
  100 m heritage accuracy (`SIM-S13`, from Blue Ghost M1) inflates to **212 m**.
  The heritage number does not transfer, because Blue Ghost flew nadir-pointing.

**Mitigation, quantified.** Canting the downward boresights **aft** (the vehicle
brakes retrograde, so its thrust axis leans backward):

| Boresight cant | TRN incidence at the flown attitude | TRN 1σ | First altimeter fix |
|---|---|---|---|
| 0° | 58–62° | 190–212 m | 4188 m |
| 30° | 28–32° | 113–118 m | — |
| 45° | 13–17° | 103–104 m | 7768 m |
| 60° | 0–2° | 100 m | 7968 m |

A 60° aft cant restores the full heritage accuracy and **nearly doubles the
altimeter acquisition altitude**, buying ~3.8 km and ~40 s of altimeter-aided
navigation. This costs nothing but a mounting bracket, and it is the single
cheapest navigation improvement available.

---

## 3. Navigation error budget — GNC-05 and GNC-06 are currently TBD

**`scripts/run_nav.py`** — for Nicola (GNC) and Ilya (SE).

A 15-state MEKF (IMU + star tracker + TRN + altimeters + velocity) replayed
through the reference descent. The filter is deliberately given a **flat**
surface model, which is what a real onboard filter knows, so terrain slope shows
up as the measurement bias it really is.

Single-seed, Config 1 (camera feature tracking):

| | first alt fix | touchdown position error | velocity error |
|---|---|---|---|
| no cant | 4188 m | 15.2 m | 0.07 m/s |
| 45° aft cant | 7768 m | 4.8 m | 0.09 m/s |
| 60° aft cant | 7968 m | 11.9 m | 0.07 m/s |
| no cant, real terrain | 4198 m | **43.5 m** | 0.08 m/s |
| 60° cant, real terrain | 7979 m | 25.2 m | 0.19 m/s |

**Terrain slope roughly triples the touchdown position error.** At 43.5 m that
is most of the 50 m accuracy budget consumed by navigation alone, before any
guidance error. Canting the boresights halves it.

The filter is consistent: the error sits inside its reported 1σ in every
altitude band, and the position σ grows monotonically through the
measurement-free coast, which is the correct behaviour.

### Candidate values for GNC-05 and GNC-06 (20 seeds, flat surface)

| | position error (mean / 95th) | velocity error (mean / 95th) | attitude error |
|---|---|---|---|
| Config 1, no cant | 18.6 / 28.0 m | 0.23 / 0.42 m/s | 0.03° |
| Config 1, 60° cant | 14.0 / 24.8 m | 0.20 / 0.34 m/s | 0.02° |
| Config 2, no cant | 17.0 / 29.8 m | 0.01 / 0.01 m/s | 0.02° |
| Config 2, 60° cant | 11.9 / 23.8 m | 0.00 / 0.01 m/s | 0.02° |

`SIM-NV01`–`NV06`. Add these to the guidance miss in quadrature, not linearly.

---

## 3a. Config 1 vs Config 2 — the trade turns on one number, and it is mine

**`TR-SIM-22`, `SIM-NV03`, `SIM-NV04`** — for Nicola (GNC).

The simulation settles what the argument is *about*:

- **Position accuracy is not a discriminator.** Config 1 and Config 2 give the
  same touchdown position knowledge (18.6 m vs 17.0 m). The landing-accuracy
  requirement `SIM-T01` does not distinguish them.
- **Touchdown velocity is the whole argument.** `SIM-T03` allows 0.5 m/s
  horizontal. Config 1's navigation error at the 95th percentile is **0.42 m/s
  — 84 % of the entire budget, consumed before any control error.** Config 2
  uses 2 %.

But this rests entirely on `SIM-S34`, the per-measurement camera velocity
accuracy, which **I invented** at 0.5 m/s. If the real figure is better than
~0.3 m/s, Config 1 is comfortable. If it is worse, Config 1 cannot meet
`SIM-T03` at all. **This is the single highest-value number you can give me**,
and the recommendation stays conditional until it exists.

---

## 4. Hazard avoidance is infeasible where the ConOps puts it

**`SIM-H01`–`H08`, `TR-SIM-13`** — for Nicola (GNC) and Ilya (SE).

ConOps phase 8 puts hazard detection and avoidance in the 6 s vertical descent
from 50 m. At 50 m:

- divert capability at 20° tilt is **5.3 m** (`SIM-H05`) against ~30 m needed to
  clear a typical hazard;
- the perception latency budget alone is **3.5 s of the 6 s** (`SIM-H07`).

HDA has to move to **300–500 m** (`SIM-H06`), or be replaced by orbital
pre-screening of the landing site against LRO NAC imagery. Note also §5.3.8:
there is **no scanning lidar in the baseline** — the DLEMs are single-beam
rangefinders and the Doppler LIDAR is the rejected Config 2 option. Anything
needing a point cloud needs new hardware.

---

## 5. Propulsion: the baseline cannot hover, and 6 kN is the worst corner

**`SIM-P04`, `TR-SIM-09`, `TR-SIM-19`** — for Mattia (PRP).

- Hovering at touchdown mass needs **13.5 % of 6 kN**. A pressure-fed
  bipropellant engine typically floors at ≈30 %. Four engines at 30 % = 1800 N
  against 812 N of weight: **the baseline cannot hover on four engines.** The
  simulation lands by shutting down a diametrically opposite pair; which pair
  matters, and picking adjacent engines tumbles the vehicle.
- **6 kN is the uniquely bad corner of the 3–6 kN range**: it is high enough to
  make the throttle floor bite at touchdown while buying little at ignition.
- **Differential-throttle authority collapses 36× at full throttle**, from
  8.45 to 0.234 rad/s² (`TR-SIM-19`), because the throttles are already near
  their upper bound and cannot be pushed further apart. The simulation reserves
  10 % headroom (`SIM-GD07`) for this reason.
- A 5 % thrust shortfall alone drives touchdown to 2.01 m/s horizontal and 7.4°
  tilt, **failing both `SIM-T03` and `SIM-T04`**. Engine acceptance tolerances
  are needed.

**Open question for Mattia:** P-13 minimum throttle is the single most
consequential unknown in this model.

---

## 6. Actuation: yaw is the binding limit

**`SIM-N10`, `SIM-G08`** — for Lorenzo (STR) and Mattia (PRP).

Yaw authority is **4.9 % of pitch/roll** (52.9 N·m against 1081 N·m). Four
parallel main engines produce **no yaw torque at all**, so yaw comes entirely
from the 8 cold-gas thrusters. This is the binding actuation limit of the design.

A related modelling trap worth recording: the RCS layout initially could only
yaw in **one direction** despite the control-effectiveness matrix being full
rank. With non-negative throttles the reachable set is a *cone*, not a subspace,
and rank tells you nothing about it. Fixed by alternating the tangential sense
of adjacent pods.

Also for Lorenzo: a 1.5 kN engine at ε≈60 has a ≈0.25 m exit diameter and ≈0.4 m
length, which consumes most of the 0.6 m deck-to-surface clearance (`SIM-G13`).
Interference risk.

---

## 7. What I made up and you need to replace

`grep 'Assumption (LLM)' design_docs/params.csv` is the authoritative list.
The ones that actually change conclusions:

| Row | What | Why it matters |
|---|---|---|
| `SIM-P04` | Minimum throttle | Decides whether §5 is a real problem |
| `SIM-S34` | Config 1 velocity accuracy (0.5 m/s) | `SIM-T03` demands 0.5 m/s at touchdown, so if this is right the requirement cannot be *verified* with Config 1, let alone met. Pivotal for `TR-SIM-10`. |
| `SIM-S03`, `S04`, `S06` | STIM300 noise figures | Whole navigation budget. Check the Safran datasheet. |
| `SIM-S43` | TRN max usable incidence (70°) | The vehicle flies at 58–68°, i.e. right on this threshold |
| `SIM-MC01`–`MC08` | Injection and engine dispersions | TR-08 has not been run, so `SIM-A15` is unknown and the Monte Carlo sweeps a scale factor instead of claiming an answer |
| `SIM-T02`–`T05` | Touchdown limits | Everything is scored against these |

**Document conflicts flagged, not silently resolved:**

- Landing accuracy: GNC-01 says a 100 m *diameter* circle (50 m radius); S1.1.3
  and P-08 say 100 m *radius*. The sim assumes the stricter 50 m (`SIM-T01`).
  **Ilya owes a resolution.**
- Apex altitude: 245 km in the ConOps figure vs 359.8 km for the analytic arc.

---

## 8. On the reinforcement-learning half

**`SIM-RL01`–`RL12`, `TR-SIM-15`, `TR-SIM-21`.**

This is the learning goal of the project, not an engineering deliverable, and
the distinction is load-bearing: **no flown spacecraft uses RL for powered
descent**, and none of §1–7 depends on RL working.

Where it stands: PPO from scratch plateaus at 0.1 % soft landings after 9.7×10⁹
environment steps — an exploration failure, since a successful touchdown is a
measure-zero event at random initialisation. Behaviour-cloning the classical
guidance reaches 31.7 %, and PPO fine-tuning on top reaches **76.1 %** soft
landings at 3.41 m/s mean impact.

It does **not** beat the classical guidance, which lands at 21 m and 0.98 m/s
meeting all five touchdown criteria. The learned policy's mean miss is ~95 m.

The honest summary of the RL effort so far is that essentially every failure was
infrastructure rather than algorithm — the action-to-thrust mapping, reward
structure, an untrained critic at the BC→PPO handoff, exploration noise scale.
Those are written up in `TR-SIM-21` and `SIM-RL04`/`RL10`/`RL11` so the next
person does not rediscover them.
