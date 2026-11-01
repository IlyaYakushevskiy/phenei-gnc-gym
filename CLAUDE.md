# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **Gymnasium-style simulation of the powered descent and landing of the PHENEI Last-Mile Hopper**,
a lunar surface-to-surface delivery vehicle designed by an ETH Zurich project team (SharePoint site
`ETHLunarDeliveryHopper`). Owner: Ilya (GNC / systems engineering).

Three goals, in priority order:

1. **Engineering**: answer design questions the MCR left open — thruster placement and count, sensor
   placement and architecture, how much sensor noise the GNC loop tolerates, and whether the landing
   accuracy requirement is met. The sim exists to produce numbers the subsystem teams can use.
2. **Learning**: Ilya learning RL methods as applied to spacecraft GNC, and current robotics practice
   (ETH RSL style massively-parallel RL) alongside classical aerospace GNC practice.
3. **Deliverable**: a video showing the hopper flying the landing manoeuvre under a trained policy,
   with live sensor telemetry overlaid, plus a Monte Carlo campaign (sensor errors drawn within
   spec, randomised landing-surface parameters).

**Scope**: the ballistic hop is *not* simulated. The episode starts at the **approach / braking-burn
ignition** and ends at touchdown — ConOps phases 7 and 8 only.

## Repository status

Phases 0–5 are done, plus most of 6 and 7. `design_docs/FINDINGS.md` is the readable summary of
what the simulation has actually established — **read it before starting anything new**, because
several of the MCR's stated numbers have been shown to be infeasible and the results build on
each other.

| Phase | State |
|---|---|
| 0–1 param loader, 3-DOF dynamics, reference descent, Rerun | done; `validate_mcr.py` 15/15 |
| 2 6-DOF hierarchical stack | done; lands meeting all five touchdown criteria |
| 2.5 presentable visuals (terrain, nozzles, plumes) | done except the `closeup/` entity |
| 3 RL | done to a defensible result: BC+PPO peaks at 85.2 % soft then collapses (`SIM-RL20`); beats its own expert in-env but not the touchdown requirement |
| 4 sensors + MEKF, closed-loop navigation | done |
| 5 Monte Carlo campaign | done; produced the decisive negative result (`TR-SIM-23`) |
| 6 engineering studies | engine-tolerance + sensor-noise curves done; thruster-count study not started |
| 7 video | `studies/phenei_landing.rrd` built and verified |

PyTorch 2.15.0+cu132 is installed and **CUDA works**: RTX 2080 SUPER, capability 7.5.

```bash
python3 -m venv .venv                                   # already created
.venv/bin/python -m pip install numpy scipy matplotlib pytest rerun-sdk

.venv/bin/python scripts/validate_mcr.py                # the gate: MUST be 15/15
.venv/bin/python scripts/params_report.py               # provenance of params.csv
.venv/bin/python -m pytest tests/ -q                    # 94 tests, ~8 min
.venv/bin/python -m pytest tests/test_descent.py -q -k touchdown   # single test

.venv/bin/python scripts/run_descent.py                 # 3-DOF reference -> .rrd + .png
.venv/bin/python scripts/run_descent6dof.py             # 6-DOF, aligned to the site
.venv/bin/python scripts/run_nav.py --cant 60 --seeds 20   # navigation error budget
.venv/bin/python scripts/mc_campaign.py --cases 48 --x0 -173.550
.venv/bin/python scripts/noise_study.py --x0 -173.550   # sensor-noise tolerance
.venv/bin/python scripts/make_video.py --x0 -173.550    # the deliverable recording
scripts/view.sh <file.rrd>                              # view a recording (see note)
```

**Always pass `--x0 -173.550` where a script accepts it.** Solving the nominal ignition point
takes five full 6-DOF descents (~9 minutes under load); that value is the converged answer for
the 6 kN / 60 km baseline and re-solving it wastes most of a turn.

### Things that will bite you

- **A single 6-DOF descent is ~89 000 RK4 steps and takes 20–90 s.** Anything that flies many of
  them needs `ProcessPoolExecutor`, and anything that re-flies the *same* trajectory should cache
  it (`scripts/run_nav.py` does; it was re-flying it once per Monte Carlo seed).
- **Never use `pkill -f <pattern>` or `pgrep -f <pattern>`** where the pattern also matches this
  shell's own command line. It kills the shell mid-command (exit 144) and silently loses
  heredoc-written files. Use `ps -eo pid,cmd | grep 'name[.]py'` and kill by PID.
- `params.csv` values carry human annotations like `0.5 (1 sigma)`; `Param.as_float()` strips a
  trailing parenthetical. Compound entries like `1 m + 0.1% of range` are **not** parseable — they
  are split into scalar rows (`SIM-S36`/`S37`) for code to read.
- Rows `P-xx` in `params.csv` are the team's and **must not be edited**.

`validate_mcr.py` is the gate: it reproduces the MCR's published delta-v, flight time and apoapsis
from the dynamics, checks conservation and RK4 convergence order, and confirms `SIM-A05`-`A08`.
Run it before trusting any result. Heavy dependencies (torch, gymnasium, SB3, mujoco) are declared
in `pyproject.toml` under the `rl` and `render` extras and are **not** installed yet.

Rerun recordings are written as `.rrd` files rather than streamed, because that is the transport
that survives a tunnel with no port forwarding and makes a run replayable.

**Viewing on this machine (WSL2): use the web viewer, via `scripts/view.sh`.** WSLg is present and
the native window does open with `DISPLAY=:0`, but there is no `dzn`/D3D12 Vulkan ICD installed,
so it falls back to `llvmpipe` software rasterisation and is slow. The web viewer renders in the
Windows browser with real GPU acceleration. Two ports: 9090 (viewer app) and 9876 (gRPC data), and
the URL needs the `?url=` query parameter the CLI prints — bare `localhost:9090` loads the app with
no data source attached.

**GPU**: the RTX 2080 SUPER *is* visible to WSL2 for compute — `/dev/dxg` and
`/usr/lib/wsl/lib/libcuda.so` are present, `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi`
(not on `PATH`), 8192 MiB, driver 591.74, compute capability 7.5. The software-rendering fallback
above is a *graphics* limitation only and does not affect CUDA, so PyTorch training in Phase 2 is
unaffected.

## Non-negotiable working rule: document every assumption

This is the part of the project Ilya called out as very important, and it takes precedence over
moving fast.

- **Every physical number used by the simulation lives in `design_docs/params.csv`** — nozzle
  diameters, noise densities, inertias, touchdown limits, all of it. Code must not contain a
  magic physical constant; it reads the CSV (or a generated module) and refers to the row by ID.
- **Every close-call decision gets a row per option in `design_docs/trades.csv`** — sensor
  architectures, RCS layouts, framework choices. If you find yourself picking between two
  defensible options, write the trade down before writing the code.
- The `Status` column is the confidence marker. Controlled vocabulary:

  | Status | Meaning |
  |---|---|
  | `Requirement` | Traceable to an L1–L3 requirement or a GNC-xx/AVI-xx requirement |
  | `Doc (MCR)` | Stated explicitly in the MCR report; cite section |
  | `Given (Ilya)` | Specified by Ilya for this simulation |
  | `Derived` | Computed from the above — **put the derivation in Notes** |
  | `Assumption (LLM)` | Invented by Claude to make the sim runnable. Needs human review. |
  | `TBD` | Genuinely unknown; do not guess silently |

- `Assumption (LLM)` is not a defect, but it must never be silently upgraded. If a real value
  arrives from the propulsion/structures/GNC team, change the Status and the Source together.
- `design_docs/params.csv` is a CSV export of the `Parameters` sheet of the team spreadsheet.
  Rows `P-xx` are the team's systems-engineering parameters — **do not edit those**; they round-trip
  with the ETH SharePoint workbook. Simulation parameters are added as `SIM-xx` rows with
  `Owner WP = GNC-SIM` so they filter out cleanly.

## Design docs and how to read them

| File | Contents |
|---|---|
| `design_docs/PheNei-report-MCR-local_copy.pdf` | 55-page MCR report. §5.3 Avionics & GNC (sensors, trades, equipment, per-phase sensor use), §5.7 Structures, §7.1 ConOps, §7.2 Hop trajectories, App. C full GNC/AVI requirement list, App. D trajectory-optimisation setup. |
| `design_docs/REQUIREMENTS+PARAMS+BUDGETS.xlsx` | `L1-L3` requirement tree, `Parameters` (P-01…P-28), `Budgets` (mostly TBD). |
| `design_docs/params.csv` | CSV export of `Parameters` + the `SIM-xx` simulation parameters. |
| `design_docs/trades.csv` | Trade log, one row per option. |
| `design_docs/Assembly_STR.x_t` | Parasolid export from Onshape of the primary structure. **Scale is wrong** — do not take dimensions from it; use the `SIM-G*` geometry rows. Untracked, 7 MB. |

`pdftotext` and `openpyxl` are not installed; `pypdf` is. Report page numbers match PDF page numbers.

```bash
python3 -c "
from pypdf import PdfReader
for i,p in enumerate(PdfReader('design_docs/PheNei-report-MCR-local_copy.pdf').pages,1):
    print(f'===== PAGE {i} ====='); print(p.extract_text())" > /tmp/mcr.txt
```

The `.xlsx` can be read by unzipping it and parsing `xl/worksheets/sheet{1,2,3}.xml` against
`xl/sharedStrings.xml` with `xml.etree`.

## The flight problem being simulated

### Approach conditions — derived, and validated against the report

The MCR gives the hop as a **minimum-energy two-impulse Keplerian arc**, 89° of central angle,
south pole to equator. Re-deriving that arc reproduces *both* numbers the report states
independently (Δv 3049.7 m/s → 3050.1; apoapsis 360 km → 359.8), so the arc is confirmed and the
approach state can be derived from it rather than guessed:

| At braking-burn ignition | MCR's 39 km | **Simulation's 60 km** (`SIM-A19`) |
|---|---|---|
| Speed | 1484 m/s | 1462 m/s |
| Flight-path angle below local horizontal | 22.0° | 21.6° |
| Descent rate / horizontal speed | 557 / 1376 m/s | 538 / 1359 m/s |
| Thrust required | **9.36 kN** — infeasible | 6 kN |
| Vehicle mass | — | 880 kg (dry 500 + 380 propellant, 368 consumed) |

Orbit eccentricity 0.419, semi-major axis 1477.6 km. **The 39 km figure is infeasible** — see the
conflicts below.

Almost all the velocity is horizontal — the braking burn is predominantly a horizontal-velocity
kill, as §7.1 describes ("retrograde, removes horizontal velocity").

### Phase structure (ConOps §7.1, reference durations)

| # | Phase | Reference duration | Notes |
|---|---|---|---|
| 7 | Braking burn | 202 s | 39 km → 50 m. Splits at ≈6–8 km where altimeters acquire. |
| 8 | Terminal descent | 6 s | vertical from 50 m, hazard detection, touchdown |

The hop is flown **fully autonomously on a preloaded trajectory**; ground can only command abort or
safe mode (AVI-23). No lunar navigation satellites are assumed at the equator.

### Dynamics the mission-analysis team uses

§7.2 / App. D use a **planar point-mass model in Moon-centred inertial polar coordinates**, solved
as a 5-phase optimal-control problem in CasADi + IPOPT. The 3-DOF development env must reproduce
it so results stay comparable:

- State `x = (r, θ, ṙ, θ̇, m)`, control `u = (T, β)`; `β` from the **local horizontal**
  (`β = 0` along `+θ`, `β = π/2` radially outward); `θ` from −89° to 0.
- `r̈ = (T/m)·sin β + r·θ̇² − μ/r²`;  `θ̈ = ((T/m)·cos β − 2·ṙ·θ̇)/r`;  `ṁ = −T/(Isp·g₀)`.
- Constraints: `0 ≤ r−R ≤ 400 km`, `|ṙ| ≤ 2000 m/s`, `|θ̇| ≤ 2000/R`, `0 ≤ T ≤ T_max`,
  `β = π/2` in the vertical phases.
- Assumptions kept from the report: non-rotating Moon, point-mass gravity, planar motion, no
  third-body terms, spherical surface, constant Isp.
- Assumptions the report flags as PDR work and **this project deliberately breaks**: no attitude or
  throttle rate limits, and constant Isp with no throttle dependence. Adding rate limits, actuator
  lag and 6-DOF attitude is precisely where a learned policy differs from the open-loop optimum —
  it is the reason this project exists.

### What the vehicle can observe (MCR §5.3.5)

Observation spaces must respect sensor availability, not assume full state:

- **Always**: 3 × Safran STIM300 IMU (MEMS, ≤2 kHz, majority voting). Star trackers (2, 4 Hz) off
  in terminal descent.
- **TRN** by monocular crater matching against preloaded maps, on 2 Redwire Argus cameras — narrow
  FOV high up, wide FOV low down. ≈100 m position accuracy (Blue Ghost M1 heritage, 50–20 km).
  Processing runs on the camera processors, not the OBCs.
- **Altimeters only work low**: DLEM 30 ≈6–8 km, DLEM 17 ≈3–4 km, radar altimeter 500 m. The
  braking burn starts at 39 km, so **39 km → ≈6–8 km is TRN + IMU only, with no altitude sensor.**
  The report lists this as an open risk; it is the most interesting region in the sim.
- **Horizontal velocity is indirect** in the baseline (Config 1): camera feature tracking fused with
  IMU and altimeters. Config 2 (Doppler LIDAR LUNA, +3 kg) measures it directly. The choice is open
  pending touchdown limits (IF-11) — **the sim is meant to settle it** (see `TR-SIM-10`).
- Heterogeneous stereo only in the last hundreds of metres (depth error grows as Z²), supporting
  hazard detection. **There is no scanning lidar in the baseline**: the DLEMs are single-beam laser
  *rangefinders*, and the Doppler LIDAR is the rejected Config 2 option. §5.3.8: "no lidar hazard
  mapping is included". Anything that needs a point cloud needs new hardware.

### Known conflicts in the source documents

Flag these rather than silently resolving them:

- **Landing accuracy**: GNC-01 says a 100 m *diameter* circle (50 m radius); S1.1.3 and P-08 say
  within 100 m *radius*. The sim assumes the stricter 50 m radius (`SIM-T01`).
- **Apex altitude**: 245 km in the ConOps figure vs 359.8 km for the analytic arc.
- **Deep throttle**: hovering at touchdown mass needs 13.5 % of the 6 kN `T_max`, but a pressure-fed
  bipropellant engine typically floors at ≈30 %. 4 engines at 30 % = 1800 N against 812 N of
  weight — **the baseline cannot hover on four engines**. See `TR-SIM-09`; this is a real finding
  for the propulsion team, not a modelling artefact.
- **Nozzle/ground clearance**: a 1.5 kN engine at ε≈60 has a ≈0.25 m exit diameter and ≈0.4 m
  length, which consumes most of the 0.6 m deck-to-surface clearance. Interference risk for
  structures.
- **The 39 km braking-burn ignition altitude is infeasible** (`SIM-A01`, `TR-SIM-18`). The arc
  arrives there descending at 557 m/s, leaving ~140 s before ground contact, while removing
  1376 m/s of horizontal velocity needs ~200 s. Required thrust is **9.36 kN** (`SIM-A20`), 1.56×
  the top of Mattia's 3–6 kN range. Ignition must move to ≈60 km at 6 kN, and scales as roughly
  1/thrust: **138/96/75/62 km for 3/4/5/6 kN** (`SIM-A21`). Two of the MCR's own numbers already
  agree with this — the 202 s burn duration implies ≈55 km, and 6 kN implies ≈62 km. Only §5.3.5's
  39 km disagrees, and it is marked TBC. **Simulated**: lands inside the touchdown box from 60 km
  using 368 kg in 222 s.
- **Sensor pointing vs flown attitude** (`SIM-A16`, `TR-SIM-14`). Simulated, not estimated: the
  vehicle holds **64–68° from local vertical for 174 s of the 222 s burn** (78 %), spiking to 86°
  at ignition where centrifugal relief makes vertical thrust nearly unnecessary, and drops below
  45° only at 2.5 km. A body-fixed "nadir" camera looks below the horizon for almost the whole
  burn (`SIM-A17`), and a body-fixed nadir DLEM 30 first acquires at **2.97 km, 49 s before
  touchdown** (`SIM-A18`) — not the 6–8 km of §5.3.5, which assumes nadir pointing. The
  altimeter-free stretch is therefore **60 km → 3 km**, far worse than the 39→8 km the report
  lists as an open risk.
- **Hazard avoidance is infeasible as specified**: ConOps phase 8 puts hazard detection and
  avoidance in the 6 s vertical descent from 50 m. Divert capability there is **5.3 m** at 20° tilt
  (`SIM-H05`) against ~30 m needed, and the perception latency alone is 3.5 s of the 6 s
  (`SIM-H07`). HDA has to move up to **300–500 m** (`SIM-H06`), or be replaced by orbital
  pre-screening of the site. See `TR-SIM-13`.
- **The guidance law does not meet the landing-accuracy requirement under dispersion**
  (`TR-SIM-23`, `SIM-MC11`). The nominal case lands at 21 m, but the Monte Carlo CEP is **4447 m**
  and R95 is 8326 m against a 50 m requirement — 0 of 24 cases pass. This is *not* a tuning
  problem: the altitude-priority law tracks velocity profiles and never looks at where the site
  is, so an injection velocity error integrates unattenuated over a 222 s burn. Closed-loop
  position guidance (`TR-SIM-11`, convex) is mandatory. **Do not try to tune your way out of
  this** — `TR-SIM-20` records two position-feedback attempts that both fail for the same
  structural reason (with the timeline fixed by a vertical profile, range and velocity are not
  independently controllable).
- **There is effectively no propellant margin** (`SIM-MC12`). The nominal burn uses 368 kg of the
  380 kg loaded, and 11 of 24 dispersed cases ran the tanks dry before touchdown.
- **Navigation error is amplified, not attenuated.** With navigation closed, 22 m of final
  position error produced a 72.6 m miss, because the position-open-loop guidance never corrects
  the steering error the estimate induced. Do not add nav error to guidance miss in quadrature
  and expect it to hold under this law.

## Conventions

- SI units everywhere. Radians in code; degrees only in reports, docstrings and CSV `Unit` columns.
- Frames: Moon-centred inertial (MCI) for propagation; a landing-site ENU frame for guidance and
  telemetry; body frame with **+Z up** (opposite the main-engine thrust).
- Quaternions **scalar-last** `(x, y, z, w)`, matching `scipy.spatial.transform.Rotation`.
- Parameter values in the team spreadsheet are quoted **with margin**; the raw requirement value is
  in the last column. State which one you are using.
- `TBD` means unknown — make it an explicitly named constant with a `TBD` comment and a
  `params.csv` row, never a guess buried in a formula.
