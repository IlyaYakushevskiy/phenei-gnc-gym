"""Full 6-DOF descent with the hierarchical control stack closed.

    10 Hz   guidance          -> (thrust magnitude, commanded attitude)
    100 Hz  attitude PD       -> body torque
            allocation        -> 12 throttles, bounded [0, 1]
    400 Hz  rigid-body RK4

The guidance law is the same altitude-priority logic as the 3-DOF reference
(``guidance/descent.py``); the difference is that its commanded thrust
direction is now a *target attitude* that the vehicle has to physically slew
to, limited by real control authority. That is the assumption MCR App. D sets
aside ("no attitude or throttle dynamics") and the reason this project exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from phenei_gym.actuators import ActuatorLayout
from phenei_gym.dynamics import PlanarVehicle
from phenei_gym.dynamics.rigid6dof import (
    I_MASS, I_OMEGA, I_POS, I_QUAT, I_VEL,
    make_state, quat_to_matrix, rk4_step, tilt_from_local_vertical, tilt_from_vertical,
)
from phenei_gym.guidance import min_energy_arc
from phenei_gym.guidance.attitude import AttitudeController, quat_pointing
from phenei_gym.params import PARAMS


@dataclass
class Descent6DOFConfig:
    thrust_max: float = 6000.0
    ignition_altitude: float = None          # m; default SIM-A19
    mass: float = None                       # kg; default SIM-M03
    dt: float = 1.0 / 400.0                  # SIM-N01
    attitude_rate: float = 100.0             # Hz, SIM-N02
    guidance_rate: float = 10.0              # Hz, SIM-N03
    t_max: float = 900.0
    omega_n: float = 1.5                     # attitude bandwidth [rad/s]
    zeta: float = 0.9
    n_engines_terminal: int = 2              # TR-SIM-09 option B
    thrust_cap: float = 0.90                 # SIM-GD07: headroom reserved for
                                             # differential-throttle attitude control
    k_vz: float = None                       # SIM-GD08 (6-DOF gain, lower than 3-DOF)
    initial_downrange: float = 0.0           # m; start upstream so touchdown lands on the
                                             # landing site at the frame origin
    range_brake: float = 0.85                # fraction of available horizontal decel used by
                                             # the range-to-go profile
    guidance_mode: str = "profile"           # "profile" | "range" | "zemzev" (targets
                                             # the landing SITE, TR-SIM-11)

    # --- Monte Carlo dispersions (Phase 5). All zero = the nominal case.
    #
    # These perturb the state the vehicle ACTUALLY starts in while leaving
    # the guidance law's internal reference (``self._ic``, from the analytic
    # arc) untouched. That asymmetry is deliberate and is the point of the
    # campaign: the real vehicle is handed over from the ballistic hop with
    # an injection error it does not know about, and the guidance has to
    # absorb it. Perturbing both together would just translate the problem
    # and measure nothing.
    #
    # TR-08 (injection dispersion) has not been run by the mission-analysis
    # team, so SIM-A15 is unknown. These are parameterised rather than
    # guessed at a single value, and the campaign reports sensitivity to
    # them -- see PLAN section 3.
    d_altitude: float = 0.0                  # m, additive
    d_v_horizontal: float = 0.0              # m/s, additive
    d_v_vertical: float = 0.0                # m/s, additive
    d_crossrange: float = 0.0                # m, additive (out of the hop plane)
    d_v_crossrange: float = 0.0              # m/s, additive
    d_mass: float = 0.0                      # kg, additive
    thrust_scale: float = 1.0                # multiplicative on delivered thrust
    isp_scale: float = 1.0                   # multiplicative on Isp

    def __post_init__(self):
        if self.ignition_altitude is None:
            self.ignition_altitude = PARAMS.value("SIM-A19") * 1e3
        if self.mass is None:
            self.mass = PARAMS.value("SIM-M03")
        if self.k_vz is None:
            self.k_vz = PARAMS.value("SIM-GD08")


@dataclass
class Result6DOF:
    t: np.ndarray
    state: np.ndarray            # (n, 14)
    throttles: np.ndarray        # (n, 12)
    tilt: np.ndarray             # (n,) rad
    tilt_cmd: np.ndarray         # (n,) rad
    reason: str
    cfg: Descent6DOFConfig
    layout: ActuatorLayout = field(repr=False, default=None)

    @property
    def altitude(self) -> np.ndarray:
        r = self.state[:, I_POS] + np.array([0.0, 0.0, PARAMS.value("SIM-E03") * 1e3])
        return np.linalg.norm(r, axis=1) - PARAMS.value("SIM-E03") * 1e3

    @property
    def propellant(self) -> float:
        return float(self.state[0, I_MASS] - self.state[-1, I_MASS])

    def touchdown(self) -> dict[str, float]:
        s = self.state[-1]
        radius = PARAMS.value("SIM-E03") * 1e3
        r_vec = s[I_POS] + np.array([0.0, 0.0, radius])
        up = r_vec / np.linalg.norm(r_vec)
        v = s[I_VEL]
        v_up = float(v @ up)
        return {
            "v_vertical": v_up,
            "v_horizontal": float(np.linalg.norm(v - v_up * up)),
            "tilt_deg": math.degrees(tilt_from_local_vertical(s[I_QUAT], s[I_POS], radius)),
            "rate_deg_s": math.degrees(float(np.linalg.norm(s[I_OMEGA]))),
            "miss_m": float(np.linalg.norm(s[I_POS][:2])),
        }


class Descent6DOF:
    def __init__(self, cfg: Descent6DOFConfig | None = None):
        self.cfg = cfg or Descent6DOFConfig()
        self.veh = PlanarVehicle.from_params(thrust_max=self.cfg.thrust_max)
        self.layout = ActuatorLayout.from_params(self.cfg.thrust_max)
        self.radius = self.veh.radius
        # Inertia scales with mass at fixed gyradius (SIM-M09).
        self._I_ref = np.array([PARAMS.value("SIM-M07"), PARAMS.value("SIM-M07"),
                                PARAMS.value("SIM-M08")])
        self._m_ref = PARAMS.value("SIM-M03")
        self.ctl = AttitudeController(self.layout, self._I_ref,
                                      omega_n=self.cfg.omega_n, zeta=self.cfg.zeta)
        self.h_terminal = PARAMS.value("SIM-A12")
        self.throttle_min = PARAMS.value("SIM-P04") / 100.0
        self.v_touchdown = PARAMS.value("SIM-GD04")
        self.k_vz = self.cfg.k_vz
        self.k_vx = PARAMS.value("SIM-GD03")
        self.k_vx_range = 0.6
        st = min_energy_arc().state_at_altitude(self.cfg.ignition_altitude)
        self.a_dec = st["v_vertical"] ** 2 / (2.0 * self.cfg.ignition_altitude)
        self._ic = st

    def inertia(self, mass: float) -> np.ndarray:
        return self._I_ref * (mass / self._m_ref)

    def initial_state(self) -> np.ndarray:
        st, c = self._ic, self.cfg
        r0 = st["radius"]
        pos = np.array([c.initial_downrange,
                        c.d_crossrange,
                        r0 - self.radius + c.d_altitude])
        vel = np.array([st["v_horizontal"] + c.d_v_horizontal,
                        c.d_v_crossrange,
                        st["v_vertical"] + c.d_v_vertical])
        mass = c.mass + c.d_mass
        # start already pointing where guidance will want it
        q0 = quat_pointing(self._thrust_direction(pos, vel, mass=mass)[0])
        return make_state(pos, vel, q0, np.zeros(3), mass)

    # ------------------------------------------------------------------
    def _local(self, pos, vel):
        r_vec = pos + np.array([0.0, 0.0, self.radius])
        r = np.linalg.norm(r_vec)
        up = r_vec / r
        v_up = float(vel @ up)
        v_h_vec = vel - v_up * up
        v_h = float(np.linalg.norm(v_h_vec))
        h_hat = v_h_vec / v_h if v_h > 1e-6 else np.array([1.0, 0.0, 0.0])
        return dict(alt=r - self.radius, up=up, v_up=v_up, v_h=v_h, h_hat=h_hat,
                    g_eff=self.veh.mu / (r * r) - v_h * v_h / r)

    def _thrust_direction(self, pos, vel, mass):
        """Guidance: returns (unit thrust direction in world, thrust magnitude)."""
        L = self._local(pos, vel)
        alt = L["alt"]
        v_z_des = -math.sqrt(2.0 * self.a_dec * max(alt, 0.0) + self.v_touchdown**2)
        # FEEDFORWARD + feedback. The constant-deceleration profile itself
        # requires a_dec of vertical deceleration on top of effective gravity;
        # without that as a feedforward term the loop has to generate it from
        # tracking error alone, which needs a large gain, which saturates the
        # whole thrust budget. The result is a commanded attitude that
        # bang-bangs between 0 and 90 deg -- survivable in 3-DOF where thrust
        # direction is instantaneous, fatal in 6-DOF.
        a_up = L["g_eff"] + self.a_dec + self.k_vz * (v_z_des - L["v_up"])
        a_h = -self.k_vx * L["v_h"]

        # Cap below T_max: differential throttle can only produce torque with
        # throttle headroom. At 100 % there is none, and attitude authority
        # collapses from ~8.4 rad/s2 to the RCS-only ~0.2 rad/s2.
        t_max = self.cfg.thrust_cap * self.cfg.thrust_max
        t_min = self.throttle_min * self.cfg.thrust_max
        if alt <= self.h_terminal:
            frac = self.cfg.n_engines_terminal / 4.0
            t_max, t_min = frac * t_max, self.throttle_min * frac * self.cfg.thrust_max

        if self.cfg.guidance_mode == "range":
            # Range-to-go velocity profile: the horizontal analogue of the
            # vertical constant-deceleration law. Instead of commanding zero
            # horizontal velocity (which lands wherever the trajectory ends,
            # SIM-GD16) or ZEM/ZEV (which demands ~27 m/s2 against ~5.7
            # available and runs the tanks dry), target the speed from which
            # the remaining range can still be braked out:
            #     v_des = sqrt(2 a_brake d)   toward the landing site.
            # Self-limiting by construction, and it goes to zero AT the target.
            to_target = np.array([-pos[0], -pos[1], 0.0])
            d = float(np.linalg.norm(to_target))
            v_horiz = vel - (vel @ L["up"]) * L["up"]
            a_max_h = math.sqrt(max((self.cfg.thrust_cap * self.cfg.thrust_max / mass) ** 2
                                    - a_up**2, 0.25))
            if d > 1e-6:
                v_des = math.sqrt(2.0 * self.cfg.range_brake * a_max_h * d) * (to_target / d)
            else:
                v_des = np.zeros(3)
            a_vec = self.k_vx_range * (v_des - v_horiz)
            n = float(np.linalg.norm(a_vec))
            if n > 1e-9:
                a_h = -n
                L = dict(L); L["h_hat"] = -a_vec / n

        if self.cfg.guidance_mode == "zemzev":
            # ZEM/ZEV on the horizontal channel: the profile law controls only
            # altitude and velocity, so where it lands is wherever the
            # trajectory ends (2.4 km miss, SIM-GD16). This adds the missing
            # position feedback. t_go comes from the vertical profile, which
            # is the one timescale both channels already share.
            t_go = max(math.sqrt(2.0 * max(alt, 0.0) / self.a_dec), 1.0)
            horiz = np.array([pos[0], pos[1], 0.0])          # target is the origin
            v_horiz = vel - (vel @ L["up"]) * L["up"]
            zem = -(horiz + v_horiz * t_go)
            zev = -v_horiz
            a_vec = (6.0 / t_go**2) * zem - (2.0 / t_go) * zev
            a_h = -float(np.linalg.norm(a_vec))
            if abs(a_h) > 1e-9:
                L = dict(L); L["h_hat"] = -a_vec / abs(a_h)

        a_mag = math.hypot(a_up, a_h)
        thrust = float(np.clip(mass * a_mag, t_min, t_max))
        a_avail = thrust / mass
        # Thrust is unidirectional along body +Z, so the vertical demand is
        # floored at zero: "descend faster" must be achieved by throttling
        # down, never by pointing the engines at the sky. Without this floor
        # the guidance commands a 180 deg flip, which is both impossible at
        # realistic slew rates and the wrong manoeuvre.
        a_up_c = float(np.clip(a_up, 0.0, a_avail))
        # Command the horizontal deceleration actually WANTED, limited by what
        # is left after the vertical demand -- not the whole leftover budget.
        # Spending the leftover unconditionally means that with the horizontal
        # velocity already nulled the guidance still tilts the vehicle over
        # (5.4 deg at touchdown), and earlier in the burn it is what makes the
        # commanded attitude bang-bang between 0 and 90 deg.
        a_h_budget = math.sqrt(max(a_avail**2 - a_up_c**2, 0.0))
        a_h_c = -min(abs(a_h), a_h_budget)
        d = a_up_c * L["up"] + a_h_c * L["h_hat"]
        n = np.linalg.norm(d)
        return (d / n if n > 1e-9 else L["up"]), thrust

    # ------------------------------------------------------------------
    def run(self) -> Result6DOF:
        cfg = self.cfg
        s = self.initial_state()
        dt = cfg.dt
        n_att = max(1, int(round(1.0 / (cfg.attitude_rate * dt))))
        n_gui = max(1, int(round(1.0 / (cfg.guidance_rate * dt))))

        ts, xs, us, tilts, tilt_cmds = [0.0], [s.copy()], [], [], []
        q_cmd = s[I_QUAT].copy()
        thrust_cmd = 0.0
        u = np.zeros(self.layout.n)
        reason = "t_max reached"
        t = 0.0

        for k in range(int(cfg.t_max / dt)):
            mass = s[I_MASS]
            if k % n_gui == 0:
                d_world, thrust_cmd = self._thrust_direction(s[I_POS], s[I_VEL], mass)
                q_cmd = quat_pointing(d_world)
            if k % n_att == 0:
                active = np.ones(self.layout.n)
                if self._local(s[I_POS], s[I_VEL])["alt"] <= self.h_terminal:
                    # TR-SIM-09 option B: shut down 2 of 4. They MUST be a
                    # diametrically opposite pair (azimuths 90 and 270); taking
                    # main_idx[2:] leaves engines at 0 and 90 deg, i.e. ADJACENT,
                    # producing a constant torque no allocator can trim.
                    mi = self.layout.main_idx
                    if self.cfg.n_engines_terminal == 2:
                        active[[mi[1], mi[3]]] = 0.0
                self.ctl.inertia = self.inertia(mass)
                self.ctl.kp = 2.0 * self.ctl.inertia * cfg.omega_n**2
                self.ctl.kd = 2.0 * self.ctl.inertia * cfg.zeta * cfg.omega_n
                u = self.ctl.command(s, q_cmd, thrust_cmd, active=active)
                if mass <= self.veh.mass_dry:
                    u = np.zeros(self.layout.n)

            w = self.layout.wrench_of(u)
            # Delivered thrust and Isp are scaled, not the commanded throttle:
            # the controller still believes it is getting what it asked for, so
            # a thrust shortfall shows up as a tracking error the loop has to
            # absorb, which is what a real engine-performance dispersion does.
            force_body = np.array([0.0, 0.0, w[0] * cfg.thrust_scale])
            torque = w[1:] * cfg.thrust_scale
            mdot = self.layout.mass_flow(u) * cfg.thrust_scale / cfg.isp_scale
            s = rk4_step(s, force_body, torque, mdot,
                         self.inertia(mass), self.veh.mu, self.radius, dt)
            t += dt

            ts.append(t); xs.append(s.copy()); us.append(u.copy())
            tilts.append(tilt_from_local_vertical(s[I_QUAT], s[I_POS], self.radius))
            tilt_cmds.append(tilt_from_local_vertical(q_cmd, s[I_POS], self.radius))

            L = self._local(s[I_POS], s[I_VEL])
            if L["alt"] <= 0.0:
                reason = "touchdown"; break
            if L["alt"] > 1.5 * cfg.ignition_altitude:
                reason = "altitude diverged"; break
            if s[I_MASS] <= self.veh.mass_dry:
                reason = "propellant exhausted"; break

        us.append(us[-1] if us else np.zeros(self.layout.n))
        tilts.append(tilts[-1] if tilts else 0.0)
        tilt_cmds.append(tilt_cmds[-1] if tilt_cmds else 0.0)
        return Result6DOF(np.asarray(ts), np.asarray(xs), np.asarray(us),
                          np.asarray(tilts), np.asarray(tilt_cmds), reason, cfg, self.layout)
