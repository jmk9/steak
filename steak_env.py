# steak_env.py
# Preference mode only (safety mode off). 9-layer thickness model. Real-time params -> auto ×60.

import math, numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------- REAL-WORLD PARAMETERS (you can change these numbers) ----------
ACCEL = 60.0                         # 60× speedup (real 1s -> sim 1/60s)
DT_REAL = 0.10                       # [s] dynamics integration step (real-time)
ACTION_HOLD_REAL = 0.50              # [s] hold last action
EPISODE_LIMIT_REAL = 600.0           # [s] 10 minutes max
FLIP_COOLDOWN_REAL = 5.0             # [s] min gap between flips

PAN_SET_MIN, PAN_SET_MAX = 170.0, 205.0    # [°C]
TAU_PAN_REAL = 30.0                  # [s] first-order heating time constant (you chose 60 s)

NZ = 9                               # layers through thickness (top=0 ... bottom=8)
K_CONTACT_REAL = 0.07                # [1/s] contact-side heating rate (real-time)
ALPHA_DIFF_REAL = 0.015              # [1/s] internal diffusion rate (real-time)

T_INIT_CORE_REAL = 5.0               # [°C] initial core
HEAT_THRESHOLD = 150.0               # [°C] start browning above this
GAMMA_BROWN_REAL = 1.2e-4            # [(s*°C)^-1] browning speed

# Doneness target ranges (°C)
DONENESS_RANGES = {
    "rare":        (52.0, 55.0),
    "mr":          (57.0, 60.0),
    "medium":      (63.0, 66.0),
    "mw":          (68.0, 70.0),     # chosen narrower than well-done
    "well":        (71.0, 100.0),
}
SIGMA_CORE = 1.5                      # [°C] temperature tolerance width

# Color targets from your images (approx Lab)
LAB_RAW   = np.array([49.26, 39.72, 12.33], dtype=float)  # pink/uncooked
LAB_BROWN = np.array([23.54, 19.85, 13.95], dtype=float)  # well-browned
DELTAE_K  = 8.0                       # sensitivity scale for color reward

# Reward weights
W_CORE, W_COLOR, W_TIME, W_FLIPS = 0.5, 0.3, 0.2, 0.02
CORE_OVERCOOK_STOP = 75.0            # [°C] hard stop

# -------------------------------------------------------------------------

# ---- helpers: sRGB<->Lab (for ΔE) ----
def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    a = 0.055
    return np.where(c <= 0.04045, c/12.92, ((c + a)/(1 + a))**2.4)

def _rgb_to_xyz(rgb):
    # rgb in 0..1
    r, g, b = _srgb_to_linear(rgb)
    # sRGB D65
    x = r*0.4124564 + g*0.3575761 + b*0.1804375
    y = r*0.2126729 + g*0.7151522 + b*0.0721750
    z = r*0.0193339 + g*0.1191920 + b*0.9503041
    return np.array([x, y, z])

def _xyz_to_lab(xyz):
    # D65 white
    xr, yr, zr = xyz / np.array([0.95047, 1.00000, 1.08883])
    def f(t): return np.where(t > 0.008856, np.cbrt(t), 7.787*t + 16/116)
    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.array([L, a, b])

def rgb_to_lab01(rgb01):
    return _xyz_to_lab(_rgb_to_xyz(rgb01))

def deltaE_lab(lab1, lab2):
    diff = lab1 - lab2
    return float(np.sqrt(np.sum(diff*diff)))

# map "c" in [0,1] to RGBA between raw->brown (for GUI/log)
RAW_RGBA   = np.array([0.90, 0.40, 0.45, 1.0])
BROWN_RGBA = np.array([0.40, 0.18, 0.08, 1.0])
def lerp_color(c):
    x = float(np.clip(c, 0.0, 1.0))
    return (1-x)*RAW_RGBA + x*BROWN_RGBA

# ---------------- Environment ----------------
class SteakEnv(gym.Env):
    """
    9-layer slab; bottom layer is contact side (changes when flipping).
    No ambient loss. Pan temp follows first-order lag. Preference mode only.
    Observation (float32):
      [T_core, T_top, T_bot, RGB_top(3), RGB_bot(3), T_pan, t_real, flips]
    Action: [Tset_norm in [-1,1], flip in {0,1}]
    """
    metadata = {"render_modes": []}

    def __init__(self, doneness="mr", seed=None, accel: float | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.doneness = doneness.lower()
        assert self.doneness in DONENESS_RANGES
        self.Tmin, self.Tmax = DONENESS_RANGES[self.doneness]
        self.Tmu = 0.5*(self.Tmin + self.Tmax)
        self.accel = float(ACCEL if accel is None else accel)

        # convert real-time params -> sim-time
        self.dt_sim = DT_REAL/self.accel
        self.action_hold_steps = max(1, int(round(ACTION_HOLD_REAL/DT_REAL)))
        self.flip_cooldown_steps = max(1, int(round(FLIP_COOLDOWN_REAL/DT_REAL)))
        self.ep_limit_steps = int(round(EPISODE_LIMIT_REAL/DT_REAL))

        self.tau_pan_sim = TAU_PAN_REAL/self.accel
        self.k_contact_sim = K_CONTACT_REAL*self.accel
        self.alpha_sim = ALPHA_DIFF_REAL*self.accel
        self.gamma_sim = GAMMA_BROWN_REAL*self.accel

        # spaces
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32),
                                       high=np.array([ 1.0, 1.0], dtype=np.float32))
        # T in °C [0..300], RGB [0..1], time 0..600, flips 0..100
        obs_hi = np.array([300,300,300, 1,1,1, 1,1,1, 300, 600, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros_like(obs_hi), high=obs_hi, dtype=np.float32)

        self.reset()

    # ---------- core simulation ----------
    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.t_real = 0.0
        self.steps = 0
        self.flips = 0
        self.last_flip_step = -10**9

        # 9-layer temperature init (all at 5°C)
        self.T = np.full(NZ, T_INIT_CORE_REAL, dtype=float)
        self.c_top = 0.0
        self.c_bot = 0.0

        # pan model
        self.T_set = PAN_SET_MIN
        self.T_pan = self.T_set

        # which index is contact? start: bottom=8 touches pan; top=0 is air side
        self.contact_is_bottom = True

        self._cached_action = np.array([0.0, 0.0], dtype=np.float32)
        return self._obs(), {}

    def step(self, action):
        # action hold at real-time rate
        if (self.steps % self.action_hold_steps) == 0:
            self._cached_action = np.array(action, dtype=np.float32)
        a = self._cached_action

        # decode action
        x = float(np.clip(a[0], -1.0, 1.0))
        self.T_set = PAN_SET_MIN + (x+1.0)*0.5*(PAN_SET_MAX-PAN_SET_MIN)
        do_flip = (a[1] > 0.5) and ((self.steps - self.last_flip_step) >= self.flip_cooldown_steps)

        # update pan (first-order lag in sim-time)
        self.T_pan += (self.T_set - self.T_pan) * (self.dt_sim / max(self.tau_pan_sim, 1e-6))
        self.T_pan = float(np.clip(self.T_pan, PAN_SET_MIN - 5.0, PAN_SET_MAX + 5.0))
        
        # diffusion for inner layers (explicit)
        Tn = self.T.copy()
        for i in range(1, NZ-1):
            Tn[i] += self.alpha_sim * (self.T[i-1] - 2*self.T[i] + self.T[i+1]) * self.dt_sim

        # boundary: contact side towards pan temp
        if self.contact_is_bottom:
            ci, ai = NZ-1, 0  # bottom is contact; top is air
        else:
            ci, ai = 0, NZ-1  # top is contact; bottom is air

        Tn[ci] += self.k_contact_sim * (self.T_pan - self.T[ci]) * self.dt_sim
        # air side has no external flux (per your request); just keep diffusion result
        self.T = Tn

        # flip handling (face identity stays with temperature "node")
        if do_flip:
            self.contact_is_bottom = not self.contact_is_bottom
            self.last_flip_step = self.steps
            self.flips += 1

        # browning progress for each face (based on its current surface temperature)
        T_surf_top = self.T[0]      # 항상 지오메트릭 위쪽 표면
        T_surf_bot = self.T[-1]     # 항상 지오메트릭 아래쪽 표면

        self.c_top = float(np.clip(
            self.c_top + self.gamma_sim * max(T_surf_top - HEAT_THRESHOLD, 0.0) * self.dt_sim, 0.0, 1.0))
        self.c_bot = float(np.clip(
            self.c_bot + self.gamma_sim * max(T_surf_bot - HEAT_THRESHOLD, 0.0) * self.dt_sim, 0.0, 1.0))
            # time/bookkeeping
        self.t_real += DT_REAL
        self.steps  += 1

        obs = self._obs()
        reward, done = self._reward_done(obs)
        truncated = (self.steps >= self.ep_limit_steps)
        return obs, reward, done, truncated, {}

    def _obs(self):
        # core temp ~ middle layer
        T_core = float(self.T[NZ//2])
        # define top/bot as geometric top/bottom for display/HUD (not contact state)
        T_top  = float(self.T[0])
        T_bot  = float(self.T[-1])

        rgb_top = lerp_color(self.c_top)[:3]
        rgb_bot = lerp_color(self.c_bot)[:3]
        return np.array([
            T_core, T_top, T_bot,
            rgb_top[0], rgb_top[1], rgb_top[2],
            rgb_bot[0], rgb_bot[1], rgb_bot[2],
            float(self.T_pan), float(self.t_real), float(self.flips)
        ], dtype=np.float32)

    # ---- rewards ----
    def _core_reward(self, T_core):
        return math.exp(-abs(T_core - self.Tmu)/SIGMA_CORE)

    def _color_reward(self):
        # map c_top/c_bot -> RGB -> Lab, compare to target LAB_BROWN
        lab_top = rgb_to_lab01(lerp_color(self.c_top)[:3])
        lab_bot = rgb_to_lab01(lerp_color(self.c_bot)[:3])
        dE_top = deltaE_lab(lab_top, LAB_BROWN)
        dE_bot = deltaE_lab(lab_bot, LAB_BROWN)
        # mean ΔE + uniformity penalty
        dE_mean = 0.5*(dE_top + dE_bot)
        dE_std  = abs(dE_top - dE_bot)
        base = math.exp(-dE_mean/DELTAE_K)
        return max(0.0, base - 0.25*(dE_std/DELTAE_K))

    def _reward_done(self, obs):
        T_core = float(obs[0])
        r_core  = self._core_reward(T_core)
        r_color = self._color_reward()
        r_time  = float(obs[10])  # t_real in seconds (acts as penalty multiplier below)

        reward = W_CORE*r_core + W_COLOR*r_color - W_TIME*(r_time/EPISODE_LIMIT_REAL) - W_FLIPS*self.flips*0.01

        done = False
        # early stop if clearly excellent (both core in band and color good enough)
        in_band = (self.Tmin <= T_core <= self.Tmax)
        if in_band and (r_color >= 0.7):
            done = True
            reward += 1.0

        if T_core > CORE_OVERCOOK_STOP:
            done = True
            reward -= 1.0

        return float(reward), done
