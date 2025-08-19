import numpy as np, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

class SteakEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([-1., 0.]), high=np.array([1., 1.]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self._reset_state()
    def _reset_state(self):
        self.T_core, self.T_surf, self.color_var, self.t = 5.0, 5.0, 0.0, 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_state()
        return self._obs(), {}
    def step(self, action):
        dt = 0.02
        temp_set = 120 + (action[0].clip(-1,1)+1)*0.5*(260-120)
        flip = action[1] > 0.5
        # 매우 단순 근사 동역학(데모용)
        self.T_surf += 0.02*(temp_set - self.T_surf)*dt
        self.T_core += 0.015*(self.T_surf - self.T_core)*dt
        if flip: self.color_var *= 0.9
        self.color_var = max(0.0, self.color_var + 1e-3*(self.T_surf-150.0)*dt)
        self.t += dt
        r = self._reward()
        done = self.t >= 30.0
        return self._obs(), r, done, False, {}
    def _obs(self):
        return np.array([self.T_core, self.T_surf, self.color_var, self.t], dtype=np.float32)
    def _reward(self):
        target = 55.0
        r1 = np.exp(-abs(self.T_core-target)/3.0)
        r2 = np.exp(-self.color_var*5.0)
        r3 = -0.02*(self.t/30.0)
        return float(0.6*r1 + 0.3*r2 + 0.1*r3)

if __name__ == "__main__":
    env = SteakEnv()
    model = SAC("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=30000)
    model.save("sac_steak.zip")
    print("Training done.")