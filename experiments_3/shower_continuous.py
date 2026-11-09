"""La ducha de la Unidad 1, versión con observación y acción continuas.

Objetivo: mantener la temperatura entre 36 °C y 38 °C durante 60 s.

    observación  Box([25], [50])   temperatura actual (°C)
    acción       Box([-1], [1])    ajuste del grifo en °C por segundo
    dinámica     T ← clip(T + a + ruido, 25, 50),  ruido ∼ 𝒩(0, 0,2²)
    recompensa   +1 si 36 ≤ T ≤ 38; 0 en otro caso
                 (opcional) − change_penalty · |aₜ − aₜ₋₁|  para un control más suave
    final        terminated si T toca 25 o 50 (límite de seguridad); truncated a los 60 pasos

Uso:
    python experiments_3/shower_continuous.py          # check_env + dos políticas de referencia

En otros scripts:
    import shower_continuous            # registra "ShowerContinuous-v0"
    env = gym.make("ShowerContinuous-v0", change_penalty=0.1)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_ID = "ShowerContinuous-v0"


class ShowerContinuousEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self, noise_std: float = 0.2, change_penalty: float = 0.0, render_mode=None):
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(np.array([25.0], np.float32), np.array([50.0], np.float32))
        self.min_temp, self.max_temp = 25.0, 50.0
        self.max_steps = 60
        self.noise_std = noise_std
        self.change_penalty = change_penalty
        self.render_mode = render_mode
        self.temp, self.t, self.prev_action = None, 0, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temp = float(self.np_random.uniform(30.0, 44.0))     # igual que en la U1
        self.t = 0
        self.prev_action = 0.0
        return np.array([self.temp], np.float32), {"temperature": self.temp}

    def step(self, action):
        delta = float(np.clip(action[0], -1.0, 1.0))               # °C/s
        noise = float(self.np_random.normal(0.0, self.noise_std))
        self.temp = float(np.clip(self.temp + delta + noise, self.min_temp, self.max_temp))
        self.t += 1

        reward = 1.0 if 36.0 <= self.temp <= 38.0 else 0.0
        if self.change_penalty:
            reward -= self.change_penalty * abs(delta - self.prev_action)
        self.prev_action = delta

        terminated = self.temp in (self.min_temp, self.max_temp)    # límite de seguridad
        truncated = self.t >= self.max_steps                         # 60 segundos
        obs = np.array([self.temp], np.float32)
        return obs, reward, terminated, truncated, {"temperature": self.temp}

    def render(self):
        if self.render_mode == "ansi":
            return f"t={self.t:2d}s  T={self.temp:5.2f} °C"
        if self.render_mode == "rgb_array":
            # termómetro horizontal: zona objetivo en verde, temperatura actual en rojo
            img = np.full((40, 250, 3), 255, np.uint8)
            x = lambda temp: int((temp - self.min_temp) / (self.max_temp - self.min_temp) * 249)
            img[10:30, x(36.0):x(38.0) + 1] = (120, 200, 120)
            img[5:35, max(x(self.temp) - 1, 0):x(self.temp) + 2] = (200, 30, 30)
            return img


def register() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=ShowerContinuousEnv)


register()


def _rollout(env, policy, episodes=20, seed=0):
    rets = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ret += r
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets)), float(np.std(rets))


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = gym.make(ENV_ID)
    check_env(env.unwrapped)
    print("check_env: OK")

    rnd = lambda obs: env.action_space.sample()
    prop = lambda obs: np.clip([(37.0 - obs[0]) * 0.5], -1.0, 1.0).astype(np.float32)
    print("Segundos dentro del rango (de 60), media ± desviación en 20 episodios:")
    print("  política aleatoria:       %.1f ± %.1f" % _rollout(env, rnd))
    print("  controlador proporcional: %.1f ± %.1f" % _rollout(env, prop))
    print("Un agente de RL debería acercarse al controlador proporcional.")
