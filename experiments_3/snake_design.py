"""Diseño del problema del Snake: observación y recompensa  (parte de la entrega).

Aquí decidís QUÉ ve el agente y QUÉ se le premia. El mismo diseño lo usan vuestro
REINFORCE (lab_3_3_reinforce.py) y las referencias de SB3 (lab_3_3_referencia_sb3.py),
así que la comparación entre algoritmos es justa.

    TODO 6  SnakeObservation   una observación mejor que la "original"
    TODO 7  SnakeReward        una recompensa propia (si decidís cambiarla)

Regla: la observación DESCRIBE el estado; no puede contener acciones recomendadas ni el
resultado de una búsqueda (BFS, A*, una heurística que decida). Los rasgos calculados,
como las casillas libres en cada dirección, se permiten si se declaran en el diario.
Cada cambio de diseño es una iteración del diario (lanzadla con su propio --tag).

Uso desde los scripts:   --obs original|custom   --reward sparse|distance|custom

Para medir el rendimiento NO se usa el retorno (depende de la recompensa que elijáis),
sino el número de manzanas por partida (info["score"]): así se pueden comparar diseños.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import snake_env  # noqa: F401  (registra DeustoSnake-v0)
from snake_env import ENV_ID, MOVES


# =========================================================================== TODO 6
class SnakeObservation(gym.ObservationWrapper):
    """Observación diseñada por vosotros.

    Tenéis acceso al estado completo en self.env.unwrapped:
        snake (lista de (x, y), cabeza primero) · apple (x, y) · direction (0–3)
        grid_size · steps_since_apple
    y a la tabla MOVES (acción -> desplazamiento).

    Preguntas para guiar el diseño (justificadlas en el informe):
      - ¿Necesita el agente coordenadas absolutas o le basta información relativa a la cabeza?
      - ¿Cómo sabe si el siguiente paso en cada dirección es mortal?
      - ¿Qué tamaño de observación es razonable para una red pequeña?

    Regla: describe el estado, no recomiendes acciones (ver la cabecera del fichero).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # TODO 6: definid self.observation_space (spaces.Box con low, high, shape y dtype=np.float32).
        raise NotImplementedError("TODO 6: define el espacio de observación")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # TODO 6: construid la observación a partir de self.env.unwrapped (ignorad `obs` si queréis).
        raise NotImplementedError("TODO 6: construye la observación")


# =========================================================================== TODO 7
class SnakeReward(gym.Wrapper):
    """Recompensa diseñada por vosotros (se aplica sobre la recompensa "sparse").

    Recordad la U2 (diseño de recompensas, efecto cobra): el agente maximizará
    exactamente lo que premiéis, aunque no sea lo que queríais.

    Podéis quedaros con "sparse" si lo justificáis; cambiar la recompensa es una iteración
    más del diario. Si la cambiáis, comprobad que las manzanas suben, no solo el retorno.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # TODO 7: modificad `reward` usando el estado de self.env.unwrapped e `info`.
        raise NotImplementedError("TODO 7: define tu recompensa")
        return obs, reward, terminated, truncated, info


# =========================================================================== (hecho)
def make_snake_env(obs: str = "original", reward: str = "sparse", grid_size: int = 12,
                   render_mode: str | None = None) -> gym.Env:
    """Crea el Snake con el diseño elegido."""
    base_reward = reward if reward in ("sparse", "distance") else "sparse"
    env = gym.make(ENV_ID, grid_size=grid_size, reward_mode=base_reward, render_mode=render_mode)
    if obs == "custom":
        env = SnakeObservation(env)
    elif obs != "original":
        raise ValueError("obs debe ser 'original' o 'custom'")
    if reward == "custom":
        env = SnakeReward(env)
    elif reward not in ("sparse", "distance"):
        raise ValueError("reward debe ser 'sparse', 'distance' o 'custom'")
    return env


def evaluate_snake(act, env: gym.Env, n_episodes: int, seed: int,
                   ends: list | None = None) -> tuple[list[float], list[int]]:
    """Juega n partidas con act(obs) -> acción. Devuelve (retornos, manzanas por partida).

    Si se pasa la lista `ends`, añade cómo acabó cada partida (pared, cuerpo, hambre, victoria).
    """
    returns, scores = [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            ret += float(r)
            done = term or trunc
        returns.append(ret)
        scores.append(int(info["score"]))
        if ends is not None:
            ends.append(str(info.get("fin")))
    return returns, scores


def snake_eval_callback(eval_env: gym.Env, out_dir, eval_every: int, n_episodes: int, seed: int):
    """Callback de SB3 que evalúa en manzanas por partida y guarda evaluations.npz
    con el mismo formato que lab_3_3_reinforce.py (results = retornos, scores = manzanas)."""
    from stable_baselines3.common.callbacks import BaseCallback

    from common3 import save_evaluations

    class SnakeEval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0
            self.t, self.r, self.s, self.e = [], [], [], []

        def _evaluate(self):
            act = lambda o: int(self.model.predict(o, deterministic=True)[0])
            ends = []
            rets, scores = evaluate_snake(act, eval_env, n_episodes, seed, ends)
            self.t.append(self.num_timesteps); self.r.append(rets); self.s.append(scores); self.e.append(ends)
            save_evaluations(out_dir, self.t, self.r, scores=self.s, ends=self.e)
            self.logger.record("eval/mean_score", float(np.mean(scores)))
            self.logger.record("eval/mean_reward", float(np.mean(rets)))

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self._evaluate()
                self.next_eval += eval_every
            return True

        def _on_training_end(self) -> None:
            self._evaluate()

    return SnakeEval()


def design_fingerprint() -> dict:
    """Huella del código de SnakeObservation y SnakeReward, para el diario.

    Se guarda en config.json: así diario.py detecta qué iteraciones cambiaron el código
    del diseño aunque los argumentos (--obs custom, --reward custom) sean los mismos.
    """
    import hashlib
    import inspect

    def h(obj):
        try:
            return hashlib.sha1(inspect.getsource(obj).encode("utf-8")).hexdigest()[:8]
        except (OSError, TypeError):
            return "?"
    return {"obs_code": h(globals()["SnakeObservation"]), "reward_code": h(globals()["SnakeReward"])}


def check_design(obs: str, reward: str, grid_size: int = 12) -> None:
    """Comprobación rápida de un diseño: formas, límites y una partida aleatoria."""
    env = make_snake_env(obs, reward, grid_size)
    o, _ = env.reset(seed=0)
    assert isinstance(env.observation_space, spaces.Box), "la observación debe ser un spaces.Box"
    assert o.dtype == np.float32, f"la observación debe ser float32, no {o.dtype}"
    assert o.shape == env.observation_space.shape, f"forma {o.shape} ≠ observation_space {env.observation_space.shape}"
    done, steps = False, 0
    while not done:
        o, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(o), f"observación fuera de observation_space: {o}"
        assert np.isfinite(r), "recompensa no finita"
        done, steps = term or trunc, steps + 1
    print(f"  [OK] diseño obs={obs}, reward={reward}: observación de tamaño {o.shape[0]}, partida de {steps} pasos")
