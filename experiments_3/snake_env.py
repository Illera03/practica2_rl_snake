"""Snake para Gymnasium · entorno de la implementación evaluable de la Unidad 3.

Versión actualizada del Snake de cursos anteriores (basado en el de Sentdex, que a
su vez adaptaba TheAILearner). Cambios respecto al original:

  - API de Gymnasium 1.x: reset(seed) -> (obs, info); step -> 5 valores.
  - Coordenadas en casillas (grid_size × grid_size) en lugar de píxeles.
  - Sin ventana ni esperas por defecto: entrena rápido y funciona en Colab.
    render_mode="human" abre una ventana (OpenCV); "rgb_array" devuelve la imagen.
  - truncated si la serpiente lleva demasiados pasos sin comer (evita bucles infinitos).
  - La manzana aparece siempre en una casilla libre.
  - Se corrige un error del original: step() devolvía la recompensa ACUMULADA
    (total_reward) en lugar de la recompensa del paso.
  - info["fin"] dice cómo acabó la partida: "pared", "cuerpo" (también al girar hacia
    atrás), "hambre" o "victoria"
    (None mientras sigue). Sirve para diagnosticar qué limita al agente.

Acciones (absolutas, como en el original):  0 izquierda · 1 derecha · 2 abajo · 3 arriba
Ir en sentido contrario al movimiento hace chocar la cabeza con el cuello: termina.

Observación por defecto ("original"), normalizada:
    [cabeza_x, cabeza_y, Δx_manzana, Δy_manzana, longitud] + últimas K acciones
    Es la misma información que el original. Diseñar una mejor es parte del trabajo.

Recompensa (reward_mode):
    "sparse"    +1 al comer, −1 al morir, 0 en el resto
    "distance"  "sparse" + shaping: +0,1 si se acerca a la manzana, −0,1 si se aleja

Estado accesible para diseñar observaciones y recompensas propias (env.unwrapped):
    snake (lista de casillas (x, y), la cabeza primero) · apple · direction · grid_size
    steps_since_apple · score · end_reason
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_ID = "DeustoSnake-v0"
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}      # izquierda, derecha, abajo, arriba


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, grid_size: int = 12, reward_mode: str = "sparse", history: int = 30,
                 max_hunger: int | None = None, render_mode: str | None = None):
        super().__init__()
        if reward_mode not in ("sparse", "distance"):
            raise ValueError("reward_mode debe ser 'sparse' o 'distance'")
        self.grid_size = grid_size
        self.reward_mode = reward_mode
        self.history = history
        self.max_hunger = max_hunger or grid_size * grid_size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5 + history,), dtype=np.float32)
        self.snake: list[tuple[int, int]] = []
        self.apple = (0, 0)
        self.direction = 1
        self.score = 0
        self.steps_since_apple = 0
        self.end_reason: str | None = None
        self.prev_actions: deque = deque(maxlen=history)

    # ------------------------------------------------------------------ dinámica
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self.grid_size // 2
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]           # mirando a la derecha
        self.direction = 1
        self.score = 0
        self.steps_since_apple = 0
        self.end_reason = None
        self.prev_actions = deque([-1] * self.history, maxlen=self.history)
        self._place_apple()
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        self.prev_actions.append(action)
        self.direction = action
        old_dist = self._apple_distance()

        dx, dy = MOVES[action]
        hx, hy = self.snake[0]
        head = (hx + dx, hy + dy)
        ate = head == self.apple
        self.snake.insert(0, head)
        if not ate:
            self.snake.pop()                                      # la cola avanza

        out = not (0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size)
        bitten = head in self.snake[1:]
        terminated = out or bitten
        truncated = False

        if terminated:
            reward = -1.0
            self.end_reason = "pared" if out else "cuerpo"
        elif ate:
            reward = 1.0
            self.score += 1
            self.steps_since_apple = 0
            if len(self.snake) == self.grid_size * self.grid_size:
                terminated = True                                 # tablero lleno: victoria
                self.end_reason = "victoria"
            else:
                self._place_apple()
        else:
            reward = 0.0
            self.steps_since_apple += 1
            if self.reward_mode == "distance":
                reward += 0.1 if self._apple_distance() < old_dist else -0.1
            truncated = self.steps_since_apple >= self.max_hunger
            if truncated:
                self.end_reason = "hambre"

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._info()

    def _place_apple(self):
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in occupied]
        self.apple = free[self.np_random.integers(len(free))]

    def _apple_distance(self) -> int:
        hx, hy = self.snake[0]
        return abs(self.apple[0] - hx) + abs(self.apple[1] - hy)

    def _obs(self) -> np.ndarray:
        n = self.grid_size
        hx, hy = self.snake[0]
        base = [hx / n, hy / n, (self.apple[0] - hx) / n, (self.apple[1] - hy) / n, len(self.snake) / (n * n)]
        hist = [a / 3 for a in self.prev_actions]                 # −1 (sin acción) se queda en −1/3
        return np.clip(np.array(base + hist, dtype=np.float32), -1.0, 1.0)

    def _info(self) -> dict:
        return {"score": self.score, "length": len(self.snake), "fin": self.end_reason}

    # ------------------------------------------------------------------ dibujo
    def render(self):
        img = self._frame()
        if self.render_mode == "rgb_array":
            return img
        if self.render_mode == "human":
            import cv2
            cv2.imshow("Snake", img[:, :, ::-1])
            cv2.waitKey(int(1000 / self.metadata["render_fps"]))

    def _frame(self, cell: int = 24) -> np.ndarray:
        n = self.grid_size
        img = np.zeros((n * cell, n * cell, 3), np.uint8)
        ax, ay = self.apple
        img[ay * cell + 3:(ay + 1) * cell - 3, ax * cell + 3:(ax + 1) * cell - 3] = (220, 40, 40)
        for i, (x, y) in enumerate(self.snake):
            if 0 <= x < n and 0 <= y < n:
                color = (40, 200, 90) if i else (170, 255, 120)
                img[y * cell + 2:(y + 1) * cell - 2, x * cell + 2:(x + 1) * cell - 2] = color
        return img

    def close(self):
        if self.render_mode == "human":
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass


def register() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=SnakeEnv)


register()


if __name__ == "__main__":
    # Comprobación del entorno y referencia de una política aleatoria.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--human", action="store_true", help="ver una partida aleatoria en una ventana")
    args = p.parse_args()

    from stable_baselines3.common.env_checker import check_env
    check_env(SnakeEnv(grid_size=args.grid))
    print("check_env: OK")

    env = gym.make(ENV_ID, grid_size=args.grid, render_mode="human" if args.human else None)
    from collections import Counter
    scores, lengths, ends = [], [], Counter()
    for ep in range(50 if not args.human else 2):
        obs, info = env.reset(seed=ep)
        done, t = False, 0
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            done, t = term or trunc, t + 1
        scores.append(info["score"]); lengths.append(t); ends[info["fin"]] += 1
    env.close()
    print(f"Política aleatoria: {np.mean(scores):.2f} manzanas por partida, {np.mean(lengths):.1f} pasos de media")
    print("Cómo acaban las partidas:", dict(ends))
