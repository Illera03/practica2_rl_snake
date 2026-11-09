"""Utilidades comunes de los laboratorios de la Unidad 3.

Todos los scripts guardan los resultados con la misma estructura:

    logs/<lab>/<configuración>/seed_<k>/evaluations.npz   (curva de evaluación)
    logs/<lab>/<configuración>/seed_<k>/best_model.zip    (mejor modelo)
    logs/tb/<lab>/...                                     (TensorBoard)

Así, plot_results.py puede comparar cualquier conjunto de configuraciones.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

LOG_ROOT = Path("logs")


# --------------------------------------------------------------------------- semillas
def parse_seeds(text: str) -> list[int]:
    """'0,1,2' -> [0, 1, 2]   ·   '0-4' -> [0, 1, 2, 3, 4]"""
    text = text.strip()
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",") if s.strip()]


def set_global_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch (si está instalado)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- rutas
def run_dir(lab: str, config: str, seed: int) -> Path:
    d = LOG_ROOT / lab / config / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tb_dir(lab: str) -> str:
    d = LOG_ROOT / "tb" / lab
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def save_config(path: Path, config: dict) -> None:
    """Guarda los hiperparámetros usados junto a los resultados (reproducibilidad)."""
    def default(o):
        return str(o)
    (path / "config.json").write_text(json.dumps(config, indent=2, default=default), encoding="utf-8")


# --------------------------------------------------------------------------- evaluación
def make_eval_callback(eval_env, out_dir: Path, eval_every: int, n_envs: int = 1,
                       n_eval_episodes: int = 10, deterministic: bool = True):
    """EvalCallback de SB3 configurado para guardar evaluations.npz en out_dir.

    eval_every se expresa en pasos de entorno TOTALES. SB3 cuenta eval_freq en
    llamadas al callback, y cada llamada avanza n_envs pasos: por eso dividimos.
    """
    from stable_baselines3.common.callbacks import EvalCallback
    return EvalCallback(
        eval_env,
        eval_freq=max(eval_every // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        log_path=str(out_dir),
        best_model_save_path=str(out_dir),
        verbose=0,
    )


def make_monitored_env(env_id, **kwargs):
    """Entorno de evaluación envuelto en Monitor (evita avisos de SB3 y registra episodios)."""
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    return Monitor(gym.make(env_id, **kwargs))


def save_evaluations(out_dir: Path, timesteps, results, successes=None, scores=None, ends=None) -> None:
    """Mismo formato que EvalCallback, para scripts que no usan EvalCallback.

    results: retornos [n_evaluaciones, n_episodios]; scores: manzanas por partida (Snake);
    ends: cómo acabó cada partida de evaluación (Snake: pared, cuerpo, hambre, victoria).
    """
    data = {"timesteps": np.asarray(timesteps), "results": np.asarray(results)}
    if successes is not None:
        data["successes"] = np.asarray(successes)
    if scores is not None:
        data["scores"] = np.asarray(scores)
    if ends is not None:
        data["ends"] = np.asarray(ends, dtype=str)
    np.savez(out_dir / "evaluations.npz", **data)


# --------------------------------------------------------------------------- CLI común
def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seeds", default="0,1,2", help="semillas: '0,1,2' o '0-4'")
    p.add_argument("--eval-every", type=int, default=5_000, help="pasos entre evaluaciones")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--device", default="auto", help="cpu, cuda o auto")
    return p


def banner(text: str) -> None:
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72, flush=True)
