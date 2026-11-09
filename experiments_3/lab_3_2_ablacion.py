"""Laboratorio 3.2 · Ablación de DQN: ¿qué pieza importa más?

Parte de la configuración de las diapositivas para LunarLander-v3 y cambia UN solo
parámetro en cada variante:

    base               configuración completa
    sin_red_objetivo   target_update_interval=1  (el objetivo se mueve en cada paso)
    buffer_min         buffer_size=1_000          (casi sin replay)
    sin_explor_final   exploration_final_eps=0.0  (explotación pura al final)

Ejemplos:
    python experiments_3/lab_3_2_ablacion.py                         # todas, semillas 0,1,2
    python experiments_3/lab_3_2_ablacion.py --variants base,buffer_min --seeds 0-4
    python experiments_3/plot_results.py logs/lab_3_2 --title "Ablación DQN · LunarLander-v3"

Consejo: lanzad cada variante en una terminal distinta para ir en paralelo:
    python experiments_3/lab_3_2_ablacion.py --variants sin_red_objetivo
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
from stable_baselines3 import DQN

from common3 import (banner, base_parser, make_eval_callback, make_monitored_env, parse_seeds, run_dir,
                     save_config, set_global_seed, tb_dir)
from lab_3_1_dqn import CONFIGS

LAB = "lab_3_2"
ENV_ID = "LunarLander-v3"
BASE = CONFIGS["slides"]

VARIANTS = {
    "base": {},
    "sin_red_objetivo": {"target_update_interval": 1},
    "buffer_min": {"buffer_size": 1_000},
    "sin_explor_final": {"exploration_final_eps": 0.0},
}


def main():
    p = base_parser(__doc__)
    p.add_argument("--variants", default=",".join(VARIANTS), help="variantes separadas por comas")
    p.add_argument("--steps", type=int, default=150_000)
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"Variante desconocida: {v}. Opciones: {', '.join(VARIANTS)}")

    for v in variants:
        hp = {**BASE, **VARIANTS[v]}
        for seed in parse_seeds(args.seeds):
            out = run_dir(LAB, v, seed)
            save_config(out, {"env": ENV_ID, "variant": v, "steps": args.steps, "seed": seed, **hp})
            set_global_seed(seed)
            env, eval_env = gym.make(ENV_ID), make_monitored_env(ENV_ID)
            model = DQN("MlpPolicy", env, seed=seed, tensorboard_log=tb_dir(LAB),
                        device=args.device, verbose=0, **hp)
            cb = make_eval_callback(eval_env, out, args.eval_every, 1, args.eval_episodes)
            banner(f"Ablación DQN · {v} · seed={seed}")
            model.learn(total_timesteps=args.steps, callback=cb, tb_log_name=f"{v}_s{seed}")
            print(f"Mejor retorno en evaluación: {cb.best_mean_reward:.1f}")

    print("\nSiguiente paso:  python experiments_3/plot_results.py logs/lab_3_2")


if __name__ == "__main__":
    main()
