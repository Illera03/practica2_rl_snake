"""Laboratorio 3.3 · Referencias de SB3 para comparar con vuestro REINFORCE.

Entrena A2C y PPO con el MISMO entorno, diseño (observación y recompensa),
presupuesto de pasos y semillas que lab_3_3_reinforce.py, y guarda los resultados
en logs/lab_3_3/ para que plot_results.py los dibuje junto a vuestras curvas.

    python experiments_3/lab_3_3_referencia_sb3.py --seeds 0-4 --obs custom --reward sparse
    python experiments_3/lab_3_3_referencia_sb3.py --algos ppo --seeds 0-4 --obs original
    python experiments_3/plot_results.py logs/lab_3_3 --metric score

Se usa un solo entorno (n_envs=1) para que la comparación con REINFORCE, que
también usa uno, sea justa en pasos de entorno. Valores por defecto de SB3.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
from stable_baselines3 import A2C, PPO

from common3 import (banner, base_parser, make_eval_callback, make_monitored_env, parse_seeds, run_dir,
                     save_config, set_global_seed, tb_dir)
from snake_design import design_fingerprint, make_snake_env, snake_eval_callback

LAB = "lab_3_3"
ALGOS = {"a2c": A2C, "ppo": PPO}


def main():
    p = base_parser(__doc__)
    p.set_defaults(seeds="0-4", eval_every=25_000, eval_episodes=20)
    p.add_argument("--env", default="snake", help="'snake' o un entorno de Gymnasium (CartPole-v1)")
    p.add_argument("--obs", default="original", choices=["original", "custom"])
    p.add_argument("--reward", default="sparse", choices=["sparse", "distance", "custom"])
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--algos", default="a2c,ppo")
    p.add_argument("--steps", type=int, default=200_000, help="mismo presupuesto que vuestro REINFORCE")
    p.add_argument("--tag", default=None, help="prefijo de la iteración (carpeta de logs), p. ej. it06_ref")
    args = p.parse_args()

    snake = args.env == "snake"
    for name in [a.strip() for a in args.algos.split(",")]:
        algo = ALGOS[name]
        config = f"{name}_{args.obs}_{args.reward}" if snake else f"{name}_{args.env}"
        if args.tag:
            config = f"{args.tag}_{name}"
        for seed in parse_seeds(args.seeds):
            out = run_dir(LAB, config, seed)
            save_config(out, {"algo": name, "env": args.env, "obs": args.obs, "reward": args.reward,
                              "grid": args.grid, "steps": args.steps, "seed": seed,
                              "hiperparámetros": "valores por defecto de SB3",
                              **(design_fingerprint() if snake else {})})
            set_global_seed(seed)
            if snake:
                env = make_snake_env(args.obs, args.reward, args.grid)
                cb = snake_eval_callback(make_snake_env(args.obs, args.reward, args.grid), out,
                                         args.eval_every, args.eval_episodes, seed=10_000 * (seed + 1))
            else:
                env = gym.make(args.env)
                cb = make_eval_callback(make_monitored_env(args.env), out, args.eval_every, 1, args.eval_episodes)
            model = algo("MlpPolicy", env, seed=seed, tensorboard_log=tb_dir(LAB),
                         device=args.device, verbose=0)
            banner(f"{name.upper()} · {config} · seed={seed} · {args.steps:,} pasos")
            model.learn(total_timesteps=args.steps, callback=cb, tb_log_name=f"{config}_s{seed}")

    print("\nSiguiente paso:  python experiments_3/plot_results.py logs/lab_3_3" + (" --metric score" if snake else ""))


if __name__ == "__main__":
    main()
