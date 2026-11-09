"""Laboratorio 3.4 · PPO frente a A2C en LunarLander-v3.

Ejemplos:
    # PPO y A2C con valores por defecto, 8 entornos en paralelo, 3 semillas
    python experiments_3/lab_3_4_ppo_a2c.py

    # Barrido del margen de clipping de PPO
    python experiments_3/lab_3_4_ppo_a2c.py --algos ppo --clip 0.1,0.2,0.4

    # PPO con valores basados en rl_zoo3
    python experiments_3/lab_3_4_ppo_a2c.py --algos ppo --zoo

    # Comparar con el DQN del Lab 3.1 (mismo número de pasos en el eje x)
    python experiments_3/plot_results.py logs/lab_3_4 logs/lab_3_1 --configs ppo_default,a2c_default,LunarLander-v3_slides

Métricas de PPO en TensorBoard (tensorboard --logdir logs/tb/lab_3_4):
    train/approx_kl · train/clip_fraction · train/entropy_loss · train/explained_variance
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

from common3 import (banner, base_parser, make_eval_callback, parse_seeds, run_dir,
                     save_config, set_global_seed, tb_dir)

LAB = "lab_3_4"
ENV_ID = "LunarLander-v3"

# Valores basados en rl_zoo3 (hyperparams/ppo.yml, LunarLander). Comprobadlos en vuestra versión.
PPO_ZOO = dict(n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999, n_epochs=4, ent_coef=0.01)


def run(algo_name: str, config: str, hp: dict, args) -> None:
    algo = {"ppo": PPO, "a2c": A2C}[algo_name]
    for seed in parse_seeds(args.seeds):
        out = run_dir(LAB, config, seed)
        save_config(out, {"algo": algo_name, "env": ENV_ID, "n_envs": args.n_envs,
                          "steps": args.steps, "seed": seed, **hp})
        set_global_seed(seed)
        env = make_vec_env(ENV_ID, n_envs=args.n_envs, seed=seed)
        eval_env = make_vec_env(ENV_ID, n_envs=1, seed=10_000 + seed)
        model = algo("MlpPolicy", env, seed=seed, tensorboard_log=tb_dir(LAB),
                     device=args.device, verbose=0, **hp)
        cb = make_eval_callback(eval_env, out, args.eval_every, args.n_envs, args.eval_episodes)
        banner(f"{config} · seed={seed} · {args.n_envs} entornos · {args.steps:,} pasos")
        model.learn(total_timesteps=args.steps, callback=cb, tb_log_name=f"{config}_s{seed}")
        print(f"Mejor retorno en evaluación: {cb.best_mean_reward:.1f}")


def main():
    p = base_parser(__doc__)
    p.set_defaults(eval_every=10_000)
    p.add_argument("--algos", default="ppo,a2c")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--clip", default=None, help="valores de clip_range para PPO, p. ej. 0.1,0.2,0.4")
    p.add_argument("--zoo", action="store_true", help="PPO con valores basados en rl_zoo3")
    args = p.parse_args()

    for algo in [a.strip() for a in args.algos.split(",")]:
        if algo == "ppo" and args.clip:
            for c in [float(x) for x in args.clip.split(",")]:
                hp = {**(PPO_ZOO if args.zoo else {}), "clip_range": c}
                run("ppo", f"ppo{'_zoo' if args.zoo else ''}_clip{c}", hp, args)
        elif algo == "ppo":
            run("ppo", "ppo_zoo" if args.zoo else "ppo_default", PPO_ZOO if args.zoo else {}, args)
        elif algo == "a2c":
            run("a2c", "a2c_default", {}, args)
        else:
            raise SystemExit(f"Algoritmo desconocido: {algo}")

    print("\nSiguiente paso:  python experiments_3/plot_results.py logs/lab_3_4")


if __name__ == "__main__":
    main()
