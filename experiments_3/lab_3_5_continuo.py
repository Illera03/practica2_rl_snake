"""Laboratorio 3.5 · Control continuo: DDPG, TD3 y SAC (y PPO como referencia).

Ejemplos:
    # Pendulum-v1: los tres off-policy, 20k pasos, 3 semillas
    python experiments_3/lab_3_5_continuo.py --env Pendulum-v1 --algos ddpg,td3,sac

    # TD3 con distintos niveles de ruido de exploración (0 = sin ruido)
    python experiments_3/lab_3_5_continuo.py --env Pendulum-v1 --algos td3 --sigma 0.0,0.1,0.5

    # La ducha continua (se normaliza la observación con VecNormalize)
    python experiments_3/lab_3_5_continuo.py --env ShowerContinuous-v0 --algos sac,ppo --steps 30000

    # Ampliación: penalizar cambios bruscos de la acción
    python experiments_3/lab_3_5_continuo.py --env ShowerContinuous-v0 --algos sac --change-penalty 0.2

    python experiments_3/plot_results.py logs/lab_3_5_Pendulum-v1
    python experiments_3/plot_results.py logs/lab_3_5_ShowerContinuous-v0

Referencias: en Pendulum el retorno es siempre ≤ 0 y en torno a −200 o mejor el
péndulo se mantiene arriba. En la ducha, el retorno son los segundos (de 60) dentro
del rango; ejecutad shower_continuous.py para ver la referencia de un controlador simple.
PPO necesita bastantes más pasos que los off-policy en Pendulum: es parte de la lección.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

import shower_continuous  # noqa: F401  (registra ShowerContinuous-v0)
from common3 import (banner, base_parser, make_eval_callback, parse_seeds, run_dir,
                     save_config, set_global_seed, tb_dir)

ALGOS = {"ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
OFF_POLICY = {"ddpg", "td3", "sac"}

HP = {
    "ddpg": dict(learning_rate=1e-3, learning_starts=1_000, gamma=0.98, tau=0.005),
    "td3": dict(learning_rate=1e-3, learning_starts=1_000, gamma=0.98, tau=0.005),
    "sac": dict(learning_rate=1e-3, learning_starts=1_000, gamma=0.98, tau=0.005, ent_coef="auto"),
    "ppo": dict(),
}


def make_envs(env_id: str, seed: int, normalize: bool, env_kwargs: dict):
    env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)
    eval_env = make_vec_env(env_id, n_envs=1, seed=10_000 + seed, env_kwargs=env_kwargs)
    if normalize:
        # Las estadísticas de normalización se sincronizan solas con el EvalCallback.
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    return env, eval_env


def main():
    p = base_parser(__doc__)
    p.set_defaults(eval_every=2_000)
    p.add_argument("--env", default="Pendulum-v1", choices=["Pendulum-v1", "ShowerContinuous-v0"])
    p.add_argument("--algos", default="ddpg,td3,sac")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--sigma", default="0.1", help="desviación del ruido gaussiano (DDPG/TD3); admite lista")
    p.add_argument("--normalize", choices=["auto", "si", "no"], default="auto",
                   help="VecNormalize de la observación (auto: solo en la ducha)")
    p.add_argument("--change-penalty", type=float, default=0.0, help="solo ducha: penaliza |aₜ − aₜ₋₁|")
    args = p.parse_args()

    lab = f"lab_3_5_{args.env}"
    normalize = args.normalize == "si" or (args.normalize == "auto" and args.env.startswith("Shower"))
    env_kwargs = {"change_penalty": args.change_penalty} if args.env.startswith("Shower") else {}
    sigmas = [float(s) for s in args.sigma.split(",")]

    for name in [a.strip() for a in args.algos.split(",")]:
        algo = ALGOS[name]
        for sigma in (sigmas if name in ("ddpg", "td3") else [None]):
            config = name if sigma is None or len(sigmas) == 1 else f"{name}_sigma{sigma}"
            if args.change_penalty:
                config += f"_pen{args.change_penalty}"
            for seed in parse_seeds(args.seeds):
                out = run_dir(lab, config, seed)
                set_global_seed(seed)
                env, eval_env = make_envs(args.env, seed, normalize, env_kwargs)
                hp = dict(HP[name])
                if sigma is not None:
                    n = env.action_space.shape[0]
                    hp["action_noise"] = NormalActionNoise(np.zeros(n), sigma * np.ones(n))
                save_config(out, {"algo": name, "env": args.env, "steps": args.steps, "seed": seed,
                                  "sigma": sigma, "normalize": normalize, **env_kwargs, **HP[name]})
                model = algo("MlpPolicy", env, seed=seed, tensorboard_log=tb_dir(lab),
                             device=args.device, verbose=0, **hp)
                cb = make_eval_callback(eval_env, out, args.eval_every, 1, args.eval_episodes)
                banner(f"{config} · {args.env} · seed={seed} · {args.steps:,} pasos")
                model.learn(total_timesteps=args.steps, callback=cb, tb_log_name=f"{config}_s{seed}")
                if normalize:
                    env.save(str(out / "vecnormalize.pkl"))
                print(f"Mejor retorno en evaluación: {cb.best_mean_reward:.1f}")

    print(f"\nSiguiente paso:  python experiments_3/plot_results.py logs/{lab}")


if __name__ == "__main__":
    main()
