"""Laboratorio 3.1 · DQN con Stable-Baselines3: de CartPole a LunarLander.

Ejemplos:
    # 1) CartPole con los valores por defecto de SB3 (¿aprende en 50k pasos?)
    python experiments_3/lab_3_1_dqn.py --env CartPole-v1 --config default --steps 50000

    # 2) CartPole con valores basados en rl_zoo3
    python experiments_3/lab_3_1_dqn.py --env CartPole-v1 --config zoo --steps 50000

    # 3) LunarLander con la configuración de las diapositivas, 3 semillas
    python experiments_3/lab_3_1_dqn.py --env LunarLander-v3 --config slides --steps 150000 --seeds 0,1,2

    # 4) Grabar un vídeo del mejor modelo (necesita moviepy)
    python experiments_3/lab_3_1_dqn.py --env LunarLander-v3 --video logs/lab_3_1/LunarLander-v3_slides/seed_0/best_model.zip

    # 5) Comparar resultados
    python experiments_3/plot_results.py logs/lab_3_1

TensorBoard:  tensorboard --logdir logs/tb
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
from stable_baselines3 import DQN

from common3 import (banner, base_parser, make_eval_callback, make_monitored_env, parse_seeds, run_dir,
                     save_config, set_global_seed, tb_dir)

LAB = "lab_3_1"

# Configuraciones. "default" = valores por defecto de SB3 (no se pasa nada).
CONFIGS = {
    "default": {},
    # Valores basados en rl_zoo3 (hyperparams/dqn.yml) para CartPole-v1.
    # Comparadlos con el YAML de vuestra versión de rl_zoo3.
    "zoo": dict(
        learning_rate=2.3e-3, batch_size=64, buffer_size=100_000, learning_starts=1_000,
        gamma=0.99, target_update_interval=10, train_freq=256, gradient_steps=128,
        exploration_fraction=0.16, exploration_final_eps=0.04,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
    # Configuración de las diapositivas (pensada para LunarLander-v3).
    "slides": dict(
        learning_rate=6.3e-4, batch_size=128, buffer_size=50_000, learning_starts=0,
        gamma=0.99, train_freq=4, gradient_steps=-1, target_update_interval=250,
        exploration_fraction=0.12, exploration_final_eps=0.1,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
}


def train(env_id: str, config_name: str, steps: int, seed: int, args) -> None:
    hp = CONFIGS[config_name]
    name = f"{env_id}_{config_name}"
    out = run_dir(LAB, name, seed)
    save_config(out, {"env": env_id, "config": config_name, "steps": steps, "seed": seed, **hp})
    set_global_seed(seed)

    env = gym.make(env_id)
    eval_env = make_monitored_env(env_id)
    model = DQN("MlpPolicy", env, seed=seed, tensorboard_log=tb_dir(LAB),
                device=args.device, verbose=0, **hp)
    cb = make_eval_callback(eval_env, out, args.eval_every, 1, args.eval_episodes)
    banner(f"DQN · {env_id} · config={config_name} · seed={seed} · {steps:,} pasos")
    model.learn(total_timesteps=steps, callback=cb, tb_log_name=f"{name}_s{seed}", progress_bar=args.progress)
    model.save(out / "final_model")
    print(f"Mejor retorno en evaluación: {cb.best_mean_reward:.1f}  ·  resultados en {out}")


def record_video(env_id: str, model_path: str, episodes: int = 1) -> None:
    """Graba episodios con la política determinista de un modelo guardado."""
    from gymnasium.wrappers import RecordVideo
    folder = Path(model_path).parent / "videos"
    env = RecordVideo(gym.make(env_id, render_mode="rgb_array"), str(folder),
                      episode_trigger=lambda e: True, name_prefix="dqn")
    model = DQN.load(model_path)
    for ep in range(episodes):
        obs, _ = env.reset(seed=1_000 + ep)
        done, ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        print(f"Episodio {ep}: retorno {ret:.1f}")
    env.close()
    print(f"Vídeos en {folder}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--env", default="CartPole-v1", choices=["CartPole-v1", "LunarLander-v3"])
    p.add_argument("--config", default="slides", choices=list(CONFIGS))
    p.add_argument("--steps", type=int, default=None, help="por defecto: 50k en CartPole, 150k en LunarLander")
    p.add_argument("--progress", action="store_true", help="barra de progreso (requiere tqdm y rich)")
    p.add_argument("--video", default=None, help="ruta a un modelo .zip: graba un episodio y termina")
    args = p.parse_args()

    if args.video:
        record_video(args.env, args.video)
        return
    steps = args.steps or (50_000 if args.env == "CartPole-v1" else 150_000)
    for seed in parse_seeds(args.seeds):
        train(args.env, args.config, steps, seed, args)


if __name__ == "__main__":
    main()
