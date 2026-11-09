"""Laboratorio 3.6 · Hindsight Experience Replay (HER).

Dos partes:

  bitflip   Problema de juguete de SB3: voltear n bits hasta igualar un objetivo.
            Con 15 bits, acertar al azar es prácticamente imposible: sin HER no hay
            señal; con HER, sí. No necesita MuJoCo.

  fetch     Brazo robótico simulado en MuJoCo (Gymnasium-Robotics). Linux, macOS o WSL2.

Ejemplos:
    python experiments_3/lab_3_6_her.py bitflip                       # DQN con y sin HER
    python experiments_3/lab_3_6_her.py fetch                         # SAC con y sin HER en FetchReach-v4
    python experiments_3/lab_3_6_her.py fetch --her-only --strategy final,future,episode
    python experiments_3/lab_3_6_her.py fetch --her-only --k 1,4,8
    python experiments_3/lab_3_6_her.py fetch --env FetchPush-v4 --her-only --steps 500000   # ampliación, larga
    python experiments_3/lab_3_6_her.py show --model logs/lab_3_6_FetchReach-v4/her_future_k4/seed_0/best_model.zip

    python experiments_3/plot_results.py logs/lab_3_6_bitflip --metric success
    python experiments_3/plot_results.py logs/lab_3_6_FetchReach-v4 --metric success
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
from stable_baselines3 import DQN, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from common3 import (banner, make_eval_callback, parse_seeds, run_dir, save_config,
                     set_global_seed, tb_dir)


def her_kwargs(strategy: str, k: int) -> dict:
    return dict(replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=k, goal_selection_strategy=strategy))


def configs(args) -> list[tuple[str, dict]]:
    """Lista de (nombre, kwargs de HER) según los argumentos."""
    out = [] if args.her_only else [("sin_her", {})]
    for strategy in args.strategy.split(","):
        for k in [int(x) for x in args.k.split(",")]:
            out.append((f"her_{strategy}_k{k}", her_kwargs(strategy, k)))
    return out


# --------------------------------------------------------------------------- bit-flipping
def run_bitflip(args) -> None:
    from stable_baselines3.common.envs import BitFlippingEnv
    lab = "lab_3_6_bitflip"
    make = lambda: Monitor(BitFlippingEnv(n_bits=args.n_bits, continuous=False, max_steps=args.n_bits))
    hp = dict(learning_rate=1e-3, buffer_size=100_000, learning_starts=1_000, batch_size=256,
              gamma=0.95, train_freq=1, gradient_steps=1, target_update_interval=500,
              exploration_fraction=0.3, exploration_final_eps=0.05,
              policy_kwargs=dict(net_arch=[256, 256]))
    for name, her in configs(args):
        for seed in parse_seeds(args.seeds):
            out = run_dir(lab, name, seed)
            save_config(out, {"env": f"BitFlipping-{args.n_bits}", "steps": args.steps, "seed": seed,
                              **hp, "her": her.get("replay_buffer_kwargs")})
            set_global_seed(seed)
            model = DQN("MultiInputPolicy", make(), seed=seed, tensorboard_log=tb_dir(lab),
                        device=args.device, verbose=0, **hp, **her)
            cb = make_eval_callback(make(), out, args.eval_every, 1, args.eval_episodes)
            banner(f"BitFlipping {args.n_bits} bits · DQN · {name} · seed={seed}")
            model.learn(total_timesteps=args.steps, callback=cb, tb_log_name=f"{name}_s{seed}")
    print(f"\nSiguiente paso:  python experiments_3/plot_results.py logs/{lab} --metric success")


# --------------------------------------------------------------------------- Fetch (MuJoCo)
def make_fetch(env_id: str, render_mode=None):
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    return Monitor(gym.make(env_id, render_mode=render_mode))


def get_algo(name: str):
    if name == "tqc":
        try:
            from sb3_contrib import TQC
        except ImportError:
            raise SystemExit("TQC está en sb3-contrib:  pip install sb3-contrib")
        return TQC
    return {"sac": SAC, "td3": TD3}[name]


def run_fetch(args) -> None:
    lab = f"lab_3_6_{args.env}"
    algo = get_algo(args.algo)
    hp = dict(learning_starts=1_000, batch_size=256, gamma=0.95, tau=0.05)
    for name, her in configs(args):
        for seed in parse_seeds(args.seeds):
            out = run_dir(lab, name, seed)
            save_config(out, {"env": args.env, "algo": args.algo, "steps": args.steps, "seed": seed,
                              **hp, "her": her.get("replay_buffer_kwargs")})
            set_global_seed(seed)
            model = algo("MultiInputPolicy", make_fetch(args.env), seed=seed, tensorboard_log=tb_dir(lab),
                         device=args.device, verbose=0, **hp, **her)
            cb = make_eval_callback(make_fetch(args.env), out, args.eval_every, 1, args.eval_episodes)
            banner(f"{args.env} · {args.algo.upper()} · {name} · seed={seed} · {args.steps:,} pasos")
            model.learn(total_timesteps=args.steps, callback=cb, tb_log_name=f"{name}_s{seed}")
    print(f"\nSiguiente paso:  python experiments_3/plot_results.py logs/{lab} --metric success")


def show(args) -> None:
    """Visualiza una política entrenada: ventana MuJoCo, o GIF con --gif (Colab)."""
    env = make_fetch(args.env, render_mode="rgb_array" if args.gif else "human")
    algo = get_algo(args.algo)
    model = algo.load(args.model, env=env)          # HER necesita el entorno para cargar
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done, info = False, {}
        while not done:
            if args.gif:
                frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        print(f"Episodio {ep}: éxito = {bool(info.get('is_success', False))}")
    env.close()
    if args.gif:
        import imageio
        imageio.mimsave(args.gif, frames[::2], duration=0.06, loop=0)
        print(f"GIF guardado en {args.gif}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="part", required=True)

    def common(sp, steps, eval_every):
        sp.add_argument("--seeds", default="0,1,2")
        sp.add_argument("--steps", type=int, default=steps)
        sp.add_argument("--eval-every", type=int, default=eval_every)
        sp.add_argument("--eval-episodes", type=int, default=20)
        sp.add_argument("--strategy", default="future", help="final, future, episode (admite lista)")
        sp.add_argument("--k", default="4", help="n_sampled_goal (admite lista)")
        sp.add_argument("--her-only", action="store_true", help="no entrenar la versión sin HER")
        sp.add_argument("--device", default="auto")

    b = sub.add_parser("bitflip", help="DQN con y sin HER en BitFlippingEnv")
    common(b, 50_000, 2_000)
    b.add_argument("--n-bits", type=int, default=15)

    f = sub.add_parser("fetch", help="SAC/TD3/TQC con y sin HER en un entorno Fetch")
    common(f, 20_000, 2_000)
    f.add_argument("--env", default="FetchReach-v4", help="FetchReach-v4, FetchPush-v4, FetchSlide-v4, FetchPickAndPlace-v4")
    f.add_argument("--algo", default="sac", choices=["sac", "td3", "tqc"])

    s = sub.add_parser("show", help="ver una política entrenada en MuJoCo")
    s.add_argument("--model", required=True)
    s.add_argument("--env", default="FetchReach-v4")
    s.add_argument("--algo", default="sac", choices=["sac", "td3", "tqc"])
    s.add_argument("--episodes", type=int, default=5)
    s.add_argument("--gif", default=None, help="guarda un GIF en lugar de abrir ventana (Colab: exportad MUJOCO_GL=egl)")

    args = p.parse_args()
    {"bitflip": run_bitflip, "fetch": run_fetch, "show": show}[args.part](args)


if __name__ == "__main__":
    main()
