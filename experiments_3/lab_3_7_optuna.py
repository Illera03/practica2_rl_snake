"""Laboratorio 3.7 · Ajustar PPO con Optuna y defender el resultado.

Dos fases, como en el protocolo de las diapositivas:

  search   búsqueda con Optuna (TPE + MedianPruner). Cada trial entrena PPO por tramos,
           evalúa al final de cada tramo y se poda si va peor que la mediana.
  final    reentrena la mejor configuración con semillas NUEVAS (no usadas al ajustar)
           y guarda las curvas para plot_results.py.

Ejemplos:
    python experiments_3/lab_3_7_optuna.py search --trials 20
    optuna-dashboard sqlite:///optuna/optuna.db           # seguir la búsqueda
    python experiments_3/lab_3_7_optuna.py final --seeds 100-104
    python experiments_3/plot_results.py logs/lab_3_7 logs/lab_3_4 --configs ppo_tuned,ppo_default,a2c_default

Como en el Lab 2.10, reutilizar --study continúa una búsqueda ya empezada.
Si va justo de tiempo:  --chunks 3 --chunk-steps 40000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from common3 import (banner, make_eval_callback, parse_seeds, run_dir, save_config,
                     set_global_seed, tb_dir)

LAB = "lab_3_7"
STORAGE_DIR = Path("optuna")


def sample_params(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda. log=True para parámetros que varían en órdenes de magnitud."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        n_steps=trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        n_epochs=trial.suggest_categorical("n_epochs", [4, 10]),
        gamma=trial.suggest_float("gamma", 0.95, 0.9999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        ent_coef=trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
    )


def make_objective(args):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        seed = trial.number                    # semillas 0, 1, 2…: las de ajuste
        env = make_vec_env(args.env, n_envs=args.n_envs, seed=seed)
        eval_env = make_vec_env(args.env, n_envs=1, seed=50_000 + seed)
        model = PPO("MlpPolicy", env, seed=seed, device=args.device, verbose=0, **params)
        mean = -np.inf
        for k in range(args.chunks):
            model.learn(args.chunk_steps, reset_num_timesteps=False)
            mean, _ = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)
            trial.report(mean, k)
            print(f"  trial {trial.number:3d} · tramo {k + 1}/{args.chunks} · retorno {mean:8.1f}")
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(mean)
    return objective


def search(args) -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=args.study, storage=f"sqlite:///{STORAGE_DIR}/optuna.db", load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    banner(f"Optuna · PPO · {args.env} · {args.trials} trials × {args.chunks} tramos de {args.chunk_steps:,} pasos")
    study.optimize(make_objective(args), n_trials=args.trials, gc_after_trial=True)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nTrials completos: {len(done)} · podados: {len(pruned)}")
    print(f"Mejor retorno: {study.best_value:.1f}")
    print("Mejores hiperparámetros:\n" + json.dumps(study.best_params, indent=2))
    (STORAGE_DIR / f"{args.study}_best.json").write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")


def final(args) -> None:
    best_file = STORAGE_DIR / f"{args.study}_best.json"
    if best_file.exists():
        params = json.loads(best_file.read_text(encoding="utf-8"))
    else:
        study = optuna.load_study(study_name=args.study, storage=f"sqlite:///{STORAGE_DIR}/optuna.db")
        params = study.best_params
    print("Hiperparámetros:", json.dumps(params))
    for seed in parse_seeds(args.seeds):
        out = run_dir(LAB, "ppo_tuned", seed)
        save_config(out, {"env": args.env, "steps": args.steps, "seed": seed, "n_envs": args.n_envs, **params})
        set_global_seed(seed)
        env = make_vec_env(args.env, n_envs=args.n_envs, seed=seed)
        eval_env = make_vec_env(args.env, n_envs=1, seed=10_000 + seed)
        model = PPO("MlpPolicy", env, seed=seed, tensorboard_log=tb_dir(LAB),
                    device=args.device, verbose=0, **params)
        cb = make_eval_callback(eval_env, out, args.eval_every, args.n_envs, args.eval_episodes)
        banner(f"PPO ajustado · seed={seed} · {args.steps:,} pasos")
        model.learn(total_timesteps=args.steps, callback=cb, tb_log_name=f"ppo_tuned_s{seed}")
        print(f"Mejor retorno en evaluación: {cb.best_mean_reward:.1f}")
    print("\nSiguiente paso:  python experiments_3/plot_results.py logs/lab_3_7 logs/lab_3_4")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)
    for name in ("search", "final"):
        sp = sub.add_parser(name)
        sp.add_argument("--env", default="LunarLander-v3")
        sp.add_argument("--study", default="ppo_lunar")
        sp.add_argument("--n-envs", type=int, default=8)
        sp.add_argument("--eval-episodes", type=int, default=10)
        sp.add_argument("--device", default="auto")
    s = sub.choices["search"]
    s.add_argument("--trials", type=int, default=20)
    s.add_argument("--chunks", type=int, default=5)
    s.add_argument("--chunk-steps", type=int, default=40_000)
    f = sub.choices["final"]
    f.add_argument("--seeds", default="100-104", help="semillas NUEVAS, distintas de las de ajuste")
    f.add_argument("--steps", type=int, default=500_000, help="mismo presupuesto que el Lab 3.4")
    f.add_argument("--eval-every", type=int, default=10_000)
    args = p.parse_args()
    {"search": search, "final": final}[args.phase](args)


if __name__ == "__main__":
    main()
