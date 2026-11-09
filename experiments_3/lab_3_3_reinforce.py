"""Laboratorio 3.3 · REINFORCE desde cero en el Snake  —  ESQUELETO DE LA ENTREGA (15 %)

Este fichero es el punto de partida de vuestra entrega. Ya está resuelto todo lo que
NO es el algoritmo: argumentos, semillas, bucle de entrenamiento, evaluación,
TensorBoard y guardado de resultados en el formato de plot_results.py.

La nota no depende de cuántas manzanas coma vuestro agente, sino de cómo llegáis a él:
el diario de iteraciones (DIARIO.md) es el núcleo de la entrega. Ver el README.

Completáis los huecos TODO, en orden:

    TODO 1  PolicyNet          red de política que devuelve logits
    TODO 2  reward_to_go       retornos descontados desde cada paso
    TODO 3  collect_episode    jugar una partida con la política y guardar lo necesario
    TODO 4  policy_loss        pérdida sustituta de REINFORCE (versión vanilla)
    TODO 5  ValueNet + baseline
    TODO 6  snake_design.py · SnakeObservation   vuestra observación
    TODO 7  snake_design.py · SnakeReward        vuestra recompensa (si decidís cambiarla)

Los TODO 1–5 son un requisito: tienen que pasar --check y aprender en CartPole.

Comprobad cada paso antes de entrenar:

    python experiments_3/lab_3_3_reinforce.py --check

Cada iteración del diario se lanza con su nombre (--tag). Explorar barato, confirmar caro:

    # exploración: 2 semillas, 100 000 pasos
    python experiments_3/lab_3_3_reinforce.py --obs original --seeds 0,1 --steps 100000 --tag it01_original
    python experiments_3/lab_3_3_reinforce.py --obs custom --seeds 0,1 --steps 100000 --tag it02_peligro

    # confirmación de las configuraciones finales: 5 semillas, 200 000 pasos
    python experiments_3/lab_3_3_reinforce.py --obs custom --seeds 0-4 --tag it07_final
    python experiments_3/lab_3_3_reinforce.py --obs custom --baseline --seeds 0-4 --tag it08_final_baseline
    python experiments_3/lab_3_3_referencia_sb3.py --obs custom --seeds 0-4 --tag it09_ref

    # tabla del diario (manzanas, intervalo, cómo acaban las partidas) y curvas
    python experiments_3/diario.py logs/lab_3_3
    python experiments_3/plot_results.py logs/lab_3_3 --metric score

Ver jugar al agente (o guardar un GIF en Colab con --gif):

    python experiments_3/lab_3_3_reinforce.py --obs custom --watch logs/lab_3_3/it02_peligro/seed_0/policy.pt

Si algo no aprende y no sabéis si es el algoritmo o el diseño, probad en CartPole:

    python experiments_3/lab_3_3_reinforce.py --env CartPole-v1 --seeds 0 --steps 50000

Reglas: PyTorch y Gymnasium sí; ninguna implementación ya hecha del algoritmo.
Podéis usar asistentes de IA: indicad en DIARIO.md para qué.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from common3 import parse_seeds, run_dir, save_config, save_evaluations, set_global_seed, tb_dir
from snake_design import check_design, design_fingerprint, evaluate_snake, make_snake_env

LAB = "lab_3_3"


# =========================================================================== TODO 1
class PolicyNet(nn.Module):
    """πθ(a|s) para acciones discretas.

    forward(obs) recibe un tensor [B, obs_dim] (o [obs_dim]) y devuelve logits
    [B, n_actions] (o [n_actions]): un número real por acción, sin softmax.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        # TODO 1: una MLP con 2 capas ocultas de `hidden` unidades y activación tanh,
        #         y una capa de salida lineal con n_actions unidades.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),          # salida lineal: logits
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO 1: devuelve los logits.
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Distribución sobre acciones. Ya está hecha: usadla en collect_episode."""
        return torch.distributions.Categorical(logits=self.forward(obs))


# =========================================================================== TODO 2
def reward_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    """Gₜ = Rₜ₊₁ + γ Rₜ₊₂ + γ² Rₜ₊₃ + …  para cada paso t del episodio.

    rewards[t] es la recompensa recibida tras la acción del paso t.
    Devuelve un tensor float32 de la misma longitud que rewards.
    Ejemplo: reward_to_go([1, 1, 1], 0.5) -> tensor([1.75, 1.50, 1.00])
    Pista: se calcula en una sola pasada recorriendo el episodio hacia atrás.
    """
    # TODO 2
    returns = torch.zeros(len(rewards), dtype=torch.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


# =========================================================================== TODO 3
def collect_episode(env: gym.Env, policy: PolicyNet, seed: int | None = None) -> dict:
    """Juega UNA partida completa muestreando acciones de la política.

    Debe devolver un diccionario con:
        "obs":       tensor float32 [T, obs_dim]  observaciones en las que se actuó
        "actions":   tensor long    [T]           acciones tomadas
        "log_probs": tensor float32 [T]           log πθ(Aₜ|Sₜ), CON gradiente
        "rewards":   list[float] de longitud T
        "return":    float, suma de recompensas sin descontar
        "length":    int, T
        "score":     int, manzanas comidas (info["score"] al final; 0 si el entorno no lo da)

    Notas:
      - Usad env.reset(seed=seed): seed puede ser None (el entorno sigue su secuencia).
      - La partida acaba si terminated o truncated.
      - Pregunta para el informe: en el Snake, ¿qué diferencia hay entre acabar por
        terminated (choque) y por truncated (demasiados pasos sin comer)? ¿Afecta a
        REINFORCE? ¿Y al baseline?
    """
    # TODO 3
    obs, info = env.reset(seed=seed)
    obs_list, actions, log_probs, rewards = [], [], [], []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
        dist = policy.distribution(obs_t)
        action = dist.sample()
        obs_list.append(obs_t)
        actions.append(action)
        log_probs.append(dist.log_prob(action))       # con gradiente
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        rewards.append(float(reward))
        done = terminated or truncated
    return {
        "obs": torch.stack(obs_list),                  # [T, obs_dim]
        "actions": torch.stack(actions).long(),        # [T]
        "log_probs": torch.stack(log_probs),           # [T]
        "rewards": rewards,
        "return": float(sum(rewards)),
        "length": len(rewards),
        "score": int(info.get("score", 0)),
    }


# =========================================================================== TODO 5 (red)
class ValueNet(nn.Module):
    """v̂(s, w): recibe [B, obs_dim] y devuelve [B] (un valor por estado)."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        # TODO 5: MLP con 2 capas ocultas (tanh) y una salida escalar.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO 5: devuelve un tensor [B] (ojo con la última dimensión).
        return self.net(obs).squeeze(-1)               # [B, 1] -> [B]


# =========================================================================== TODO 4 y 5
def policy_loss(episodes: list[dict], gamma: float, value_net: ValueNet | None = None,
                normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pérdida sustituta de REINFORCE para un lote de episodios.

    Devuelve (loss_pi, loss_v):
      - loss_pi: escalar cuyo gradiente respecto a θ es −∇J(θ) estimado con el lote.
      - loss_v:  escalar para entrenar la red de valor, o None si value_net es None.

    TODO 4 (vanilla, value_net=None):
      - Gₜ de cada episodio con reward_to_go.
      - Si normalize: restad la media y dividid por la desviación de G en el lote
        (sumad 1e-8 al denominador).
      - loss_pi = −(log_probs · G), promediado sobre los pasos del lote.

    TODO 5 (con baseline, value_net no es None):
      - Ventaja Âₜ = Gₜ − v̂(Sₜ). El baseline NO debe recibir gradiente desde loss_pi.
      - loss_v = error cuadrático medio entre v̂(Sₜ) y Gₜ.
      - Decidid y justificad en el informe si normalizáis G, Â o ninguno.
    """
    # TODO 4 / TODO 5
    log_probs = torch.cat([ep["log_probs"] for ep in episodes])                       # [N]
    returns = torch.cat([reward_to_go(ep["rewards"], gamma) for ep in episodes])      # [N]

    if value_net is None:
        weights = returns
        if normalize:
            weights = (weights - weights.mean()) / (weights.std() + 1e-8)
        return -(log_probs * weights).mean(), None

    obs = torch.cat([ep["obs"] for ep in episodes])               # [N, obs_dim]
    values = value_net(obs)                                       # [N], con gradiente
    advantages = returns - values.detach()                        # Â, sin gradiente hacia w
    if normalize:                                                 # se normaliza Â, no G
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss_pi = -(log_probs * advantages).mean()
    loss_v = ((values - returns) ** 2).mean()
    return loss_pi, loss_v


# =========================================================================== (hecho)
def make_env(args) -> gym.Env:
    if args.env == "snake":
        return make_snake_env(args.obs, args.reward, args.grid)
    return gym.make(args.env)


def config_name(args) -> str:
    if args.tag:                                      # nombre de la iteración del diario
        return args.tag
    name = "reinforce_baseline" if args.baseline else "reinforce"
    if args.env == "snake":
        return f"{name}_{args.obs}_{args.reward}"
    return f"{name}_{args.env}"


@torch.no_grad()
def evaluate(policy: PolicyNet, env: gym.Env, n_episodes: int, seed: int):
    """Evalúa la política determinista (acción más probable) en un entorno aparte.

    Devuelve (retornos, manzanas por partida, cómo acabó cada partida); en entornos
    que no son el Snake, las dos últimas son None.
    """
    act = lambda obs: int(torch.argmax(policy(torch.as_tensor(obs, dtype=torch.float32))).item())
    if "score" in env.reset(seed=seed)[1]:                # Snake: manzanas y causa de fin
        ends = []
        rets, scores = evaluate_snake(act, env, n_episodes, seed, ends)
        return rets, scores, ends
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(act(obs))
            ret += float(r)
            done = term or trunc
        returns.append(ret)
    return returns, None, None


def train(args, seed: int) -> None:
    from torch.utils.tensorboard import SummaryWriter

    config = config_name(args)
    out = run_dir(LAB, config, seed)
    save_config(out, {**vars(args), "seed": seed, "config": config,
                      **(design_fingerprint() if args.env == "snake" else {})})
    set_global_seed(seed)

    env, eval_env = make_env(args), make_env(args)
    env.action_space.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions, args.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    value_net, v_opt = None, None
    if args.baseline:
        value_net = ValueNet(obs_dim, args.hidden)
        v_opt = torch.optim.Adam(value_net.parameters(), lr=args.lr_value)

    writer = SummaryWriter(f"{tb_dir(LAB)}/{config}_s{seed}")
    steps, n_ep, next_eval = 0, 0, 0
    eval_t, eval_r, eval_s, eval_e = [], [], [], []
    t0 = time.time()
    print(f"\n== REINFORCE · {config} · seed={seed} · {args.steps:,} pasos ==")

    def do_eval():
        rets, scores, ends = evaluate(policy, eval_env, args.eval_episodes, seed=10_000 * (seed + 1))
        eval_t.append(steps); eval_r.append(rets)
        if scores is not None:
            eval_s.append(scores); eval_e.append(ends)
        save_evaluations(out, eval_t, eval_r, scores=eval_s or None, ends=eval_e or None)
        writer.add_scalar("eval/mean_reward", float(np.mean(rets)), steps)
        extra = ""
        if scores is not None:
            writer.add_scalar("eval/mean_score", float(np.mean(scores)), steps)
            extra = f"  ·  manzanas {np.mean(scores):5.2f}"
        print(f"  pasos {steps:>9,}  ·  retorno {np.mean(rets):8.2f}{extra}  ·  {time.time() - t0:5.0f} s")

    while steps < args.steps:
        if steps >= next_eval:                       # evaluación periódica (incluye t = 0)
            do_eval()
            next_eval += args.eval_every

        batch = []
        for _ in range(args.batch_episodes):
            ep = collect_episode(env, policy, seed=seed if n_ep == 0 else None)
            batch.append(ep)
            n_ep += 1
            steps += ep["length"]
            writer.add_scalar("rollout/ep_return", ep["return"], steps)
            writer.add_scalar("rollout/ep_score", ep.get("score", 0), steps)

        loss_pi, loss_v = policy_loss(batch, args.gamma, value_net, args.normalize)
        opt.zero_grad()
        loss_pi.backward()
        opt.step()
        writer.add_scalar("train/policy_loss", loss_pi.item(), steps)
        if loss_v is not None:
            v_opt.zero_grad()
            loss_v.backward()
            v_opt.step()
            writer.add_scalar("train/value_loss", loss_v.item(), steps)

    do_eval()
    torch.save(policy.state_dict(), out / "policy.pt")
    writer.close()
    print(f"  resultados en {out}")


def watch(args) -> None:
    """Ver jugar a una política guardada (policy.pt): ventana, o GIF con --gif (Colab)."""
    mode = "rgb_array" if args.gif else "human"
    env = make_snake_env(args.obs, args.reward, args.grid, render_mode=mode)
    obs_dim = int(np.prod(env.observation_space.shape))
    policy = PolicyNet(obs_dim, env.action_space.n, args.hidden)
    policy.load_state_dict(torch.load(args.watch))
    frames, scores, ends = [], [], []
    with torch.no_grad():
        for ep in range(1 if args.gif else 3):
            obs, info = env.reset(seed=123 + ep)
            done = False
            while not done:
                if args.gif:
                    frames.append(env.render())
                a = int(torch.argmax(policy(torch.as_tensor(obs, dtype=torch.float32))).item())
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
            scores.append(info["score"]); ends.append(info.get("fin"))
    env.close()
    print("Manzanas por partida:", scores)
    print("Cómo acabó cada partida:", ends)
    if args.gif:
        import imageio
        imageio.mimsave(args.gif, frames, duration=0.08, loop=0)
        print(f"GIF guardado en {args.gif}")


# =========================================================================== autocomprobaciones
def self_check(args) -> None:
    """Comprobaciones rápidas de cada TODO. No garantizan que el algoritmo aprenda."""
    ok = lambda msg: print(f"  [OK] {msg}")

    print("TODO 2 · reward_to_go")
    g = reward_to_go([1.0, 1.0, 1.0], 0.5)
    assert torch.is_tensor(g) and g.dtype == torch.float32, "debe devolver un tensor float32"
    assert torch.allclose(g, torch.tensor([1.75, 1.5, 1.0])), f"esperado [1.75, 1.5, 1.0], obtenido {g}"
    assert torch.allclose(reward_to_go([0.0, 0.0, 10.0], 0.9), torch.tensor([8.1, 9.0, 10.0]))
    ok("valores correctos (incluido el ejercicio de la U1: G₀ = 8,1)")

    env = make_snake_env("original", "sparse", args.grid)
    obs_dim = env.observation_space.shape[0]

    print("TODO 1 · PolicyNet")
    torch.manual_seed(0)
    pol = PolicyNet(obs_dim, 4)
    assert pol(torch.zeros(7, obs_dim)).shape == (7, 4), f"forward([7, {obs_dim}]) debe dar [7, 4]"
    assert pol(torch.zeros(obs_dim)).shape == (4,), f"forward([{obs_dim}]) debe dar [4]"
    ok("formas de salida correctas")

    print("TODO 3 · collect_episode")
    ep = collect_episode(env, pol, seed=0)
    T = ep["length"]
    for k in ("obs", "actions", "log_probs", "rewards", "return", "length", "score"):
        assert k in ep, f"falta la clave '{k}'"
    assert ep["obs"].shape == (T, obs_dim) and ep["actions"].shape == (T,) and ep["log_probs"].shape == (T,)
    assert len(ep["rewards"]) == T and abs(ep["return"] - sum(ep["rewards"])) < 1e-6
    assert ep["log_probs"].requires_grad, "log_probs debe conservar el gradiente"
    re = pol.distribution(ep["obs"]).log_prob(ep["actions"])
    assert torch.allclose(re, ep["log_probs"].detach(), atol=1e-5), "log_probs no corresponde a las acciones guardadas"
    ok(f"partida de {T} pasos coherente")

    print("TODO 4 · policy_loss (vanilla)")
    eps = [collect_episode(env, pol, seed=s) for s in (1, 2)]
    for e in eps:                                    # todas las recompensas positivas
        e["rewards"] = [1.0] * e["length"]
    before = sum(pol.distribution(e["obs"]).log_prob(e["actions"]).sum() for e in eps).item()
    loss_pi, loss_v = policy_loss(eps, 0.99, None, normalize=False)
    assert loss_pi.dim() == 0, "loss_pi debe ser un escalar"
    assert loss_v is None, "sin value_net, loss_v debe ser None"
    sgd = torch.optim.SGD(pol.parameters(), lr=1e-3)
    sgd.zero_grad(); loss_pi.backward(); sgd.step()
    after = sum(pol.distribution(e["obs"]).log_prob(e["actions"]).sum() for e in eps).item()
    assert after > before, "con retornos positivos, un paso debe AUMENTAR la probabilidad de las acciones tomadas: revisad el signo"
    ok("el signo de la pérdida es correcto")

    print("TODO 5 · ValueNet y baseline")
    try:
        vnet = ValueNet(obs_dim)
    except NotImplementedError:
        print("  [--] ValueNet sin implementar: se omite (necesaria para --baseline)")
        vnet = None
    if vnet is not None:
        assert vnet(torch.zeros(5, obs_dim)).shape == (5,), f"ValueNet([5, {obs_dim}]) debe dar [5]"
        eps = [collect_episode(env, pol, seed=s) for s in (3, 4)]
        loss_pi, loss_v = policy_loss(eps, 0.99, vnet, normalize=True)
        assert loss_v is not None and loss_v.dim() == 0, "con value_net, loss_v debe ser un escalar"
        vnet.zero_grad(); loss_pi.backward()
        leaked = any(p.grad is not None and p.grad.abs().sum() > 0 for p in vnet.parameters())
        assert not leaked, "loss_pi está propagando gradiente a la red de valor: falta un detach()"
        ok("la pérdida del actor no entrena al crítico")

    print("TODO 6 y 7 · diseño (snake_design.py)")
    for obs, rew in (("custom", "sparse"), ("original", "custom")):
        try:
            check_design(obs, rew, args.grid)
        except NotImplementedError as e:
            print(f"  [--] obs={obs}, reward={rew}: sin implementar ({e})")
    print("\nAutocomprobaciones terminadas.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="ejecuta las autocomprobaciones y termina")
    p.add_argument("--watch", default=None, help="ruta a un policy.pt: ver jugar a la política")
    p.add_argument("--gif", default=None, help="con --watch: guarda un GIF en lugar de abrir ventana (Colab)")
    p.add_argument("--env", default="snake", help="'snake' o un entorno discreto de Gymnasium (CartPole-v1)")
    p.add_argument("--obs", default="original", choices=["original", "custom"])
    p.add_argument("--reward", default="sparse", choices=["sparse", "distance", "custom"])
    p.add_argument("--grid", type=int, default=12, help="tamaño del tablero del Snake")
    p.add_argument("--seeds", default="0", help="'0,1,2' o '0-4'")
    p.add_argument("--tag", default=None, help="nombre de la iteración (carpeta de logs), p. ej. it03_peligro")
    p.add_argument("--steps", type=int, default=200_000, help="presupuesto de pasos de entorno")
    p.add_argument("--baseline", action="store_true", help="usa la red de valor como baseline")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-value", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-episodes", type=int, default=8, help="partidas por actualización")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    args = p.parse_args()

    if args.check:
        self_check(args)
        return
    if args.watch:
        watch(args)
        return
    for seed in parse_seeds(args.seeds):
        train(args, seed)
    print("\nSiguiente paso:  python experiments_3/plot_results.py logs/lab_3_3 --metric score")


if __name__ == "__main__":
    main()
