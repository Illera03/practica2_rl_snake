"""Tabla del diario de iteraciones del Snake, generada a partir de los logs.

Lee logs/lab_3_3/<iteración>/seed_*/config.json y evaluations.npz y escribe una fila
por iteración. Pegad la tabla en DIARIO.md: los números del informe salen de aquí.

    python experiments_3/diario.py logs/lab_3_3
    python experiments_3/diario.py logs/lab_3_3 --out tabla_diario.md
    python experiments_3/diario.py logs/lab_3_3 --configs it01_original,it02_peligro

Columnas:
    Manzanas      media de las últimas evaluaciones (--last) de cada semilla, promediada entre semillas
    IC 95 %       intervalo de confianza de esa media (bootstrap sobre semillas)
    Pared/Cuerpo/Hambre   cómo acabaron las partidas de esas mismas evaluaciones (%)
    Cambia        qué difiere respecto a la fila anterior (argumentos o código del diseño).
                  Si aparece más de una cosa, la iteración no cambió una sola variable.

Para nombrar cada iteración, lanzad los entrenamientos con --tag (it01_original, it02_peligro…).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import bootstrap_ci, iqm  # noqa: E402

# Qué campos de config.json describen una iteración (nombre en la tabla).
FIELDS = {
    "algo": "algoritmo", "baseline": "baseline", "obs": "obs", "reward": "recompensa",
    "obs_code": "código de la observación", "reward_code": "código de la recompensa",
    "steps": "pasos", "lr": "lr", "lr_value": "lr del crítico", "gamma": "γ",
    "batch_episodes": "partidas por lote", "hidden": "neuronas", "normalize": "normalización",
    "grid": "tablero",
}
ENDS = ("pared", "cuerpo", "hambre", "victoria")


def load_iteration(folder: Path, last: int):
    seeds = sorted(folder.glob("seed_*"))
    cfg, finals, ends, steps = None, [], Counter(), []
    for sd in seeds:
        npz, cj = sd / "evaluations.npz", sd / "config.json"
        if not npz.exists():
            continue
        data = np.load(npz)
        if "scores" not in data:
            continue
        scores = np.asarray(data["scores"], dtype=float).mean(axis=1)
        finals.append(scores[-last:].mean())
        steps.append(int(data["timesteps"][-1]))
        if "ends" in data:
            for row in np.asarray(data["ends"])[-last:]:
                ends.update(str(e) for e in row)
        if cfg is None and cj.exists():
            cfg = json.loads(cj.read_text(encoding="utf-8"))
    if not finals:
        return None
    cfg = cfg or {}
    if "algo" not in cfg:
        cfg["algo"] = "reinforce"
    return {"name": folder.name, "cfg": cfg, "finals": np.array(finals), "ends": ends, "steps": min(steps)}


def describe_algo(cfg: dict) -> str:
    algo = str(cfg.get("algo", "reinforce"))
    if algo == "reinforce":
        return "REINFORCE + baseline" if cfg.get("baseline") else "REINFORCE"
    return algo.upper()


def diff(prev: dict | None, cfg: dict) -> str:
    if prev is None:
        return "punto de partida"
    same_algo = prev.get("algo") == cfg.get("algo")
    changed = []
    for k, label in FIELDS.items():
        if k == "obs_code" and not (prev.get("obs") == cfg.get("obs") == "custom"):
            continue                       # el código solo cuenta si ambas usan la observación propia
        if k == "reward_code" and not (prev.get("reward") == cfg.get("reward") == "custom"):
            continue
        if not same_algo and k not in ("algo", "obs", "reward", "obs_code", "reward_code", "steps", "grid"):
            continue                       # hiperparámetros de algoritmos distintos no se comparan
        if prev.get(k) != cfg.get(k):
            changed.append(label)
    return ", ".join(changed) if changed else "nada (¿repetición?)"


def pct(c: Counter, key: str) -> str:
    n = sum(c.values())
    return f"{100 * c[key] / n:.0f} %" if n else "—"


def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logdir", nargs="?", default="logs/lab_3_3")
    p.add_argument("--configs", default=None, help="iteraciones separadas por comas (por defecto, todas)")
    p.add_argument("--last", type=int, default=3, help="evaluaciones finales que se promedian")
    p.add_argument("--out", default=None, help="guardar la tabla en un fichero .md")
    args = p.parse_args()

    root = Path(args.logdir)
    if not root.exists():
        raise SystemExit(f"No existe {root}. ¿Habéis entrenado algo con --tag?")
    wanted = set(args.configs.split(",")) if args.configs else None
    its = []
    for folder in sorted(x for x in root.iterdir() if x.is_dir()):
        if wanted and folder.name not in wanted:
            continue
        it = load_iteration(folder, args.last)
        if it is not None:
            its.append(it)
    if not its:
        raise SystemExit("No hay evaluaciones del Snake en esa carpeta.")

    head = ("| Iteración | Algoritmo | Obs | Recompensa | Semillas | Pasos | Manzanas | IC 95 % | IQM "
            "| Pared | Cuerpo | Hambre | Cambia respecto a la anterior |")
    lines = [head, "|" + "---|" * 13]
    prev = None
    for it in its:
        cfg, f = it["cfg"], it["finals"]
        lo, hi = bootstrap_ci(f[:, None])
        ci = f"[{fmt(float(lo[0]))}, {fmt(float(hi[0]))}]" if len(f) > 1 else "— (1 semilla)"
        lines.append(
            f"| {it['name']} | {describe_algo(cfg)} | {cfg.get('obs', '?')} | {cfg.get('reward', '?')} "
            f"| {len(f)} | {miles(it['steps'])} | {fmt(f.mean())} | {ci} | {fmt(iqm(f))} "
            f"| {pct(it['ends'], 'pared')} | {pct(it['ends'], 'cuerpo')} | {pct(it['ends'], 'hambre')} "
            f"| {diff(prev, cfg)} |")
        prev = cfg
    table = "\n".join(lines)
    print(table)
    print(f"\n(Manzanas: media de las últimas {args.last} evaluaciones de cada semilla; "
          "Pared/Cuerpo/Hambre: cómo acabaron esas partidas)")
    if any(len(it["finals"]) < 5 for it in its):
        print("Aviso: las filas con menos de 5 semillas son exploración; las conclusiones van con 5.")
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"Tabla guardada en {args.out}")


if __name__ == "__main__":
    main()
