"""Compara configuraciones a partir de los evaluations.npz de varias semillas.

Uso:
    python plot_results.py logs/lab_3_2                      # todas las configuraciones
    python plot_results.py logs/lab_3_2 --configs base,buffer_min
    python plot_results.py logs/lab_3_3 logs/lab_3_3_sb3     # varias carpetas a la vez
    python plot_results.py logs/lab_3_6_fetch --metric success
    python plot_results.py logs/lab_3_3 --metric score        # Snake: manzanas por partida

Para cada configuración:
    - curva media entre semillas con intervalo de confianza del 95 % (bootstrap);
    - tabla final: media, IC 95 % e IQM (media intercuartílica) del último tramo.

Solo necesita numpy y matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_runs(config_dir: Path, metric: str):
    """Devuelve (timesteps, matriz [n_semillas, n_evaluaciones]) alineada en el tiempo."""
    curves = []
    for f in sorted(config_dir.glob("seed_*/evaluations.npz")):
        data = np.load(f)
        t = data["timesteps"]
        if metric == "success":
            if "successes" not in data or len(data["successes"]) == 0:
                print(f"  (se omite {f.parent.parent.name}: no tiene tasa de éxito)")
                return None, None
            y = np.asarray(data["successes"], dtype=float).mean(axis=1)
        elif metric == "score":
            if "scores" not in data:
                print(f"  (se omite {f.parent.parent.name}: no tiene manzanas por partida)")
                return None, None
            y = np.asarray(data["scores"], dtype=float).mean(axis=1)
        else:
            y = np.asarray(data["results"], dtype=float).mean(axis=1)
        if len(t) > 0:
            curves.append((t, y))
    if not curves:
        return None, None
    # rejilla común: hasta el final de la semilla más corta
    t_end = min(c[0][-1] for c in curves)
    grid = curves[0][0][curves[0][0] <= t_end]
    mat = np.stack([np.interp(grid, t, y) for t, y in curves])
    return grid, mat


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, rng=None):
    """IC del 95 % de la media por bootstrap sobre semillas (axis 0)."""
    rng = rng or np.random.default_rng(0)
    n = values.shape[0]
    if n < 2:
        return values.mean(axis=0), values.mean(axis=0)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2, axis=0)
    hi = np.percentile(means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def iqm(x: np.ndarray) -> float:
    """Media intercuartílica: media del 50 % central (Agarwal et al., 2021)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = int(np.floor(0.25 * n))
    core = x[k:n - k] if n - 2 * k > 0 else x
    return float(core.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="carpetas logs/<lab> con subcarpetas por configuración")
    p.add_argument("--configs", default=None, help="lista separada por comas (por defecto, todas)")
    p.add_argument("--metric", choices=["reward", "success", "score"], default="reward")
    p.add_argument("--last", type=int, default=3, help="nº de evaluaciones finales que se promedian en la tabla")
    p.add_argument("--out", default=None, help="fichero de salida (png/pdf). Por defecto, <primera carpeta>.png")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wanted = set(args.configs.split(",")) if args.configs else None
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rows = []
    for d in args.dirs:
        d = Path(d)
        for cfg in sorted(x for x in d.iterdir() if x.is_dir()):
            if wanted and cfg.name not in wanted:
                continue
            grid, mat = load_runs(cfg, args.metric)
            if grid is None:
                continue
            label = cfg.name if len(args.dirs) == 1 else f"{d.name}/{cfg.name}"
            mean = mat.mean(axis=0)
            lo, hi = bootstrap_ci(mat)
            line, = ax.plot(grid, mean, lw=2, label=f"{label} (n={mat.shape[0]})")
            ax.fill_between(grid, lo, hi, color=line.get_color(), alpha=0.18, lw=0)
            final = mat[:, -args.last:].mean(axis=1)          # una cifra por semilla
            flo, fhi = bootstrap_ci(final[:, None])
            rows.append((label, mat.shape[0], int(grid[-1]), final.mean(), float(flo[0]), float(fhi[0]), iqm(final)))

    if not rows:
        raise SystemExit("No se encontraron evaluations.npz en las carpetas indicadas.")

    ax.set_xlabel("pasos de entorno")
    ax.set_ylabel({"success": "tasa de éxito", "score": "manzanas por partida"}.get(args.metric, "retorno medio en evaluación"))
    ax.set_title(args.title or ", ".join(Path(x).name for x in args.dirs))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out = args.out or (str(Path(args.dirs[0]).with_suffix("")) + ".png")
    fig.savefig(out, dpi=150)

    print(f"\nGráfica guardada en {out}\n")
    head = f"{'configuración':32s} {'sem.':>4s} {'pasos':>9s} {'media':>9s} {'IC 95 %':>20s} {'IQM':>9s}"
    print(head + "\n" + "-" * len(head))
    for label, n, steps, m, lo, hi, q in rows:
        print(f"{label:32s} {n:4d} {steps:9d} {m:9.2f} {f'[{lo:.2f}, {hi:.2f}]':>20s} {q:9.2f}")
    print(f"\n(media de las últimas {args.last} evaluaciones de cada semilla)")


if __name__ == "__main__":
    main()
