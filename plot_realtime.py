#!/usr/bin/env python3
"""
plot_realtime_benchmark.py

Parsea logs del benchmark "NEUN REAL-TIME TEMPORAL RESOLUTION" (neun_realtime_benchmark.ino)
y genera gráficas comparativas entre configuraciones. Una configuración es la
combinación de:

  - plataforma  (línea "Chip:" del log)
  - cpu         (línea "CPU MHz:")
  - integrador  (columna 'integrator' del log; en logs antiguos, del nombre)
  - precisión   (del nombre del fichero: float, double)

Nombre de fichero esperado (el orden de los tokens no importa):
    real-time-RK4-float.log   esp32s3_rk4_double.log

Las etiquetas solo incluyen los factores que VARÍAN entre los logs pasados.

Formatos de fila aceptados:
  14 cols (nuevo): model,integrator,neurons,dt_ms,mean_tick_us,max_tick_us,
                   min_tick_us,mean_step_us,neuron_steps_per_s,ratio_mean,
                   ratio_worst,est_max_neurons,spikes,realtime
  10 / 9 cols (antiguo, valores por neurona; se convierten a tick completo).

Definiciones:
  - "step" = coste POR NEURONA = tick / nº de neuronas (igual que en las
    versiones anteriores de las figuras).
  - "tick" = toda la red avanza un dt.
  - time_ratio = mean_tick / dt  (para N neuronas en una CPU, todas deben
    avanzar dt dentro de un mismo periodo dt).
  - FAIL (aspa): max_tick >= dt * safety_margin. Se recalcula aquí.

Gráficas:
  1. mean_step_vs_neurons : tiempo medio por paso (banda min-max) vs neuronas,
                            una línea por configuración y dt, un panel por modelo.
  2. max_step_vs_neurons  : tiempo máximo por paso vs neuronas (misma estructura).
  3. time_ratio_vs_dt     : time_ratio vs dt (log-log), una línea por
                            configuración y tamaño de red, con ratio = 1 y margen.

Uso:
    python plot_realtime_benchmark.py real-time-RK4-float.log real-time-RK4-double.log
    python plot_realtime_benchmark.py real-time-*.log --outdir figuras --format pdf
    python plot_realtime_benchmark.py real-time-*.log --ratio-sizes 1 4 16
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS_NEW = [
    "model", "integrator", "neurons", "dt_ms", "mean_tick_us", "max_tick_us",
    "min_tick_us", "mean_step_us", "neuron_steps_per_s", "ratio_mean",
    "ratio_worst", "est_max_neurons", "spikes", "realtime",
]
COLUMNS_10 = [
    "model", "neurons", "dt_ms", "mean_step_us", "min_step_us", "max_step_us",
    "steps_per_second", "time_ratio", "utilization_pct", "realtime",
]
COLUMNS_9 = COLUMNS_10[:8] + ["realtime"]
COLUMNS_BY_LEN = {14: COLUMNS_NEW, 10: COLUMNS_10, 9: COLUMNS_9}

MODEL_LABELS = {
    "HH": "Hodgkin-Huxley",
    "HR": "Hindmarsh-Rose",
    "IZH": "Izhikevich RS",
}

FACTORS = ["platform", "cpu", "integrator", "precision"]

INTEGRATOR_RE = re.compile(
    r"^(rk\d+|rkf\d+|euler|heun|midpoint|dopri\d*|ab\d+)$", re.IGNORECASE)
PRECISION_ALIASES = {"float": "float", "f32": "float",
                     "double": "double", "f64": "double"}


def normalize_integrator(tok: str) -> str:
    return tok.upper() if tok.lower().startswith(("rk", "ab")) else tok.capitalize()


def parse_filename(path: Path):
    """Devuelve (integrador, precisión) deducidos del nombre del fichero."""
    tokens = [t for t in re.split(r"[-_.\s]+", path.stem) if t]
    integrator, precision = "n/a", "n/a"
    for tok in tokens:
        if INTEGRATOR_RE.match(tok):
            integrator = normalize_integrator(tok)
        elif tok.lower() in PRECISION_ALIASES:
            precision = PRECISION_ALIASES[tok.lower()]
    return integrator, precision


def parse_header(lines, path: Path):
    """Extrae plataforma y MHz de la cabecera del log."""
    platform, cpu = None, "n/a"
    for ln in lines:
        if ln.startswith("Chip:"):
            platform = ln.split(":", 1)[1].strip()
        elif ln.startswith("CPU MHz:"):
            cpu = ln.split(":", 1)[1].strip() + " MHz"

    if platform is None:  # logs antiguos: línea tras el título
        for i, ln in enumerate(lines):
            if "TEMPORAL RESOLUTION" in ln.upper():
                for j in range(i + 1, len(lines)):
                    if lines[j] and not set(lines[j]) <= {"="}:
                        platform = lines[j]
                        break
                break
    return platform or path.stem, cpu


def parse_log(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]
    platform, cpu = parse_header(lines, path)

    records = []
    for ln in lines:
        if not ln or ln.startswith("===") or ln.startswith("model,"):
            continue
        parts = [p.strip() for p in ln.split(",")]
        cols = COLUMNS_BY_LEN.get(len(parts))
        if cols is None or not re.match(r"^[A-Za-z0-9_]+$", parts[0]):
            continue
        if parts[-1].upper() not in ("PASS", "FAIL"):
            continue
        rec = dict(zip(cols, parts))
        rec["_legacy"] = len(parts) != 14
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    non_numeric = {"model", "integrator", "realtime", "_legacy"}
    for c in df.columns:
        if c not in non_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Logs antiguos: valores por neurona -> tick completo
    legacy = df["_legacy"].astype(bool)
    if legacy.any():
        n = df["neurons"]
        for new, old in (("mean_tick_us", "mean_step_us"),
                         ("max_tick_us", "max_step_us"),
                         ("min_tick_us", "min_step_us")):
            if new not in df.columns:
                df[new] = np.nan
            df.loc[legacy, new] = df.loc[legacy, old] * n[legacy]
        if "spikes" not in df.columns:
            df["spikes"] = np.nan

    fn_integrator, precision = parse_filename(path)
    if "integrator" not in df.columns:
        df["integrator"] = np.nan
    df["integrator"] = df["integrator"].map(
        lambda v: normalize_integrator(v) if isinstance(v, str) and v else fn_integrator)

    df["platform"] = platform
    df["cpu"] = cpu
    df["precision"] = precision
    df["source"] = path.name

    required = ["neurons", "dt_ms", "mean_tick_us", "max_tick_us",
                "min_tick_us", "mean_step_us"]
    return df.dropna(subset=required).copy()


def derive_metrics(data: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Métricas por paso (por neurona), ratios de tick y PASS/FAIL con el mismo criterio."""
    n = data["neurons"]
    dt_us = data["dt_ms"] * 1000.0

    data["mean_step_us"] = data["mean_tick_us"] / n
    data["min_step_us"] = data["min_tick_us"] / n
    data["max_step_us"] = data["max_tick_us"] / n

    data["time_ratio"] = data["mean_tick_us"] / dt_us   # tick completo / dt
    data["ratio_worst"] = data["max_tick_us"] / dt_us
    data["pass"] = data["max_tick_us"] < dt_us * margin
    return data


def add_config_labels(data: pd.DataFrame) -> pd.DataFrame:
    varying = [f for f in FACTORS if data[f].nunique() > 1]
    if not varying:
        varying = FACTORS
    data["config"] = data[varying].astype(str).agg(" · ".join, axis=1)

    # Variante = factores que varían salvo el integrador (el integrador fija el color)
    others = [f for f in varying if f != "integrator"]
    data["variant"] = (data[others].astype(str).agg(" · ".join, axis=1)
                       if others else "")

    for cfg, n in data.groupby("config")["source"].nunique().items():
        if n > 1:
            print(f"AVISO: la configuración '{cfg}' aparece en {n} logs "
                  f"distintos; se mezclarán. Revisa los nombres de fichero.")
    return data


def warn_inactive(data: pd.DataFrame):
    """Avisa si alguna configuración no produjo spikes (régimen no representativo)."""
    if "spikes" not in data.columns:
        return
    sp = data.dropna(subset=["spikes"])
    for (model, cfg), g in sp.groupby(["model", "config"]):
        silent = g[g["spikes"] == 0]
        if len(silent):
            dts = ", ".join(f"{d:g}" for d in sorted(silent["dt_ms"].unique()))
            print(f"AVISO: {model} / {cfg}: 0 spikes con dt = {dts} ms. "
                  f"El régimen puede no ser representativo.")


# Una familia de color por integrador (mapas secuenciales: claro -> oscuro)
INTEGRATOR_CMAPS = ["Blues", "Oranges", "Greens", "Reds", "Purples",
                    "YlOrBr", "Greys", "PuRd"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":"]


def ordered_configs(df):
    """Configuraciones ordenadas por integrador y variante (agrupa la leyenda)."""
    info = (df.drop_duplicates("config")[["config", "integrator", "variant"]]
              .sort_values(["integrator", "variant"]))
    return list(info["config"])


class GroupedStyle:
    """Color por integrador (familia) + tono según un índice (dt o tamaño).

    Si dentro de un mismo integrador hay varias variantes (plataforma, CPU,
    precisión...), se distinguen con marcador y tipo de línea.
    """

    def __init__(self, df):
        self.info = (df.drop_duplicates("config")
                       .set_index("config")[["integrator", "variant"]])
        integrators = sorted(df["integrator"].unique())
        self.cmap = {i: INTEGRATOR_CMAPS[k % len(INTEGRATOR_CMAPS)]
                     for k, i in enumerate(integrators)}
        variants = sorted(df["variant"].unique())
        self.variant_style = {v: (MARKERS[k % len(MARKERS)],
                                  LINESTYLES[k % len(LINESTYLES)])
                              for k, v in enumerate(variants)}

    def get(self, cfg, idx, n):
        """(color, marker, linestyle) para la config; idx de 0..n-1 (claro->oscuro)."""
        integ = self.info.loc[cfg, "integrator"]
        var = self.info.loc[cfg, "variant"]
        t = 0.75 if n <= 1 else 0.4 + 0.55 * idx / (n - 1)
        color = plt.get_cmap(self.cmap[integ])(t)
        marker, ls = self.variant_style[var]
        return color, marker, ls


def place_legend(ax, n_lines, fontsize):
    """Leyenda dentro si hay pocas líneas; fuera (a la derecha) si hay muchas."""
    if n_lines > 8:
        ax.legend(fontsize=fontsize, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    else:
        ax.legend(fontsize=fontsize)


def fig_width(n_lines, base):
    return base if n_lines <= 8 else base + 3.5


# ------------------------------------------------------------
# Figuras 1 y 2: coste por paso vs neuronas
# ------------------------------------------------------------

def _plot_step_vs_neurons(df, outdir: Path, fmt: str, ycol: str, ylabel: str,
                          suptitle: str, fname: str, band: bool):
    models = df["model"].unique()
    configs = ordered_configs(df)
    dts_desc = sorted(df["dt_ms"].unique(), reverse=True)
    style = GroupedStyle(df)
    n_lines = len(configs) * len(dts_desc)
    fig, axes = plt.subplots(len(models), 1,
                             figsize=(9.0, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for cfg in configs:
            sub_c = sub[sub["config"] == cfg]
            for dt in sorted(sub_c["dt_ms"].unique(), reverse=True):
                line = sub_c[sub_c["dt_ms"] == dt].sort_values("neurons")
                label = f"{cfg} · dt={dt:g} ms"
                color, marker, ls = style.get(
                    cfg, dts_desc.index(dt), len(dts_desc))
                (handle,) = ax.plot(line["neurons"], line[ycol],
                                    marker=marker, linestyle=ls, label=label,
                                    color=color)
                if band:
                    ax.fill_between(line["neurons"], line["min_step_us"],
                                    line["max_step_us"], alpha=0.15,
                                    color=handle.get_color())
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Neurons")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)
        place_legend(ax, n_lines, 7)

    fig.suptitle(suptitle)
    fig.tight_layout()
    out = outdir / f"{fname}.{fmt}"
    fig.savefig(out, dpi=200, format=fmt)
    plt.close(fig)
    return out


def plot_mean_step_vs_neurons(df, outdir, fmt):
    return _plot_step_vs_neurons(
        df, outdir, fmt, "mean_step_us", "Mean time per step (µs)",
        "Computational step cost vs. network size (band = min–max)",
        "mean_step_vs_neurons", band=True)


def plot_max_step_vs_neurons(df, outdir, fmt):
    return _plot_step_vs_neurons(
        df, outdir, fmt, "max_step_us", "Max time per step (µs)",
        "Worst-case step cost vs. network size (max tick / neurons)",
        "max_step_vs_neurons", band=False)


# ------------------------------------------------------------
# Figura 3: time_ratio vs dt
# ------------------------------------------------------------

def plot_time_ratio_vs_dt(df, outdir: Path, fmt: str, safety_margin: float,
                          ratio_sizes=None):
    models = df["model"].unique()
    configs = ordered_configs(df)
    all_sizes = [int(s) for s in sorted(df["neurons"].unique())
                 if not ratio_sizes or int(s) in ratio_sizes]
    style = GroupedStyle(df)
    n_lines = len(configs) * len(all_sizes) + 2
    fig, axes = plt.subplots(len(models), 1,
                             figsize=(9, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for cfg in configs:
            sub_c = sub[sub["config"] == cfg]
            sizes = sorted(sub_c["neurons"].unique())
            if ratio_sizes:
                sizes = [s for s in sizes if int(s) in ratio_sizes]
            for n in sizes:
                line = (sub_c[sub_c["neurons"] == n]
                        .drop_duplicates("dt_ms").sort_values("dt_ms"))
                color, marker, ls = style.get(
                    cfg, all_sizes.index(int(n)), len(all_sizes))
                (handle,) = ax.plot(line["dt_ms"], line["time_ratio"],
                                    marker=marker, linestyle=ls,
                                    label=f"{cfg} · {int(n)} neurons",
                                    color=color)
                fail = line[~line["pass"]]
                ax.scatter(fail["dt_ms"], fail["time_ratio"], marker="x",
                           color=handle.get_color(), s=60, zorder=5)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
                   label="Real-time limit \n(ratio = 1)")
        ax.axhline(safety_margin, color="gray", linestyle=":", linewidth=1,
                   label=f"Safety margin \n(ratio = {safety_margin:g})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("dt (ms)")
        ax.set_ylabel("time_ratio (t_tick / dt)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.grid(True, which="both", alpha=0.3)
        place_legend(ax, n_lines, 6)

    fig.suptitle("Real-time margin vs. integration step "
                 "(x = worst tick exceeds the safety margin)")
    fig.tight_layout()
    out = outdir / f"time_ratio_vs_dt.{fmt}"
    fig.savefig(out, dpi=200, format=fmt)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path,
                    help="Logs de benchmark (p. ej. real-time-RK4-float.log)")
    ap.add_argument("--outdir", type=Path, default=Path("figuras"),
                    help="Carpeta de salida (default: figuras/)")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                    help="Formato de las figuras (default: png)")
    ap.add_argument("--safety-margin", type=float, default=0.5,
                    help="SAFETY_MARGIN usado en el firmware (default: 0.5)")
    ap.add_argument("--ratio-sizes", type=int, nargs="+", default=None,
                    help="Tamaños de red a dibujar en time_ratio_vs_dt "
                         "(default: todos), p. ej. --ratio-sizes 1 4 16")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="Ruta opcional para exportar los datos combinados")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for log_path in args.logs:
        if not log_path.exists():
            sys.exit(f"No existe el fichero: {log_path}")
        df = parse_log(log_path)
        if df.empty:
            sys.exit(f"No se han encontrado filas de datos válidas en {log_path}")
        r = df.iloc[0]
        kind = "antiguo (convertido a tick)" if r["_legacy"] else "nuevo"
        print(f"[{log_path.name}] formato {kind}: plataforma={r['platform']}, "
              f"cpu={r['cpu']}, integrador={r['integrator']}, "
              f"precisión={r['precision']} ({len(df)} filas, "
              f"modelos: {', '.join(df['model'].unique())})")
        if r["precision"] == "n/a":
            print(f"  AVISO: no se pudo deducir la precisión del nombre "
                  f"'{log_path.name}'. Usa p. ej. real-time-RK4-float.log")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = derive_metrics(data, args.safety_margin)
    data = add_config_labels(data)
    warn_inactive(data)

    if args.csv_out:
        data.to_csv(args.csv_out, index=False)
        print(f"Datos combinados exportados a {args.csv_out}")

    out1 = plot_mean_step_vs_neurons(data, args.outdir, args.format)
    out2 = plot_max_step_vs_neurons(data, args.outdir, args.format)
    out3 = plot_time_ratio_vs_dt(data, args.outdir, args.format,
                                 args.safety_margin, args.ratio_sizes)

    print("Figuras generadas:")
    for f in (out1, out2, out3):
        print(f"  - {f}")


if __name__ == "__main__":
    main()