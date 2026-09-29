#!/usr/bin/env python3
"""
plot_realtime_benchmark.py

Parsea uno o varios logs del benchmark "NEUN REAL-TIME TEMPORAL RESOLUTION" y
genera gráficas comparativas entre configuraciones. Una configuración es la
combinación de:

  - plataforma  (se lee del contenido del log: ESP32-S3, ESP8266, ...)
  - integrador  (se lee del nombre del fichero: RK4, RK6, Euler, ...)
  - precisión   (se lee del nombre del fichero: float, double)

Nombre de fichero esperado (el orden de los tokens no importa):
    real-time-RK4-float.log
    real-time-RK6-double.log
    esp32s3_rk4_double.log

Las etiquetas de las gráficas solo incluyen los factores que VARÍAN entre los
logs pasados. Ej.: si todos son ESP32-S3 y float pero cambia el integrador, la
leyenda mostrará solo "RK4" / "RK6".

Gráficas:
  1. Tiempo medio por paso vs nº de neuronas (banda min-max), una línea por
     configuración y dt, un panel por modelo.
  2. time_ratio vs dt (log-log) con frontera ratio = 1 y margen de seguridad.
  3. Steps per second vs nº de neuronas.

Formatos de fila aceptados (por nº de columnas):
  9 cols:  model,neurons,dt_ms,mean_step_us,min_step_us,max_step_us,
           steps_per_second,time_ratio,realtime
  10 cols: model,neurons,dt_ms,mean_step_us,min_step_us,max_step_us,
           steps_per_second,time_ratio,utilization_pct,realtime

Uso:
    python plot_realtime_benchmark.py real-time-RK4-float.log real-time-RK4-double.log
    python plot_realtime_benchmark.py real-time-*.log --outdir figuras --format pdf
    python plot_realtime_benchmark.py real-time-RK4-float.log --safety-margin 0.5
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLUMNS_9 = [
    "model", "neurons", "dt_ms", "mean_step_us", "min_step_us", "max_step_us",
    "steps_per_second", "time_ratio", "realtime",
]
COLUMNS_10 = [
    "model", "neurons", "dt_ms", "mean_step_us", "min_step_us", "max_step_us",
    "steps_per_second", "time_ratio", "utilization_pct", "realtime",
]
COLUMNS_BY_LEN = {len(COLUMNS_9): COLUMNS_9, len(COLUMNS_10): COLUMNS_10}

NUMERIC_COLS = ["neurons", "dt_ms", "mean_step_us", "min_step_us",
                "max_step_us", "steps_per_second", "time_ratio",
                "utilization_pct"]

MODEL_LABELS = {
    "HH": "Hodgkin-Huxley",
    "HR": "Hindmarsh-Rose",
    "IZH": "Izhikevich RS",
}

# Factores que definen una configuración, en el orden en que salen en la leyenda
FACTORS = ["platform", "integrator", "precision"]

INTEGRATOR_RE = re.compile(
    r"^(rk\d+|rkf\d+|euler|heun|midpoint|dopri\d*|ab\d+)$", re.IGNORECASE)
PRECISION_ALIASES = {"float": "float", "f32": "float",
                     "double": "double", "f64": "double"}


def parse_filename(path: Path):
    """Devuelve (integrador, precisión) deducidos del nombre del fichero."""
    tokens = [t for t in re.split(r"[-_.\s]+", path.stem) if t]
    integrator, precision = "n/a", "n/a"
    for tok in tokens:
        low = tok.lower()
        if INTEGRATOR_RE.match(tok):
            integrator = tok.upper() if low.startswith(("rk", "ab")) else tok.capitalize()
        elif low in PRECISION_ALIASES:
            precision = PRECISION_ALIASES[low]
    return integrator, precision


def parse_log(path: Path) -> pd.DataFrame:
    """Extrae plataforma y filas CSV de un log; integrador/precisión vienen del nombre."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    platform = path.stem
    for i, ln in enumerate(lines):
        if "TEMPORAL RESOLUTION" in ln.upper():
            for j in range(i + 1, len(lines)):
                if lines[j] and not set(lines[j]) <= {"="}:
                    platform = lines[j]
                    break
            break

    records = []
    for ln in lines:
        if not ln or ln.startswith("===") or ln.startswith("model,"):
            continue
        parts = [p.strip() for p in ln.split(",")]
        cols = COLUMNS_BY_LEN.get(len(parts))
        if cols is None:
            continue
        if not re.match(r"^[A-Za-z0-9_]+$", parts[0]):
            continue
        if parts[-1].upper() not in ("PASS", "FAIL"):
            continue
        try:
            float(parts[2])
        except ValueError:
            continue
        records.append(dict(zip(cols, parts)))

    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "utilization_pct" not in df.columns:
        df["utilization_pct"] = pd.NA
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["utilization_pct"] = df["utilization_pct"].fillna(df["time_ratio"] * 100.0)

    integrator, precision = parse_filename(path)
    df["platform"] = platform
    df["integrator"] = integrator
    df["precision"] = precision
    df["source"] = path.name
    df["pass"] = df["realtime"].str.upper() == "PASS"
    return df.dropna(subset=[c for c in NUMERIC_COLS if c != "utilization_pct"])


def add_config_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Crea la columna 'config' con solo los factores que varían entre logs."""
    varying = [f for f in FACTORS if data[f].nunique() > 1]
    if not varying:
        # Un único tipo de configuración: etiqueta descriptiva completa
        varying = FACTORS
    data["config"] = data[varying].astype(str).agg(" · ".join, axis=1)

    dup = data.groupby("config")["source"].nunique()
    for cfg, n in dup.items():
        if n > 1:
            print(f"AVISO: la configuración '{cfg}' aparece en {n} logs distintos; "
                  f"se mezclarán. Revisa los nombres de fichero.")
    return data


def plot_mean_step_vs_neurons(df, outdir: Path, fmt: str):
    models = df["model"].unique()
    configs = df["config"].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(6, 4.5 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for cfg in configs:
            sub_c = sub[sub["config"] == cfg]
            for dt in sorted(sub_c["dt_ms"].unique(), reverse=True):
                line = sub_c[sub_c["dt_ms"] == dt].sort_values("neurons")
                label = f"{cfg} · dt={dt:g} ms"
                (handle,) = ax.plot(line["neurons"], line["mean_step_us"],
                                    marker="o", label=label)
                ax.fill_between(line["neurons"], line["min_step_us"],
                                line["max_step_us"], alpha=0.15,
                                color=handle.get_color())
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Mean time per step (µs)")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Computational step cost vs. network size")
    fig.tight_layout()
    out = outdir / f"mean_step_vs_neurons.{fmt}"
    fig.savefig(out, dpi=200, format=fmt)
    plt.close(fig)
    return out


def plot_time_ratio_vs_dt(df, outdir: Path, fmt: str, safety_margin: float):
    models = df["model"].unique()
    configs = df["config"].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(6, 4.5 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for cfg in configs:
            sub_c = sub[sub["config"] == cfg]
            # Red más grande disponible por dt (caso más exigente)
            agg = (sub_c.sort_values("neurons")
                        .groupby("dt_ms", as_index=False)
                        .last()
                        .sort_values("dt_ms"))
            (handle,) = ax.plot(agg["dt_ms"], agg["time_ratio"], marker="o",
                                label=cfg)
            fail = agg[~agg["pass"]]
            ax.scatter(fail["dt_ms"], fail["time_ratio"], marker="x",
                       color=handle.get_color(), s=60, zorder=5)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
                   label="Real-time limit (ratio = 1)")
        ax.axhline(safety_margin, color="gray", linestyle=":", linewidth=1,
                   label=f"Safety margin (ratio = {safety_margin:g})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("dt (ms)")
        ax.set_ylabel("time_ratio (t_computation / t_simulated)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Real-time margin vs. integration step (x = FAIL)")
    fig.tight_layout()
    out = outdir / f"time_ratio_vs_dt.{fmt}"
    fig.savefig(out, dpi=200, format=fmt)
    plt.close(fig)
    return out


def plot_steps_per_second(df, outdir: Path, fmt: str):
    models = df["model"].unique()
    configs = df["config"].unique()
    fig, ax = plt.subplots(figsize=(8, 4.8))

    n_groups = max(len(models) * len(configs), 1)
    width = 0.8 / n_groups
    dt_ref = df["dt_ms"].max()
    sizes = sorted(df["neurons"].unique())
    positions = range(len(sizes))

    idx = 0
    for model in models:
        for cfg in configs:
            sub = (df[(df["model"] == model) & (df["config"] == cfg)
                      & (df["dt_ms"] == dt_ref)]
                   .drop_duplicates("neurons")
                   .set_index("neurons")
                   .reindex(sizes))
            if sub["steps_per_second"].isna().any():
                print(f"AVISO: {model} / {cfg} no tiene datos para todos los "
                      f"tamaños de red con dt={dt_ref:g} ms")
            ax.bar([p + idx * width for p in positions],
                   sub["steps_per_second"].fillna(0),
                   width=width, label=f"{model} · {cfg}")
            idx += 1

    ax.set_xticks([p + width * (n_groups - 1) / 2 for p in positions])
    ax.set_xticklabels([str(int(n)) for n in sizes])
    ax.set_xlabel("Neurons")
    ax.set_ylabel(f"Steps per second (dt={dt_ref:g} ms)")
    ax.set_title("Performance (steps/s) by network size")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = outdir / f"steps_per_second.{fmt}"
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
        print(f"[{log_path.name}] plataforma={r['platform']}, "
              f"integrador={r['integrator']}, precisión={r['precision']} "
              f"({len(df)} filas, modelos: {', '.join(df['model'].unique())})")
        if r["integrator"] == "n/a" or r["precision"] == "n/a":
            print(f"  AVISO: no se pudo deducir integrador/precisión del nombre "
                  f"'{log_path.name}'. Usa p. ej. real-time-RK4-float.log")
        frames.append(df)

    data = add_config_labels(pd.concat(frames, ignore_index=True))

    if args.csv_out:
        data.to_csv(args.csv_out, index=False)
        print(f"Datos combinados exportados a {args.csv_out}")

    out1 = plot_mean_step_vs_neurons(data, args.outdir, args.format)
    out2 = plot_time_ratio_vs_dt(data, args.outdir, args.format,
                                 args.safety_margin)
    out3 = plot_steps_per_second(data, args.outdir, args.format)

    print("Figuras generadas:")
    for f in (out1, out2, out3):
        print(f"  - {f}")


if __name__ == "__main__":
    main()