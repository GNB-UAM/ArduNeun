#!/usr/bin/env python3
"""
plot_realtime_benchmark.py

Parsea el log de benchmark "NEUN REAL-TIME TEMPORAL RESOLUTION" (uno o varios,
uno por plataforma: ESP32-S3, ESP8266, ...) y genera gráficas comparativas:

  1. Tiempo medio por paso (mean_step_us) vs nº de neuronas, una línea por dt,
     con banda min-max, un panel por modelo (HH, HR, IZH...).
  2. Time-ratio (t_cómputo / t_simulado) vs dt, log-log, con la frontera
     ratio = 1 y el umbral con margen de seguridad, marcando FAIL.
  3. Steps per second vs nº de neuronas.

Formatos de fila aceptados (se detecta por número de columnas):
  9 cols:  model,neurons,dt_ms,mean_step_us,min_step_us,max_step_us,
           steps_per_second,time_ratio,realtime
  10 cols: model,neurons,dt_ms,mean_step_us,min_step_us,max_step_us,
           steps_per_second,time_ratio,utilization_pct,realtime

Uso:
    python plot_realtime_benchmark.py esp32s3.log
    python plot_realtime_benchmark.py esp32s3.log esp8266.log
    python plot_realtime_benchmark.py esp32s3.log --outdir figuras --format pdf
    python plot_realtime_benchmark.py esp32s3.log --safety-margin 0.5
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


def parse_log(path: Path) -> pd.DataFrame:
    """Extrae la plataforma y las filas CSV de un fichero de log de benchmark."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    # Nombre de la plataforma: primera línea útil tras el título del bloque
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
            float(parts[2])  # dt_ms debe ser numérico
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
    # Si el log no traía el porcentaje, se deriva del ratio
    df["utilization_pct"] = df["utilization_pct"].fillna(df["time_ratio"] * 100.0)

    df["platform"] = platform
    df["pass"] = df["realtime"].str.upper() == "PASS"
    return df.dropna(subset=[c for c in NUMERIC_COLS if c != "utilization_pct"])


def plot_mean_step_vs_neurons(df: pd.DataFrame, outdir: Path, fmt: str):
    models = df["model"].unique()
    platforms = df["platform"].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(5.5, 4.5 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for platform in platforms:
            sub_p = sub[sub["platform"] == platform]
            for dt in sorted(sub_p["dt_ms"].unique(), reverse=True):
                line = sub_p[sub_p["dt_ms"] == dt].sort_values("neurons")
                label = (f"{platform} · dt={dt:g} ms" if len(platforms) > 1
                         else f"dt={dt:g} ms")
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
        ax.legend(fontsize=8)

    fig.suptitle("Computational step cost vs. network size")
    fig.tight_layout()
    out = outdir / f"mean_step_vs_neurons.{fmt}"
    fig.savefig(out, dpi=200, format=fmt)
    plt.close(fig)
    return out


def plot_time_ratio_vs_dt(df: pd.DataFrame, outdir: Path, fmt: str,
                          safety_margin: float):
    models = df["model"].unique()
    platforms = df["platform"].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(5.5, 4.5 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for platform in platforms:
            sub_p = sub[sub["platform"] == platform]
            # Red más grande disponible por dt (caso más exigente)
            agg = (sub_p.sort_values("neurons")
                        .groupby("dt_ms", as_index=False)
                        .last()
                        .sort_values("dt_ms"))
            ax.plot(agg["dt_ms"], agg["time_ratio"], marker="o",
                    label=platform)
            fail = agg[~agg["pass"]]
            ax.scatter(fail["dt_ms"], fail["time_ratio"], marker="x",
                       color="red", zorder=5)

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
        ax.legend(fontsize=8)

    fig.suptitle("Real-time margin vs. integration step (x = FAIL)")
    fig.tight_layout()
    out = outdir / f"time_ratio_vs_dt.{fmt}"
    fig.savefig(out, dpi=200, format=fmt)
    plt.close(fig)
    return out


def plot_steps_per_second(df: pd.DataFrame, outdir: Path, fmt: str):
    models = df["model"].unique()
    platforms = df["platform"].unique()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_groups = max(len(models) * len(platforms), 1)
    width = 0.8 / n_groups
    dt_ref = df["dt_ms"].max()
    sizes = sorted(df["neurons"].unique())
    positions = range(len(sizes))

    idx = 0
    for model in models:
        for platform in platforms:
            sub = (df[(df["model"] == model) & (df["platform"] == platform)
                      & (df["dt_ms"] == dt_ref)]
                   .set_index("neurons")
                   .reindex(sizes))
            if sub["steps_per_second"].isna().any():
                print(f"AVISO: {model}/{platform} no tiene datos para todos "
                      f"los tamaños de red")
            label = f"{model} · {platform}" if len(platforms) > 1 else model
            ax.bar([p + idx * width for p in positions],
                   sub["steps_per_second"].fillna(0),
                   width=width, label=label)
            idx += 1

    ax.set_xticks([p + width * (n_groups - 1) / 2 for p in positions])
    ax.set_xticklabels([str(int(n)) for n in sizes])
    ax.set_xlabel("Neurons")
    ax.set_ylabel(f"Steps per second (dt={dt_ref:g} ms)")
    ax.set_title("Performance (steps/s) by network size")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
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
                    help="Uno o más ficheros de log (uno por plataforma)")
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
        print(f"[{log_path.name}] plataforma detectada: {df['platform'].iloc[0]} "
              f"({len(df)} filas, modelos: {', '.join(df['model'].unique())})")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

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