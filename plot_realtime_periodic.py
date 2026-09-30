#!/usr/bin/env python3
"""
plot_periodic_deadline.py

Parsea logs del test "NEUN PERIODIC DEADLINE TEST" (neun_periodic_deadline_test.ino)
y genera gráficas de plazos y jitter. Una configuración es la combinación de:

  - plataforma  (línea "Chip:" del log)
  - cpu         (línea "CPU MHz:")
  - núcleo RT   (línea "RT core:")
  - integrador  (columna 'integrator' de cada fila)
  - precisión   (línea "Precision:" del log o, si falta, del nombre del fichero)

Las etiquetas solo incluyen los factores que VARÍAN entre los logs pasados.

Formato de fila (15 columnas):
  model,integrator,neurons,dt_ms,ticks_done,mean_resp_us,max_resp_us,
  mean_jitter_us,max_jitter_us,util_mean,util_worst,margin_viol,misses,
  aborted,verdict

Gráficas:
  1. Mapa de veredictos (neuronas x dt) por modelo y configuración, con el
     ratio del peor caso (max_resp / dt) anotado en cada celda.
  2. Ratio del peor caso vs nº de neuronas (una línea por dt).
  3. Jitter máximo de inicio vs nº de neuronas, con el periodo de cada dt
     como referencia.
  4. Tasa de plazos incumplidos (%) vs nº de neuronas.

Uso:
    python plot_periodic_deadline.py periodic-RK4-float.log
    python plot_periodic_deadline.py periodic-*.log --outdir figuras --format pdf
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


COLUMNS = [
    "model", "integrator", "neurons", "dt_ms", "ticks_done", "mean_resp_us",
    "max_resp_us", "mean_jitter_us", "max_jitter_us", "util_mean",
    "util_worst", "margin_viol", "misses", "aborted", "verdict",
]

NUMERIC_COLS = [
    c for c in COLUMNS
    if c not in ("model", "integrator", "verdict")
]

VERDICTS_VERBOSED = ("missing-rate = 0", "0 < missing-rate <= 10 ", "aborted or missing-rate > 10")
VERDICTS = ("PASS", "WARN", "FAIL")


MODEL_LABELS = {
    "HH": "Hodgkin-Huxley",
    "HR": "Hindmarsh-Rose",
    "IZH": "Izhikevich RS",
}


FACTORS = [
    "platform",
    "cpu",
    "core",
    "integrator",
    "precision",
]


INTEGRATOR_RE = re.compile(
    r"^(rk\d+|rkf\d+|euler|heun|midpoint|dopri\d*|ab\d+)$",
    re.IGNORECASE,
)

PRECISION_ALIASES = {
    "float": "float",
    "f32": "float",
    "double": "double",
    "f64": "double",
}


# ------------------------------------------------------------
# Colores y estilos
# ------------------------------------------------------------

# Misma familia de colores que en plot_realtime_benchmark.py.
#
# Cada integrador recibe una familia:
#
#   RK4     -> Blues
#   Euler   -> Oranges
#   Heun    -> Greens
#   ...
#
# Dentro de una familia, el dt determina el tono:
# claro -> oscuro.
INTEGRATOR_CMAPS = [
    "Blues",
    "Oranges",
    "Greens",
    "Reds",
    "Purples",
    "YlOrBr",
    "Greys",
    "PuRd",
]

MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
]

LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
]


# Los colores del mapa de veredictos se mantienen independientes,
# porque aquí el color tiene significado:
#
#   PASS = verde
#   WARN = amarillo
#   FAIL = rojo
VERDICT_CODE = {
    "PASS": 0,
    "WARN": 1,
    "FAIL": 2,
}

VERDICT_COLORS = [
    "#9af09d",
    "#f7a062",
    "#f1716f",
]


def normalize_integrator(tok: str) -> str:
    return (
        tok.upper()
        if tok.lower().startswith(("rk", "ab"))
        else tok.capitalize()
    )


def precision_from_filename(path: Path) -> str:
    for tok in re.split(r"[-_.\s]+", path.stem):
        if tok.lower() in PRECISION_ALIASES:
            return PRECISION_ALIASES[tok.lower()]
    return "n/a"



def place_legend(ax, n_lines, fontsize):
    """Leyenda dentro si hay pocas líneas; fuera (a la derecha) si hay muchas."""
    if n_lines > 8:
        ax.legend(fontsize=fontsize, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    else:
        ax.legend(fontsize=fontsize)
# ------------------------------------------------------------
# Lectura de logs
# ------------------------------------------------------------

def parse_log(path: Path) -> pd.DataFrame:
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = [
        ln.strip()
        for ln in text.splitlines()
    ]

    header = {
        "platform": path.stem,
        "cpu": "n/a",
        "core": "n/a",
        "precision": None,
    }

    for ln in lines:
        if ln.startswith("Chip:"):
            header["platform"] = ln.split(":", 1)[1].strip()

        elif ln.startswith("CPU MHz:"):
            header["cpu"] = (
                ln.split(":", 1)[1].strip()
                + " MHz"
            )

        elif ln.startswith("RT core:"):
            header["core"] = (
                "core "
                + ln.split(":", 1)[1].strip()
            )

        elif ln.startswith("Precision:"):
            p = ln.split(":", 1)[1].strip().lower()
            header["precision"] = PRECISION_ALIASES.get(p, p)

    if header["precision"] is None:
        header["precision"] = precision_from_filename(path)

    records = []

    for ln in lines:
        if (
            not ln
            or ln.startswith("===")
            or ln.startswith("model,")
        ):
            continue

        parts = [
            p.strip()
            for p in ln.split(",")
        ]

        if len(parts) != len(COLUMNS):
            continue

        if not re.match(
            r"^[A-Za-z0-9_]+$",
            parts[0],
        ):
            continue

        if parts[-1].upper() not in VERDICTS:
            continue

        records.append(
            dict(zip(COLUMNS, parts))
        )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=NUMERIC_COLS
    ).copy()

    df["verdict"] = (
        df["verdict"]
        .str.upper()
    )

    df["integrator"] = (
        df["integrator"]
        .map(normalize_integrator)
    )

    df["aborted"] = (
        df["aborted"].astype(int) == 1
    )

    df["dt_ms"] = (
        df["dt_ms"].round(6)
    )

    df["neurons"] = (
        df["neurons"].astype(int)
    )

    df["miss_rate_pct"] = np.where(
        df["ticks_done"] > 0,
        df["misses"]
        / df["ticks_done"]
        * 100.0,
        np.nan,
    )

    for k, v in header.items():
        df[k] = v

    df["source"] = path.name

    return df


# ------------------------------------------------------------
# Configuraciones
# ------------------------------------------------------------

def add_config_labels(
    data: pd.DataFrame,
) -> pd.DataFrame:

    varying = [
        f
        for f in FACTORS
        if data[f].nunique() > 1
    ]

    if not varying:
        varying = FACTORS

    data["config"] = (
        data[varying]
        .astype(str)
        .agg(" · ".join, axis=1)
    )

    # Igual que en plot_realtime_benchmark.py:
    #
    # El integrador controla el color.
    # Los demás factores que varían controlan
    # marcador y tipo de línea.
    others = [
        f
        for f in varying
        if f != "integrator"
    ]

    data["variant"] = (
        data[others]
        .astype(str)
        .agg(" · ".join, axis=1)
        if others
        else ""
    )

    for cfg, n in (
        data.groupby("config")["source"]
        .nunique()
        .items()
    ):
        if n > 1:
            print(
                f"AVISO: la configuración '{cfg}' "
                f"aparece en {n} logs distintos; "
                f"se mezclarán. Revisa los nombres "
                f"de fichero."
            )

    return data


def ordered_configs(df):
    """
    Configuraciones ordenadas por integrador y variante,
    igual que en plot_realtime_benchmark.py.
    """
    info = (
        df.drop_duplicates("config")
        [
            [
                "config",
                "integrator",
                "variant",
            ]
        ]
        .sort_values(
            [
                "integrator",
                "variant",
            ]
        )
    )

    return list(
        info["config"]
    )


class GroupedStyle:
    """
    Color por integrador + tono según dt.

    Integrador:
        -> familia de color.

    dt:
        -> tono dentro de la familia,
           claro -> oscuro.

    Variante:
        -> marcador + tipo de línea.
    """

    def __init__(self, df):

        self.info = (
            df.drop_duplicates("config")
            .set_index("config")
            [
                [
                    "integrator",
                    "variant",
                ]
            ]
        )

        integrators = sorted(
            df["integrator"].unique()
        )

        self.cmap = {
            integ: INTEGRATOR_CMAPS[
                k % len(INTEGRATOR_CMAPS)
            ]
            for k, integ in enumerate(
                integrators
            )
        }

        variants = sorted(
            df["variant"].unique()
        )

        self.variant_style = {
            variant: (
                MARKERS[
                    k % len(MARKERS)
                ],
                LINESTYLES[
                    k % len(LINESTYLES)
                ],
            )
            for k, variant in enumerate(
                variants
            )
        }

    def get(
        self,
        cfg,
        idx,
        n,
    ):
        """
        Devuelve:

            color, marker, linestyle

        idx:
            posición del dt.

        n:
            número total de dt.
        """

        integ = self.info.loc[
            cfg,
            "integrator",
        ]

        variant = self.info.loc[
            cfg,
            "variant",
        ]

        # Exactamente la misma interpolación
        # que usa plot_realtime_benchmark.py.
        if n <= 1:
            t = 0.75
        else:
            t = (
                0.4
                + 0.55
                * idx
                / (n - 1)
            )

        color = plt.get_cmap(
            self.cmap[integ]
        )(t)

        marker, linestyle = (
            self.variant_style[variant]
        )

        return (
            color,
            marker,
            linestyle,
        )


# ------------------------------------------------------------
# Estilos de líneas
# ------------------------------------------------------------

def _draw_lines(
    ax,
    sub,
    configs,
    dts,
    style,
    ycol,
):
    """
    Una línea por (config, dt).

    El color depende del integrador y dt.
    El marcador y linestyle dependen de la variante.

    Los FAIL aparecen como una x del mismo color
    que la línea correspondiente.
    """

    for cfg in configs:

        sub_c = sub[
            sub["config"] == cfg
        ]

        for dt in dts:

            line = (
                sub_c[
                    sub_c["dt_ms"] == dt
                ]
                .sort_values("neurons")
            )

            if line.empty:
                continue

            color, marker, ls = (
                style.get(
                    cfg,
                    dts.index(dt),
                    len(dts),
                )
            )

            ax.plot(
                line["neurons"],
                line[ycol],
                marker=marker,
                linestyle=ls,
                color=color,
                label=(
                    f"{cfg} · "
                    f"dt={dt:g} ms"
                ),
            )

            fail = line[
                line["verdict"] == "FAIL"
            ]

            if not fail.empty:
                ax.scatter(
                    fail["neurons"],
                    fail[ycol],
                    marker="x",
                    color=color,
                    s=70,
                    zorder=5,
                )


def _panels(
    models,
    height=4,
    width=9.0,
):
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(
            width,
            height * len(models),
        ),
        squeeze=False,
    )

    return fig, axes[:, 0]


def _finish(
    fig,
    outdir,
    fmt,
    name,
):
    fig.tight_layout()

    out = (
        outdir
        / f"{name}.{fmt}"
    )

    fig.savefig(
        out,
        dpi=200,
        format=fmt,
    )

    plt.close(fig)

    return out


# ------------------------------------------------------------
# 1. Mapa de veredictos
# ------------------------------------------------------------

def plot_verdict_heatmap(
    df,
    outdir,
    fmt,
):
    models = list(
        df["model"].unique()
    )

    configs = list(
        df["config"].unique()
    )

    sizes = sorted(
        df["neurons"].unique()
    )

    dts = sorted(
        df["dt_ms"].unique(),
        reverse=True,
    )

    size_idx = {
        n: j
        for j, n in enumerate(sizes)
    }

    dt_idx = {
        d: i
        for i, d in enumerate(dts)
    }

    cmap = ListedColormap(
        VERDICT_COLORS
    )

    cmap.set_bad("white")

    fig, axes = plt.subplots(
        len(configs),
        len(models),
        figsize=(
            4.4 * len(models) + 0.5,
            3.4 * len(configs) + 0.9,
        ),
        squeeze=False,
    )

    for r, cfg in enumerate(configs):

        for c, model in enumerate(models):

            ax = axes[r][c]

            grid = np.full(
                (
                    len(dts),
                    len(sizes),
                ),
                np.nan,
            )

            sub = df[
                (df["model"] == model)
                & (df["config"] == cfg)
            ]

            for _, row in sub.iterrows():

                i = dt_idx[
                    row["dt_ms"]
                ]

                j = size_idx[
                    row["neurons"]
                ]

                grid[i, j] = (
                    VERDICT_CODE[
                        row["verdict"]
                    ]
                )

                # if row["miss_rate_pct"] > 20:
                #     txt = "failed"
                # elif row["miss_rate_pct"] > 10:
                #     txt = "warning"
                # else:
                txt = (
                    f"{(row['misses']/row['ticks_done'])*100:.0f}"
                    # f"{row['misses']:.0f}/{row['ticks_done']:.0f}"
                )

                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

            ax.imshow(
                np.ma.masked_invalid(grid),
                cmap=cmap,
                vmin=0,
                vmax=2,
                aspect="auto",
            )

            ax.set_xticks(
                range(len(sizes))
            )

            ax.set_xticklabels(
                [str(n) for n in sizes]
            )

            ax.set_yticks(
                range(len(dts))
            )

            ax.set_yticklabels(
                [f"{d:g}" for d in dts]
            )


            if c == 0:
                ax.set_ylabel(f"{cfg}\ndt (ms)")

            if r==0:
                title = MODEL_LABELS.get(
                    model,
                    model,
                )
            # if len(configs) > 1:
            #     title += (
            #         f"\n{cfg}"
            #     )
            # if r == 0:
                ax.set_title(
                    title,
                    fontsize=10,
                )
            if r == len(configs)-1:
                ax.set_xlabel("# Neurons")
        

    fig.legend(
        handles=[
            Patch(
                color=col,
                label=v,
            )
            for col, v in zip(
                VERDICT_COLORS,
                VERDICTS_VERBOSED,
            )
        ],
        loc="lower center",
        ncol=3,
        fontsize=10,
    )

    fig.suptitle(
        "Real-time verdict "
        "(cell value = percentage of misses)"
    )

    fig.tight_layout(
        rect=(
            0,
            0.05,
            1,
            0.96,
        )
    )

    out = (
        outdir
        / f"verdict_map.{fmt}"
    )

    fig.savefig(
        out,
        dpi=200,
        format=fmt,
    )

    plt.close(fig)

    return out


# ------------------------------------------------------------
# 2. Ratio del peor caso vs neuronas
# ------------------------------------------------------------

def plot_worst_ratio(
    df,
    outdir,
    fmt,
    margin,
):
    models = list(
        df["model"].unique()
    )

    configs = ordered_configs(df)

    dts = sorted(
        df["dt_ms"].unique(),
        reverse=True,
    )

    style = GroupedStyle(df)

    fig, axes = _panels(
        models,
        width=9.0,
    )

    for ax, model in zip(
        axes,
        models,
    ):

        _draw_lines(
            ax,
            df[
                df["model"] == model
            ],
            configs,
            dts,
            style,
            "util_worst",
        )

        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Deadline \n(ratio = 1)",
        )

        ax.axhline(
            margin,
            color="gray",
            linestyle=":",
            linewidth=1,
            label=(
                f"Safety margin\n "
                f"(ratio = {margin:g})"
            ),
        )

        ax.set_title(
            MODEL_LABELS.get(
                model,
                model,
            )
        )

        ax.set_xlabel(
            "Neurons"
        )

        ax.set_ylabel(
            "Worst response time / dt"
        )

        ax.set_xscale(
            "log",
            base=2,
        )

        ax.set_yscale("log")

        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )

        place_legend(ax, 9, 7)

    fig.suptitle(
        "Worst-case response vs. network size "
        "(x = FAIL)"
    )

    return _finish(
        fig,
        outdir,
        fmt,
        "worst_ratio_vs_neurons",
    )


# ------------------------------------------------------------
# 3. Jitter máximo vs neuronas
# ------------------------------------------------------------

def plot_jitter(
    df,
    outdir,
    fmt,
):
    models = list(
        df["model"].unique()
    )

    configs = ordered_configs(df)

    dts = sorted(
        df["dt_ms"].unique(),
        reverse=True,
    )

    style = GroupedStyle(df)

    fig, axes = _panels(
        models,
        width=9.0,
    )

    for ax, model in zip(
        axes,
        models,
    ):

        sub = df[
            df["model"] == model
        ].copy()

        sub["max_jitter_us"] = (
            sub["max_jitter_us"]
            .clip(lower=1e-3)
        )

        _draw_lines(
            ax,
            sub,
            configs,
            dts,
            style,
            "max_jitter_us",
        )

        # Periodo de cada dt.
        #
        # Usamos la misma familia de color
        # correspondiente al integrador.
        #
        # Si hay varios integradores, las líneas
        # de referencia se dibujan con la familia
        # de color del primer integrador. El significado
        # de la línea sigue siendo exclusivamente el dt.
        if configs:

            reference_cfg = configs[0]

            for dt in dts:

                color, _, _ = style.get(
                    reference_cfg,
                    dts.index(dt),
                    len(dts),
                )

                ax.axhline(
                    dt * 1000.0,
                    color=color,
                    linestyle=":",
                    linewidth=1,
                )

        ax.set_title(
            MODEL_LABELS.get(
                model,
                model,
            )
        )

        ax.set_xlabel(
            "Neurons"
        )

        ax.set_ylabel(
            "Max start jitter (µs)"
        )

        ax.set_xscale(
            "log",
            base=2,
        )

        ax.set_yscale("log")

        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )

        place_legend(ax, 9, 7)


    fig.suptitle(
        "Max start jitter "
        "(dotted = period of each dt; x = FAIL)"
    )

    return _finish(
        fig,
        outdir,
        fmt,
        "jitter_vs_neurons",
    )


# ------------------------------------------------------------
# 4. Tasa de plazos incumplidos
# ------------------------------------------------------------

def plot_miss_rate(
    df,
    outdir,
    fmt,
):
    models = list(
        df["model"].unique()
    )

    configs = ordered_configs(df)

    dts = sorted(
        df["dt_ms"].unique(),
        reverse=True,
    )

    style = GroupedStyle(df)

    fig, axes = _panels(
        models,
        width=9.0,
    )

    for ax, model in zip(
        axes,
        models,
    ):

        sub = df[
            df["model"] == model
        ]

        _draw_lines(
            ax,
            sub,
            configs,
            dts,
            style,
            "miss_rate_pct",
        )

        ab = sub[
            sub["aborted"]
        ]

        if not ab.empty:
            ax.scatter(
                ab["neurons"],
                ab["miss_rate_pct"],
                marker="s",
                facecolors="none",
                edgecolors="black",
                s=110,
                zorder=6,
                label=(
                    "aborted "
                    "(fell behind)"
                ),
            )

        ax.set_title(
            MODEL_LABELS.get(
                model,
                model,
            )
        )

        ax.set_xlabel(
            "Neurons"
        )

        ax.set_ylabel(
            "Missed deadlines "
            "(% of ticks)"
        )

        ax.set_xscale(
            "log",
            base=2,
        )

        ax.set_yscale(
            "symlog",
            linthresh=1e-3,
        )

        ax.set_ylim(
            bottom=0
        )

        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )

        place_legend(ax, 9, 7)

    fig.suptitle(
        "Deadline miss rate "
        "(symlog axis; x = FAIL)"
    )

    return _finish(
        fig,
        outdir,
        fmt,
        "miss_rate_vs_neurons",
    )


# ------------------------------------------------------------
# Resumen por consola
# ------------------------------------------------------------

def print_summary(df):

    print(
        "\nResumen: mayor nº de neuronas "
        "probado por dt"
    )

    print(
        "  (sin_fallos = PASS o WARN sin abortar; "
        "margen_ok = PASS)"
    )

    for (
        model,
        cfg,
    ), g in df.groupby(
        ["model", "config"]
    ):

        print(
            f"\n  "
            f"{MODEL_LABELS.get(model, model)}"
            f" · {cfg}"
        )

        for dt in sorted(
            g["dt_ms"].unique(),
            reverse=True,
        ):

            gd = g[
                g["dt_ms"] == dt
            ]

            ok_hard = gd[
                (gd["verdict"] != "FAIL")
                & (~gd["aborted"])
            ]

            ok_margin = gd[
                gd["verdict"] == "PASS"
            ]

            hard = (
                int(
                    ok_hard["neurons"].max()
                )
                if len(ok_hard)
                else 0
            )

            marg = (
                int(
                    ok_margin["neurons"].max()
                )
                if len(ok_margin)
                else 0
            )

            print(
                f"    dt={dt:g} ms: "
                f"sin_fallos={hard:>3}  "
                f"margen_ok={marg:>3}"
            )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    ap.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Logs del test periódico",
    )

    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("figuras"),
    )

    ap.add_argument(
        "--format",
        default="png",
        choices=[
            "png",
            "pdf",
            "svg",
        ],
    )

    ap.add_argument(
        "--safety-margin",
        type=float,
        default=0.5,
        help=(
            "SAFETY_MARGIN del firmware, "
            "solo para la línea de referencia "
            "(default: 0.5)"
        ),
    )

    ap.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help=(
            "Ruta opcional para exportar "
            "los datos combinados"
        ),
    )

    args = ap.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    for log_path in args.logs:

        if not log_path.exists():
            sys.exit(
                f"No existe el fichero: "
                f"{log_path}"
            )

        df = parse_log(log_path)

        if df.empty:
            sys.exit(
                "No se han encontrado filas "
                "(15 columnas) en "
                f"{log_path}"
            )

        r = df.iloc[0]

        print(
            f"[{log_path.name}] "
            f"plataforma={r['platform']}, "
            f"cpu={r['cpu']}, "
            f"{r['core']}, "
            f"integrador={r['integrator']}, "
            f"precisión={r['precision']} "
            f"({len(df)} filas, "
            f"modelos: "
            f"{', '.join(df['model'].unique())})"
        )

        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data = add_config_labels(
        data
    )

    if args.csv_out:
        data.to_csv(
            args.csv_out,
            index=False,
        )

        print(
            "Datos combinados exportados a "
            f"{args.csv_out}"
        )

    outs = [
        plot_verdict_heatmap(
            data,
            args.outdir,
            args.format,
        ),
        plot_worst_ratio(
            data,
            args.outdir,
            args.format,
            args.safety_margin,
        ),
        plot_jitter(
            data,
            args.outdir,
            args.format,
        ),
        plot_miss_rate(
            data,
            args.outdir,
            args.format,
        ),
    ]

    print_summary(data)

    print(
        "\nFiguras generadas:"
    )

    for f in outs:
        print(
            f"  - {f}"
        )


if __name__ == "__main__":
    main()