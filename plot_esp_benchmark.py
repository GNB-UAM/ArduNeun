import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

LOG_DIR = Path("main_benchmark_logs")
OUTPUT_DIR = Path("plots")

OUTPUT_DIR.mkdir(exist_ok=True)




# ============================================================
# LOG PARSER
# ============================================================

def parse_log(filepath):
    """
    Parse one NEUN benchmark log.

    Returns:
        dict containing:
            - name
            - parameters
            - neurons
            - elapsed_s
            - neuron_steps_per_second
            - us_per_neuron_step
            - heap_consumed
            - raw_neuron_memory
            - sizeof_neuron
            - integration_step
            - cpu_frequency
    """

    text = filepath.read_text(errors="replace")

    # --------------------------------------------------------
    # Global information
    # --------------------------------------------------------

    def extract(pattern, default=None):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return default

    cpu_frequency = extract(r"CPU frequency:\s*([0-9]+)")
    sizeof_neuron = extract(r"sizeof\(Neuron\):\s*([0-9]+)")
    integration_step = extract(r"Integration step:\s*([0-9.eE+-]+)")

    # --------------------------------------------------------
    # Find each neuron block
    # --------------------------------------------------------

    blocks = re.split(r"(?=Neurons:\s*[0-9]+)", text)

    neurons = []
    elapsed_s = []
    neuron_steps_per_second = []
    us_per_neuron_step = []
    heap_consumed = []
    raw_neuron_memory = []

    for block in blocks:

        neuron_match = re.search(
            r"Neurons:\s*([0-9]+)",
            block
        )

        if not neuron_match:
            continue

        n = int(neuron_match.group(1))

        elapsed_match = re.search(
            r"Elapsed time:\s*([0-9.]+)\s*s",
            block
        )

        throughput_match = re.search(
            r"Neuron-steps / second:\s*([0-9.]+)",
            block
        )

        cost_match = re.search(
            r"us / neuron-step:\s*([0-9.]+)",
            block
        )

        heap_match = re.search(
            r"Heap consumed:\s*([0-9]+)",
            block
        )

        raw_memory_match = re.search(
            r"Raw neuron memory:\s*([0-9]+)",
            block
        )

        # Skip incomplete blocks
        if not all([
            elapsed_match,
            throughput_match,
            cost_match,
            heap_match,
            raw_memory_match,
        ]):
            continue

        neurons.append(n)

        elapsed_s.append(
            float(elapsed_match.group(1))
        )

        neuron_steps_per_second.append(
            float(throughput_match.group(1))
        )

        us_per_neuron_step.append(
            float(cost_match.group(1))
        )

        heap_consumed.append(
            int(heap_match.group(1))
        )

        raw_neuron_memory.append(
            int(raw_memory_match.group(1))
        )

    # --------------------------------------------------------
    # Extract benchmark parameters from filename
    # --------------------------------------------------------

    filename = filepath.stem

    # Example:
    # main_benchmark-float-RK4-0.1
    #
    # -> float
    # -> RK4
    # -> 0.1

    parts = filename.split("-")

    parameters = {
        "numeric_type": "unknown",
        "integrator": "unknown",
        "dt": "unknown",
    }

    if len(parts) >= 4:
        parameters["numeric_type"] = parts[-3]
        parameters["integrator"] = parts[-2]
        parameters["dt"] = parts[-1]

    label = (
        f"{parameters['numeric_type']} | "
        f"{parameters['integrator']} | "
        f"dt={parameters['dt']}"
    )

    return {
        "name": filepath.name,
        "label": label,
        "parameters": parameters,

        "neurons": np.array(neurons),

        "elapsed_s": np.array(elapsed_s),

        "neuron_steps_per_second": np.array(
            neuron_steps_per_second
        ),

        "us_per_neuron_step": np.array(
            us_per_neuron_step
        ),

        "heap_consumed": np.array(
            heap_consumed
        ),

        "raw_neuron_memory": np.array(
            raw_neuron_memory
        ),

        "sizeof_neuron": int(sizeof_neuron)
        if sizeof_neuron is not None else None,

        "integration_step": float(integration_step)
        if integration_step is not None else None,

        "cpu_frequency": int(cpu_frequency)
        if cpu_frequency is not None else None,
    }


# ============================================================
# LOAD ALL LOGS
# ============================================================

log_files = sorted(LOG_DIR.glob("*.log"))

if not log_files:
    raise RuntimeError(
        f"No .log files found in {LOG_DIR.resolve()}"
    )

benchmarks = []

for logfile in log_files:

    benchmark = parse_log(logfile)

    benchmarks.append(benchmark)

    print(
        f"Loaded: {logfile.name} "
        f"({len(benchmark['neurons'])} points)"
    )


print()
print(f"Found {len(benchmarks)} benchmark(s).")


# ============================================================
# COMMON PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.figsize": (9, 4),
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ============================================================
# PLOT STYLE — COLORBLIND FRIENDLY
# ============================================================

# Based on the Okabe-Ito color palette.
#
# We use:
#
#   float  -> blue family
#   double -> orange family
#
# dt determines the shade.
#
# Integrator determines:
#
#   RK4   -> solid line + circle
#   Euler -> dashed line + square
#
# This avoids relying on red/green differences.


# ------------------------------------------------------------
# FLOAT — BLUE FAMILY
# ------------------------------------------------------------
FLOAT_COLORS = [
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#56B4E9",  # sky blue
    "#4B0082",  # indigo
    "#008B8B",  # dark cyan
    "#6A3D9A",  # purple
]

# ------------------------------------------------------------
# DOUBLE — ORANGE FAMILY
# ------------------------------------------------------------

DOUBLE_COLORS = [
    "#E69F00",  # orange
    "#D55E00",  # vermillion
    "#CC79A7",  # purple-pink
    "#A65628",  # brown
    "#F0E442",  # yellow
    "#E7298A",  # magenta
]


# ------------------------------------------------------------
# UNKNOWN / OTHER
# ------------------------------------------------------------

OTHER_COLORS = [
    "#000000",
    "#555555",
    "#888888",
    "#AAAAAA",
]


# ------------------------------------------------------------
# LINE STYLE
# ------------------------------------------------------------

LINE_STYLES = {
    "RK4": "-",
    "RK6": "--",
    # "Euler": "--",
}


# ------------------------------------------------------------
# MARKERS
# ------------------------------------------------------------

MARKERS = {
    "RK4": "o",
    "RK6": "s",
    # "Euler": "s",
}


# ============================================================
# STYLE FUNCTION
# ============================================================

def get_style(benchmark, dt_values):

    numeric_type = (
        benchmark["parameters"]["numeric_type"]
    )

    integrator = (
        benchmark["parameters"]["integrator"]
    )

    dt = benchmark["parameters"]["dt"]


    # --------------------------------------------------------
    # Select color family
    # --------------------------------------------------------

    if numeric_type.lower() == "float":

        colors = FLOAT_COLORS

    elif numeric_type.lower() == "double":

        colors = DOUBLE_COLORS

    else:

        colors = OTHER_COLORS


    # --------------------------------------------------------
    # Select shade according to dt
    # --------------------------------------------------------

    try:

        dt_value = float(dt)

        sorted_dt = sorted(
            float(x)
            for x in dt_values
        )

        dt_index = sorted_dt.index(dt_value)

        # Largest dt = darkest
        #
        # Smallest dt = lightest

        color_index = (
            len(sorted_dt)
            - 1
            - dt_index
        )

        # Do not exceed available colors

        color_index = min(
            color_index,
            len(colors) - 1
        )

    except (ValueError, TypeError):

        color_index = 0


    color = colors[color_index]


    # --------------------------------------------------------
    # Line / marker
    # --------------------------------------------------------

    linestyle = LINE_STYLES.get(
        integrator,
        "-"
    )

    marker = MARKERS.get(
        integrator,
        "o"
    )


    return {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "linewidth": 2,
        "markersize": 6,
        "alpha": 0.6,
    }

    dt_values = sorted({
        benchmark["parameters"]["dt"]
        for benchmark in benchmarks
    })


# ============================================================
# COLLECT DT VALUES
# ============================================================

dt_values = sorted({
    benchmark["parameters"]["dt"]
    for benchmark in benchmarks
})



# ============================================================
# 1. TOTAL SIMULATION TIME
# ============================================================

plt.figure()
for benchmark in benchmarks:

    style = get_style(
        benchmark,
        dt_values
    )

    plt.plot(
        benchmark["neurons"],
        benchmark["elapsed_s"],
        label=benchmark["label"],
        **style,
    )
    # Ideal linear scaling for THIS benchmark
    neurons = benchmark["neurons"]
    elapsed = benchmark["elapsed_s"]

    if len(elapsed) > 0:
        ideal_linear = elapsed[0] * neurons / neurons[0]

        plt.plot(
            neurons,
            ideal_linear,
            linestyle="--",
            alpha=0.4,
        )

plt.xlabel("Number of neurons")
plt.ylabel("Elapsed time (s)")
plt.title("NEUN ESP32-S3 — Simulation time for 1000 steps")

plt.xscale("log")
plt.yscale("log")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "elapsed_time.pdf",
    format='pdf',
)

plt.show()


# ============================================================
# 2. MICROSECONDS PER NEURON-STEP
# ============================================================

plt.figure()
for benchmark in benchmarks:

    style = get_style(
        benchmark,
        dt_values
    )

    plt.plot(
        benchmark["neurons"],
        benchmark["us_per_neuron_step"],
        label=benchmark["label"],
        **style,
    )

plt.xlabel("Number of neurons")
plt.ylabel("Time per neuron-step (µs)")
plt.title(
    "NEUN ESP32-S3 — Cost per neuron-step"
)

plt.xscale("log")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "us_per_neuron_step.pdf",
    format='pdf',
)

plt.show()


# ============================================================
# 3. NEURON-STEPS PER SECOND
# ============================================================

plt.figure()

for benchmark in benchmarks:

    style = get_style(
        benchmark,
        dt_values
    )

    plt.plot(
        benchmark["neurons"],
        benchmark["neuron_steps_per_second"],
        label=benchmark["label"],
        **style,
    )


plt.xlabel("Number of neurons")
plt.ylabel("Neuron-steps / second")
plt.title(
    "NEUN ESP32-S3 — Throughput"
)

plt.xscale("log")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "throughput.pdf",
    format='pdf',
)

plt.show()


# ============================================================
# 4. MEMORY
# ============================================================

plt.figure(figsize=(9, 4))
for benchmark in benchmarks:

    style = get_style(
        benchmark,
        dt_values
    )

    plt.plot(
        benchmark["neurons"],
        benchmark["raw_neuron_memory"] / 1024,
        label=benchmark["label"],
        **style,
    )

# sizeof(Neuron) theoretical reference
sizeof_values = [
    b["sizeof_neuron"]
    for b in benchmarks
    if b["sizeof_neuron"] is not None
]

if sizeof_values:

    # Use the first value. They should normally all be equal.
    sizeof_neuron = sizeof_values[0]

    # Use the largest neuron count available
    max_neurons = max(
        np.max(b["neurons"])
        for b in benchmarks
    )

    reference_neurons = np.array([
        1,
        max_neurons
    ])

    plt.plot(
        reference_neurons,
        reference_neurons * sizeof_neuron / 1024,
        linestyle="--",
        linewidth=2,
        label=f"sizeof(Neuron) × N ({sizeof_neuron} B)",
    )


plt.xlabel("Number of neurons")
plt.ylabel("Memory (KiB)")
plt.title(
    "NEUN ESP32-S3 — Neuron memory"
)

plt.xscale("log")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "memory.pdf",
    format='pdf',
)

plt.show()


# ============================================================
# 5. SUMMARY
# ============================================================

print()
print("=" * 80)
print("NEUN ESP32-S3 BENCHMARK SUMMARY")
print("=" * 80)

for benchmark in benchmarks:

    print()
    print("-" * 80)
    print(benchmark["name"])
    print("-" * 80)

    print(
        f"CPU frequency:     "
        f"{benchmark['cpu_frequency']} MHz"
    )

    print(
        f"Neuron size:       "
        f"{benchmark['sizeof_neuron']} bytes"
    )

    print(
        f"Integration step:  "
        f"{benchmark['integration_step']}"
    )

    print(
        f"Best throughput:   "
        f"{benchmark['neuron_steps_per_second'].max():.2f} neuron-steps/s"
    )

    print(
        f"Best cost:         "
        f"{benchmark['us_per_neuron_step'].min():.4f} µs/neuron-step"
    )

    print()

    for n, t, throughput, cost in zip(
        benchmark["neurons"],
        benchmark["elapsed_s"],
        benchmark["neuron_steps_per_second"],
        benchmark["us_per_neuron_step"],
    ):

        print(
            f"{n:5d} neurons | "
            f"{t:12.6f} s | "
            f"{throughput:10.2f} steps/s | "
            f"{cost:10.4f} µs/step"
        )


print()
print("=" * 80)
print("Plots saved to:")
print(OUTPUT_DIR.resolve())
print("=" * 80)
