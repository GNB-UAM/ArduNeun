import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# NEUN ESP32-S3 BENCHMARK
# ============================================================

neurons = np.array([1, 10, 100, 200, 500])

# Results
elapsed_s = np.array([
    0.679814,
    6.796249,
    67.962036,
    135.924138,
    340.753393,
])

neuron_steps_per_second = np.array([
    1470.99,
    1471.40,
    1471.41,
    1471.41,
    1467.34,
])

us_per_neuron_step = np.array([
    679.8140,
    679.6249,
    679.6204,
    679.6207,
    681.5068,
])

heap_consumed = np.array([
    112,
    996,
    0,
    0,
    0,
])

sizeof_neuron = 96

raw_neuron_memory = np.array([
    96,
    960,
    9600,
    19200,
    48000,
])


# ============================================================
# 1. TOTAL SIMULATION TIME
# ============================================================

plt.figure(figsize=(5, 4.5))

plt.plot(
    neurons,
    elapsed_s,
    marker='o',
    linewidth=2,
    label='Measured'
)

# Ideal linear scaling based on the 1-neuron measurement
ideal_linear = elapsed_s[0] * neurons
plt.plot(
    neurons,
    ideal_linear,
    linestyle='--',
    label='Ideal linear scaling'
)

plt.xlabel('Number of neurons')
plt.ylabel('Elapsed time (s)')
plt.title('NEUN ESP32-S3 — Simulation time')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig('esp32s3_elapsed_time.png', dpi=150)
plt.show()


# ============================================================
# 2. MICROSECONDS PER NEURON-STEP
# ============================================================

plt.figure(figsize=(5, 4.5))

plt.plot(
    neurons,
    us_per_neuron_step,
    marker='o',
    linewidth=2
)

plt.xlabel('Number of neurons')
plt.ylabel('Time per neuron-step (µs)')
plt.title('NEUN ESP32-S3 — Cost per neuron-step')
plt.xscale('log')
plt.grid(True, which='both', alpha=0.25)
plt.tight_layout()

plt.savefig('esp32s3_us_per_neuron_step.png', dpi=150)
plt.show()


# ============================================================
# 3. NEURON-STEPS PER SECOND
# ============================================================

plt.figure(figsize=(5, 4.5))

plt.plot(
    neurons,
    neuron_steps_per_second,
    marker='o',
    linewidth=2
)

plt.xlabel('Number of neurons')
plt.ylabel('Neuron-steps / second')
plt.title('NEUN ESP32-S3 — Throughput')
plt.xscale('log')
plt.grid(True, which='both', alpha=0.25)
plt.tight_layout()

plt.savefig('esp32s3_throughput.png', dpi=150)
plt.show()


# ============================================================
# 4. MEMORY
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    neurons,
    raw_neuron_memory / 1024,
    marker='o',
    linewidth=2,
    label='Raw neuron memory'
)

plt.plot(
    neurons,
    neurons * sizeof_neuron / 1024,
    linestyle='--',
    label='sizeof(Neuron) × N'
)

# Only show measured heap values where they are non-zero
valid_heap = heap_consumed > 0

plt.scatter(
    neurons[valid_heap],
    heap_consumed[valid_heap] / 1024,
    s=60,
    label='Measured heap consumption'
)

plt.xlabel('Number of neurons')
plt.ylabel('Memory (KiB)')
plt.title('NEUN ESP32-S3 — Neuron memory')
plt.xscale('log')
plt.grid(True, which='both', alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig('esp32s3_memory.png', dpi=150)
plt.show()


# ============================================================
# SUMMARY
# ============================================================

print()
print('=' * 60)
print('NEUN ESP32-S3 BENCHMARK SUMMARY')
print('=' * 60)

print(f'Neuron size:              {sizeof_neuron} bytes')
print(f'Best measured throughput: {neuron_steps_per_second.max():.2f} neuron-steps/s')
print(f'Best measured cost:       {us_per_neuron_step.min():.4f} µs/neuron-step')
print()

for n, t, throughput, cost in zip(
    neurons,
    elapsed_s,
    neuron_steps_per_second,
    us_per_neuron_step
):
    print(
        f'{n:4d} neurons | '
        f'{t:10.6f} s | '
        f'{throughput:8.2f} steps/s | '
        f'{cost:8.4f} µs/step'
    )