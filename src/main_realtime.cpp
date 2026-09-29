#include <Arduino.h>

#include <string>
#include <DifferentialNeuronWrapper.h>
#include <HindmarshRoseModel.h>
#include <HodgkinHuxleyModel.h>
#include <IzhikevichModel.h>
#include <RungeKutta4.h>
#include <SystemWrapper.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

// ============================================================
// Integrator
// ============================================================

using Integrator = RungeKutta4;


// ============================================================
// Neuron types
// ============================================================

typedef DifferentialNeuronWrapper<
    SystemWrapper<HodgkinHuxleyModel<float>>,
    Integrator
> HH;

typedef  DifferentialNeuronWrapper<
    SystemWrapper<HindmarshRoseModel<float>>,
    Integrator
> HR;

typedef DifferentialNeuronWrapper<
    SystemWrapper<IzhikevichModel<float>>,
    Integrator
> IZH;


// ============================================================
// Benchmark configuration
// ============================================================

// Simulation time steps.
// These are simulation-time intervals, NOT execution times.
constexpr double DT_VALUES[] = {
    0.1,    // 0.1 ms  -> 10 kHz
    0.05,   // 0.05 ms -> 20 kHz
    0.025,  // 0.025 ms -> 40 kHz
    0.01    // 0.01 ms -> 100 kHz
};

constexpr std::size_t NUM_DT = sizeof(DT_VALUES) / sizeof(DT_VALUES[0]);

// Population sizes.
constexpr std::size_t NETWORK_SIZES[] = {
    1,
    2,
    4,
    8,
    16
};

constexpr std::size_t NUM_NETWORK_SIZES =
    sizeof(NETWORK_SIZES) / sizeof(NETWORK_SIZES[0]);

// Number of timed simulation steps.
//
// Keep this reasonably large so that micros() resolution has
// little influence on the result.
constexpr std::size_t NUM_STEPS = 1000;

//Maximum fraction of time from dt that can elapse the computing
// realtime safety margin.
constexpr double SAFETY_MARGIN = 0.5;


// ============================================================
// Result structure
// ============================================================

struct BenchmarkResult {
    double dt_ms;

    std::size_t neurons;

    double total_us;

    double mean_step_us;
    double min_step_us;
    double max_step_us;

    double steps_per_second;

    double time_ratio;

    bool realtime;
};


// ============================================================
// Utility
// ============================================================

double dt_to_microseconds(double dt_seconds) {
    return dt_seconds * 1000;
}


// ============================================================
// HH initialization
// ============================================================

HH::ConstructorArgs makeHHArgs() {

    HH::ConstructorArgs args;

    args.params[HH::cm]  = 1.0 * 7.854e-3;
    args.params[HH::vna] = 50.0;
    args.params[HH::vk]  = -77.0;
    args.params[HH::vl]  = -54.387;

    args.params[HH::gna] = 120.0 * 7.854e-3;
    args.params[HH::gk]  = 36.0 * 7.854e-3;
    args.params[HH::gl]  = 0.3 * 7.854e-3;

    return args;
}


void initializeHH(HH& neuron) {

    neuron.set(HH::v, -80.0);
    neuron.set(HH::m, 0.1);
    neuron.set(HH::n, 0.7);
    neuron.set(HH::h, 0.01);
}


// ============================================================
// Hindmarsh-Rose initialization
// ============================================================

HR::ConstructorArgs makeHRArgs() {

    HR::ConstructorArgs args;

    args.params[HR::e]  = 0.0;
    args.params[HR::mu] = 0.006;
    args.params[HR::S]  = 4.0;
    args.params[HR::a]  = 1.0;
    args.params[HR::b]  = 3.0;
    args.params[HR::c]  = 1.0;
    args.params[HR::d]  = 5.0;
    args.params[HR::xr] = -1.6;
    args.params[HR::vh] = 1.0;

    return args;
}


void initializeHR(HR& neuron) {

    neuron.set(HR::x, -0.712841);
    neuron.set(HR::y, -1.93688);
    neuron.set(HR::z, 3.16568);
}


// ============================================================
// Izhikevich initialization
//
// Regular Spiking (RS) parameters from Izhikevich (2003):
//
// a = 0.02
// b = 0.20
// c = -65
// d = 8
//
// The original model uses a 30 mV spike threshold.
// ============================================================

IZH::ConstructorArgs makeIZHArgs() {

    IZH::ConstructorArgs args;

    args.params[IZH::a]         = 0.02;
    args.params[IZH::b]         = 0.20;
    args.params[IZH::c]         = -65.0;
    args.params[IZH::d]         = 8.0;
    args.params[IZH::threshold] = 30.0;

    return args;
}


void initializeIZH(IZH& neuron) {

    neuron.set(IZH::v, -65.0);
    neuron.set(IZH::u, 0.2 * (-65.0));
}


// ============================================================
// Benchmark HH
// ============================================================

BenchmarkResult benchmarkHH(
    std::size_t neuron_count,
    double dt
) {

    const std::size_t memory_size =
        neuron_count * sizeof(HH);

    void* memory = std::malloc(memory_size);

    if (memory == nullptr) {
        Serial.println("ERROR: HH allocation failed.");
        return {};
    }

    HH* neurons = static_cast<HH*>(memory);

    HH::ConstructorArgs args = makeHHArgs();

    for (std::size_t i = 0; i < neuron_count; ++i) {

        new (&neurons[i]) HH(args);

        initializeHH(neurons[i]);

        // Small constant input to keep the computational path
        // representative of normal use.
        neurons[i].add_synaptic_input(0.5);
    }

    // --------------------------------------------------------
    // Warm-up
    // --------------------------------------------------------

    for (std::size_t i = 0; i < neuron_count; ++i) {
        neurons[i].step(dt);
    }

    // --------------------------------------------------------
    // Timed section
    // --------------------------------------------------------

    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;

    uint64_t total_us = 0;

    volatile double checksum = 0.0;

    for (std::size_t step = 0; step < NUM_STEPS; ++step) {

        const uint32_t start = micros();

        for (std::size_t i = 0; i < neuron_count; ++i) {

            neurons[i].add_synaptic_input(0.5);

            neurons[i].step(dt);

            checksum += neurons[i].get(HH::v);
        }

        const uint32_t elapsed = micros() - start;

        if (elapsed < min_us)
            min_us = elapsed;

        if (elapsed > max_us)
            max_us = elapsed;

        total_us += elapsed;
    }

    (void)checksum;

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    for (std::size_t i = 0; i < neuron_count; ++i) {
        neurons[i].~HH();
    }

    std::free(memory);

    // --------------------------------------------------------
    // Results
    // --------------------------------------------------------

    const double mean_tick_us =
        static_cast<double>(total_us) /
        static_cast<double>(NUM_STEPS);

    const double mean_step_us =
        mean_tick_us /
        static_cast<double>(neuron_count);

    const double min_step_us =
        static_cast<double>(min_us) /
        static_cast<double>(neuron_count);

    const double max_step_us =
        static_cast<double>(max_us) /
        static_cast<double>(neuron_count);

    const double steps_per_second =
        1'000'000.0 / mean_step_us;

    const double dt_us =
        dt_to_microseconds(dt);

    const double ratio =
        mean_step_us / dt_us;

    return {
        //dt * 1000.0,
        dt,
        neuron_count,
        static_cast<double>(total_us),
        mean_step_us,
        min_step_us,
        max_step_us,
        steps_per_second,
        ratio,
        mean_step_us < dt_us * SAFETY_MARGIN
    };
}


// ============================================================
// Benchmark HR
// ============================================================

BenchmarkResult benchmarkHR(
    std::size_t neuron_count,
    double dt
) {

    const std::size_t memory_size =
        neuron_count * sizeof(HR);

    void* memory = std::malloc(memory_size);

    if (memory == nullptr) {
        Serial.println("ERROR: HR allocation failed.");
        return {};
    }

    HR* neurons = static_cast<HR*>(memory);

    HR::ConstructorArgs args = makeHRArgs();

    for (std::size_t i = 0; i < neuron_count; ++i) {

        new (&neurons[i]) HR(args);

        initializeHR(neurons[i]);

        neurons[i].add_synaptic_input(2.5);
    }

    // Warm-up
    for (std::size_t i = 0; i < neuron_count; ++i) {
        neurons[i].step(dt);
    }

    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;

    uint64_t total_us = 0;

    volatile double checksum = 0.0;

    for (std::size_t step = 0; step < NUM_STEPS; ++step) {

        const uint32_t start = micros();

        for (std::size_t i = 0; i < neuron_count; ++i) {

            neurons[i].add_synaptic_input(2.5);

            neurons[i].step(dt);

            checksum += neurons[i].get(HR::x);
        }

        const uint32_t elapsed = micros() - start;

        if (elapsed < min_us)
            min_us = elapsed;

        if (elapsed > max_us)
            max_us = elapsed;

        total_us += elapsed;
    }

    (void)checksum;

    for (std::size_t i = 0; i < neuron_count; ++i) {
        neurons[i].~HR();
    }

    std::free(memory);

    const double mean_tick_us =
        static_cast<double>(total_us) /
        static_cast<double>(NUM_STEPS);

    const double mean_step_us =
        mean_tick_us /
        static_cast<double>(neuron_count);

    const double min_step_us =
        static_cast<double>(min_us) /
        static_cast<double>(neuron_count);

    const double max_step_us =
        static_cast<double>(max_us) /
        static_cast<double>(neuron_count);

    const double steps_per_second =
        1'000'000.0 / mean_step_us;

    const double dt_us =
        dt_to_microseconds(dt);

    const double ratio =
        mean_step_us / dt_us;

    return {
        //dt * 1000.0,
        dt,
        neuron_count,
        static_cast<double>(total_us),
        mean_step_us,
        min_step_us,
        max_step_us,
        steps_per_second,
        ratio,
        mean_step_us < dt_us * SAFETY_MARGIN
    };
}


// ============================================================
// Benchmark Izhikevich
// ============================================================

BenchmarkResult benchmarkIZH(
    std::size_t neuron_count,
    double dt
) {

    const std::size_t memory_size =
        neuron_count * sizeof(IZH);

    void* memory = std::malloc(memory_size);

    if (memory == nullptr) {
        Serial.println("ERROR: IZH allocation failed.");
        return {};
    }

    IZH* neurons = static_cast<IZH*>(memory);

    IZH::ConstructorArgs args = makeIZHArgs();

    for (std::size_t i = 0; i < neuron_count; ++i) {

        new (&neurons[i]) IZH(args);

        initializeIZH(neurons[i]);

        neurons[i].add_synaptic_input(10.0);
    }

    // Warm-up
    for (std::size_t i = 0; i < neuron_count; ++i) {
        neurons[i].step(dt);
    }

    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;

    uint64_t total_us = 0;

    volatile double checksum = 0.0;

    for (std::size_t step = 0; step < NUM_STEPS; ++step) {

        const uint32_t start = micros();

        for (std::size_t i = 0; i < neuron_count; ++i) {

            neurons[i].add_synaptic_input(10.0);

            neurons[i].step(dt);

            checksum += neurons[i].get(IZH::v);
        }

        const uint32_t elapsed = micros() - start;

        if (elapsed < min_us)
            min_us = elapsed;

        if (elapsed > max_us)
            max_us = elapsed;

        total_us += elapsed;
    }

    (void)checksum;

    for (std::size_t i = 0; i < neuron_count; ++i) {
        neurons[i].~IZH();
    }

    std::free(memory);

    const double mean_tick_us =
        static_cast<double>(total_us) /
        static_cast<double>(NUM_STEPS);

    const double mean_step_us =
        mean_tick_us /
        static_cast<double>(neuron_count);

    const double min_step_us =
        static_cast<double>(min_us) /
        static_cast<double>(neuron_count);

    const double max_step_us =
        static_cast<double>(max_us) /
        static_cast<double>(neuron_count);

    const double steps_per_second =
        1'000'000.0 / mean_step_us;

    const double dt_us =
        dt_to_microseconds(dt);

    const double ratio =
        mean_step_us / dt_us;

    return {
        ////dt * 1000.0,
        dt,
        neuron_count,
        static_cast<double>(total_us),
        mean_step_us,
        min_step_us,
        max_step_us,
        steps_per_second,
        ratio,
        mean_step_us < dt_us * SAFETY_MARGIN
    };
}


// ============================================================
// Output
// ============================================================

void printResult(
    const char* model,
    const BenchmarkResult& r
) {

    Serial.print(model);
    Serial.print(",");

    Serial.print(r.neurons);
    Serial.print(",");

    Serial.print(r.dt_ms, 5);
    Serial.print(",");

    Serial.print(r.mean_step_us, 3);
    Serial.print(",");

    Serial.print(r.min_step_us, 3);
    Serial.print(",");

    Serial.print(r.max_step_us, 3);
    Serial.print(",");

    Serial.print(r.steps_per_second, 2);
    Serial.print(",");

    Serial.print(r.time_ratio, 4);
    Serial.print(",");
    
    Serial.print(r.time_ratio * 100.0, 1);  // % de utilización del ciclo
    Serial.print(",");

    Serial.println(r.realtime ? "PASS" : "FAIL");
}


// ============================================================
// Run model
// ============================================================

void runHH() {

    Serial.println();
    Serial.println("=== HODGKIN-HUXLEY ===");

    for (std::size_t d = 0; d < NUM_DT; ++d) {

        const double dt = DT_VALUES[d];

        for (std::size_t n = 0;
             n < NUM_NETWORK_SIZES;
             ++n) {

            const BenchmarkResult result =
                benchmarkHH(
                    NETWORK_SIZES[n],
                    dt
                );

            printResult("HH", result);
        }
    }
}


void runHR() {

    Serial.println();
    Serial.println("=== HINDMARSH-ROSE ===");

    for (std::size_t d = 0; d < NUM_DT; ++d) {

        const double dt = DT_VALUES[d];

        for (std::size_t n = 0;
             n < NUM_NETWORK_SIZES;
             ++n) {

            const BenchmarkResult result =
                benchmarkHR(
                    NETWORK_SIZES[n],
                    dt
                );

            printResult("HR", result);
        }
    }
}


void runIZH() {

    Serial.println();
    Serial.println("=== IZHIKEVICH RS ===");

    for (std::size_t d = 0; d < NUM_DT; ++d) {

        const double dt = DT_VALUES[d];

        for (std::size_t n = 0;
             n < NUM_NETWORK_SIZES;
             ++n) {

            const BenchmarkResult result =
                benchmarkIZH(
                    NETWORK_SIZES[n],
                    dt
                );

            printResult("IZH", result);
        }
    }
}


// ============================================================
// Arduino
// ============================================================

void setup() {

    Serial.begin(115200);

    delay(1000);

    Serial.println();
    Serial.println("========================================");
    Serial.println("NEUN REAL-TIME TEMPORAL RESOLUTION");
    Serial.println("ESP32-S3");
    Serial.println("========================================");

    Serial.println();

    Serial.println(
        "model,neurons,dt_ms,"
        "mean_step_us,min_step_us,max_step_us,"
        "steps_per_second,time_ratio,realtime"
    );

    runHH();

    runHR();

    runIZH();

    Serial.println();
    Serial.println("========================================");
    Serial.println("BENCHMARK COMPLETE");
    Serial.println("========================================");
}


void loop() {
    // No periodic workload here.
    //
    // This benchmark measures computational cost only.
    // A separate scheduler/periodic-task benchmark can be
    // added later if required.
}