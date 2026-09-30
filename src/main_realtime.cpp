#include <Arduino.h>

#include <string>
#include <DifferentialNeuronWrapper.h>
#include <HindmarshRoseModel.h>
#include <HodgkinHuxleyModel.h>
#include <IzhikevichSystemWrapper.h>
#include <IntegratedSystemWrapper.h>

#include <IzhikevichModel.h>
#include <RungeKutta4.h>
#include <RungeKutta6.h>
#include <Euler.h>
#include <SystemWrapper.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

// ============================================================
// Integrator (descomenta para probar RK4; recompila y compara)
// ============================================================

// #define USE_RK4
#define USE_RK6

#ifdef USE_RK6
using Integrator = RungeKutta6;
constexpr const char* INTEGRATOR_NAME = "RK6";
#elifdef USE_RK4
using Integrator = RungeKutta4;
constexpr const char* INTEGRATOR_NAME = "RK4";
#else
using Integrator = Euler;
constexpr const char* INTEGRATOR_NAME = "Euler";
#endif

// ============================================================
// Tipos de neurona
// ============================================================

typedef DifferentialNeuronWrapper<
    SystemWrapper<HodgkinHuxleyModel<float>>, Integrator> HH;

typedef DifferentialNeuronWrapper<
    SystemWrapper<HindmarshRoseModel<float>>, Integrator> HR;

typedef IntegratedSystemWrapper<
        IzhikevichSystemWrapper<float>, 
        Integrator
    > IZH;

// ============================================================
// Configuración del benchmark
// ============================================================

// Pasos de simulación en MILISEGUNDOS de tiempo simulado.
// Se asume 1 ms simulado = 1 ms de reloj real (requisito de tiempo real).
constexpr float DT_VALUES_MS[] = {0.1f, 0.05f, 0.025f, 0.01f};
constexpr std::size_t NUM_DT = sizeof(DT_VALUES_MS) / sizeof(DT_VALUES_MS[0]);

constexpr std::size_t NETWORK_SIZES[] = {1, 2, 4, 8, 16};
constexpr std::size_t NUM_NETWORK_SIZES =
    sizeof(NETWORK_SIZES) / sizeof(NETWORK_SIZES[0]);
constexpr std::size_t MAX_NEURONS = 16;  // >= mayor valor de NETWORK_SIZES

constexpr std::size_t WARMUP_STEPS = 100;  // ticks de calentamiento (caché de flash)
constexpr std::size_t SPIKE_STEPS  = 2000; // ticks (sin cronometrar) para contar spikes
constexpr std::size_t NUM_STEPS    = 1000; // ticks cronometrados

// Fracción máxima del periodo dt que puede ocupar el cómputo del TICK COMPLETO
// (todas las neuronas) en el PEOR caso observado.
constexpr double SAFETY_MARGIN = 0.5;

// Sumidero para evitar que el compilador elimine el cálculo.
volatile float g_sink = 0.0f;

// ============================================================
// Resultado
// ============================================================

struct BenchmarkResult {
    bool valid;
    float dt_ms;
    std::size_t neurons;

    double mean_tick_us;
    double max_tick_us;
    double min_tick_us;
    double mean_step_us;        // por neurona (informativo)
    double neuron_steps_per_s;  // neuronas-paso por segundo

    double ratio_mean;          // mean_tick / dt
    double ratio_worst;         // max_tick / dt
    std::size_t est_max_neurons;// neuronas que caben con el coste medio y el margen

    uint32_t spikes;            // spikes totales en la fase de verificación
    bool realtime;
};

// ============================================================
// Traits por modelo: parámetros, inicialización, lectura, spike
// ============================================================

struct HHTraits {
    using Neuron = HH;
    static constexpr const char* name = "HH";
    static constexpr float input = 0.5f;

    static Neuron::ConstructorArgs args() {
        Neuron::ConstructorArgs a;
        a.params[HH::cm]  = 1.0 * 7.854e-3;
        a.params[HH::vna] = 50.0;
        a.params[HH::vk]  = -77.0;
        a.params[HH::vl]  = -54.387;
        a.params[HH::gna] = 120.0 * 7.854e-3;
        a.params[HH::gk]  = 36.0 * 7.854e-3;
        a.params[HH::gl]  = 0.3 * 7.854e-3;
        return a;
    }
    static void init(Neuron& n) {
        n.set(HH::v, -80.0f);
        n.set(HH::m, 0.1f);
        n.set(HH::n, 0.7f);
        n.set(HH::h, 0.01f);
    }
    static float read(Neuron& n) { return n.get(HH::v); }
    // Spike: cruce ascendente de 0 mV
    static bool spiked(float prev, float cur) { return prev < 0.0f && cur >= 0.0f; }
};

struct HRTraits {
    using Neuron = HR;
    static constexpr const char* name = "HR";
    static constexpr float input = 2.5f;

    static Neuron::ConstructorArgs args() {
        Neuron::ConstructorArgs a;
        a.params[HR::e]  = 0.0;
        a.params[HR::mu] = 0.006;
        a.params[HR::S]  = 4.0;
        a.params[HR::a]  = 1.0;
        a.params[HR::b]  = 3.0;
        a.params[HR::c]  = 1.0;
        a.params[HR::d]  = 5.0;
        a.params[HR::xr] = -1.6;
        a.params[HR::vh] = 1.0;
        return a;
    }
    static void init(Neuron& n) {
        n.set(HR::x, -0.712841f);
        n.set(HR::y, -1.93688f);
        n.set(HR::z, 3.16568f);
    }
    static float read(Neuron& n) { return n.get(HR::x); }
    // Spike: cruce ascendente de x = 0 (adimensional)
    static bool spiked(float prev, float cur) { return prev < 0.0f && cur >= 0.0f; }
};

// Izhikevich RS (Izhikevich 2003): a=0.02 b=0.2 c=-65 d=8, umbral 30 mV
struct IZHTraits {
    using Neuron = IZH;
    static constexpr const char* name = "IZH";
    static constexpr float input = 10.0f;

    static Neuron::ConstructorArgs args() {
        Neuron::ConstructorArgs a;
        a.params[IZH::a]         = 0.02;
        a.params[IZH::b]         = 0.20;
        a.params[IZH::c]         = -65.0;
        a.params[IZH::d]         = 8.0;
        a.params[IZH::threshold] = 30.0;
        return a;
    }
    static void init(Neuron& n) {
        n.set(IZH::v, -65.0f);
        n.set(IZH::u, 0.2f * -65.0f);
    }
    static float read(Neuron& n) { return n.get(IZH::v); }
    // Spike: v se resetea a c, así que se detecta como una caída brusca
    static bool spiked(float prev, float cur) { return (prev - cur) > 30.0f; }
};

// ============================================================
// Benchmark genérico
// ============================================================

template <typename T>
BenchmarkResult runBenchmark(std::size_t neuron_count, float dt_ms) {
    using Neuron = typename T::Neuron;

    BenchmarkResult r{};
    r.valid = false;

    if (neuron_count == 0 || neuron_count > MAX_NEURONS) {
        Serial.println("ERROR: numero de neuronas fuera de rango.");
        return r;
    }

    void* memory = std::malloc(neuron_count * sizeof(Neuron));
    if (memory == nullptr) {
        Serial.print("ERROR: fallo de reserva de memoria para ");
        Serial.println(T::name);
        return r;
    }

    Neuron* neurons = static_cast<Neuron*>(memory);
    typename Neuron::ConstructorArgs args = T::args();

    for (std::size_t i = 0; i < neuron_count; ++i) {
        new (&neurons[i]) Neuron(args);
        T::init(neurons[i]);
    }

    // Un "tick" = todas las neuronas avanzan dt (ejecución secuencial en 1 CPU).
    auto tick = [&]() {
        for (std::size_t i = 0; i < neuron_count; ++i) {
            neurons[i].add_synaptic_input(T::input);
            neurons[i].step(dt_ms);
        }
    };

    // --- Warm-up -------------------------------------------------
    for (std::size_t s = 0; s < WARMUP_STEPS; ++s) tick();

    // --- Verificación de actividad (fuera del cronometraje) -------
    float prev[MAX_NEURONS];
    for (std::size_t i = 0; i < neuron_count; ++i) prev[i] = T::read(neurons[i]);

    uint32_t spikes = 0;
    for (std::size_t s = 0; s < SPIKE_STEPS; ++s) {
        tick();
        for (std::size_t i = 0; i < neuron_count; ++i) {
            const float cur = T::read(neurons[i]);
            if (T::spiked(prev[i], cur)) ++spikes;
            prev[i] = cur;
        }
    }

    // --- Sección cronometrada (contador de ciclos de la CPU) ------
    const double cpu_mhz = static_cast<double>(ESP.getCpuFreqMHz());

    uint32_t min_cyc = UINT32_MAX;
    uint32_t max_cyc = 0;
    uint64_t total_cyc = 0;
    float acc = 0.0f;  // float local: sin coste de double emulado ni volatile

    for (std::size_t s = 0; s < NUM_STEPS; ++s) {
        const uint32_t c0 = ESP.getCycleCount();

        for (std::size_t i = 0; i < neuron_count; ++i) {
            neurons[i].add_synaptic_input(T::input);
            neurons[i].step(dt_ms);
            acc += T::read(neurons[i]);
        }

        const uint32_t cyc = ESP.getCycleCount() - c0;  // seguro ante wrap de 32 bits

        if (cyc < min_cyc) min_cyc = cyc;
        if (cyc > max_cyc) max_cyc = cyc;
        total_cyc += cyc;
    }

    g_sink = acc;  // fuera del cronometraje

    // --- Limpieza -------------------------------------------------
    for (std::size_t i = 0; i < neuron_count; ++i) neurons[i].~Neuron();
    std::free(memory);

    // --- Resultados -----------------------------------------------
    const double dt_us = static_cast<double>(dt_ms) * 1000.0;

    r.valid = true;
    r.dt_ms = dt_ms;
    r.neurons = neuron_count;

    r.mean_tick_us = (static_cast<double>(total_cyc) / NUM_STEPS) / cpu_mhz;
    r.max_tick_us  = static_cast<double>(max_cyc) / cpu_mhz;
    r.min_tick_us  = static_cast<double>(min_cyc) / cpu_mhz;
    r.mean_step_us = r.mean_tick_us / static_cast<double>(neuron_count);
    r.neuron_steps_per_s = 1e6 / r.mean_step_us;

    r.ratio_mean  = r.mean_tick_us / dt_us;
    r.ratio_worst = r.max_tick_us / dt_us;
    r.est_max_neurons =
        static_cast<std::size_t>((dt_us * SAFETY_MARGIN) / r.mean_step_us);

    r.spikes = spikes;

    // Criterio: el PEOR tick de la red completa debe caber en el margen de dt.
    r.realtime = r.max_tick_us < dt_us * SAFETY_MARGIN;

    return r;
}

// ============================================================
// Salida
// ============================================================

void printHeader() {
    Serial.println(
        "model,integrator,neurons,dt_ms,"
        "mean_tick_us,max_tick_us,min_tick_us,mean_step_us,"
        "neuron_steps_per_s,ratio_mean,ratio_worst,"
        "est_max_neurons,spikes,realtime");
}

void printResult(const char* model, const BenchmarkResult& r) {
    if (!r.valid) return;

    Serial.print(model);                  Serial.print(",");
    Serial.print(INTEGRATOR_NAME);        Serial.print(",");
    Serial.print(r.neurons);              Serial.print(",");
    Serial.print(r.dt_ms, 4);             Serial.print(",");
    Serial.print(r.mean_tick_us, 3);      Serial.print(",");
    Serial.print(r.max_tick_us, 3);       Serial.print(",");
    Serial.print(r.min_tick_us, 3);       Serial.print(",");
    Serial.print(r.mean_step_us, 3);      Serial.print(",");
    Serial.print(r.neuron_steps_per_s, 0);Serial.print(",");
    Serial.print(r.ratio_mean, 4);        Serial.print(",");
    Serial.print(r.ratio_worst, 4);       Serial.print(",");
    Serial.print(r.est_max_neurons);      Serial.print(",");
    Serial.print(r.spikes);               Serial.print(",");
    Serial.println(r.realtime ? "PASS" : "FAIL");
}

template <typename T>
void runModel(const char* title) {
    Serial.println();
    Serial.print("=== ");
    Serial.print(title);
    Serial.println(" ===");

    for (std::size_t d = 0; d < NUM_DT; ++d) {
        for (std::size_t n = 0; n < NUM_NETWORK_SIZES; ++n) {
            printResult(T::name, runBenchmark<T>(NETWORK_SIZES[n], DT_VALUES_MS[d]));
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
    Serial.println("========================================");
    Serial.print("Chip: ");        Serial.println(ESP.getChipModel());
    Serial.print("CPU MHz: ");     Serial.println(ESP.getCpuFreqMHz());
    Serial.print("Core: ");        Serial.println(xPortGetCoreID());
    Serial.print("Integrator: ");  Serial.println(INTEGRATOR_NAME);
    Serial.print("Safety margin: "); Serial.println(SAFETY_MARGIN, 2);
    Serial.println("Criterio: max_tick_us < dt_us * margin (red completa, peor caso)");
    Serial.println();

    printHeader();

    runModel<HHTraits>("HODGKIN-HUXLEY");
    runModel<HRTraits>("HINDMARSH-ROSE");
    runModel<IZHTraits>("IZHIKEVICH RS");

    Serial.println();
    Serial.println("========================================");
    Serial.println("BENCHMARK COMPLETE");
    Serial.println("========================================");
}

void loop() {
    // Este benchmark mide solo coste de cómputo. Un test de plazos
    // periódicos (esp_timer / tarea FreeRTOS) sería un benchmark aparte.
}
