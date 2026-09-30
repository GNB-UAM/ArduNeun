#include <Arduino.h>

// Necesarios ANTES de las cabeceras de neun (HindmarshRoseModel.h no los incluye).
#include <string>
#include <vector>

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
// TEST PERIÓDICO DE TIEMPO REAL (plazos y jitter)
//
// Una tarea FreeRTOS de máxima prioridad, fijada a un núcleo, ejecuta un
// "tick" (toda la red avanza dt) cada dt de tiempo real. Como dt < 1 ms
// (el tick de FreeRTOS es 1 ms), la temporización se hace por espera activa
// contra el contador de ciclos de la CPU.
//
// Para cada tick k:
//   release_k  = t0 + k * periodo          (instante teórico de liberación)
//   jitter     = inicio_real - release_k   (incluye atraso acumulado)
//   respuesta  = fin_real   - release_k
//   plazo      = release_k + periodo       (el tick debe acabar antes del siguiente)
//   fallo      = respuesta > periodo
// ============================================================

#define USE_RK4
// #define USE_RK6

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

typedef DifferentialNeuronWrapper<
    SystemWrapper<HodgkinHuxleyModel<float>>, Integrator> HH;
typedef DifferentialNeuronWrapper<
    SystemWrapper<HindmarshRoseModel<float>>, Integrator> HR;

typedef IntegratedSystemWrapper<
        IzhikevichSystemWrapper<float>, 
        Integrator
    > IZH;

// ============================================================
// Configuración
// ============================================================

constexpr float DT_VALUES_MS[] = {0.1f, 0.05f, 0.025f, 0.01f};
constexpr std::size_t NUM_DT = sizeof(DT_VALUES_MS) / sizeof(DT_VALUES_MS[0]);

constexpr std::size_t NETWORK_SIZES[] = {1, 2, 4, 8, 16};
constexpr std::size_t NUM_NETWORK_SIZES =
    sizeof(NETWORK_SIZES) / sizeof(NETWORK_SIZES[0]);
constexpr std::size_t MAX_NEURONS = 16;

// Duración de cada prueba en tiempo real (ms). Mantenla por debajo del
// timeout del task watchdog (5 s por defecto): la tarea RT ocupa su núcleo.
constexpr uint32_t TEST_DURATION_MS = 2000;

constexpr std::size_t WARMUP_STEPS = 100;

// Un tick que ocupe más de SAFETY_MARGIN * periodo cuenta como violación de margen.
constexpr double SAFETY_MARGIN = 0.5;

// Se aborta la prueba si el atraso acumulado supera este nº de periodos
// (la simulación se está quedando irremediablemente atrás).
constexpr uint32_t MAX_LAG_PERIODS = 50;

// Núcleo y prioridad de la tarea de tiempo real.
// loopTask (Arduino) corre en el núcleo 1 y queda bloqueada durante la prueba.
#ifndef RT_CORE
#define RT_CORE 1
#endif
constexpr uint32_t RT_STACK_BYTES = 16384;
constexpr UBaseType_t RT_PRIORITY = configMAX_PRIORITIES - 1;

volatile float g_sink = 0.0f;

// ============================================================
// Traits por modelo
// ============================================================

struct HHTraits {
    using Neuron = HH;
    static const char* name() { return "HH"; }
    static float input() { return 0.5f; }

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
};

struct HRTraits {
    using Neuron = HR;
    static const char* name() { return "HR"; }
    static float input() { return 2.5f; }

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
};

struct IZHTraits {
    using Neuron = IZH;
    static const char* name() { return "IZH"; }
    static float input() { return 10.0f; }

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
};

// ============================================================
// Estructuras de trabajo y resultado
// ============================================================

struct RtResult {
    bool valid;
    bool aborted;
    uint32_t ticks_planned;
    uint32_t ticks_done;

    double mean_resp_us;
    double max_resp_us;
    double mean_jitter_us;
    double max_jitter_us;

    uint32_t margin_viol;  // respuesta > SAFETY_MARGIN * periodo
    uint32_t misses;       // respuesta > periodo (plazo incumplido)
};

struct RtJob {
    std::size_t neurons;
    float dt_ms;
    RtResult* out;
    TaskHandle_t notify;
};

// ============================================================
// Tarea de tiempo real (una instancia por prueba)
// ============================================================

template <typename T>
static void rtTask(void* arg) {
    using Neuron = typename T::Neuron;

    RtJob* job = static_cast<RtJob*>(arg);
    RtResult res{};
    const std::size_t n = job->neurons;
    const float dt_ms = job->dt_ms;

    void* mem = std::malloc(n * sizeof(Neuron));
    if (mem == nullptr) {
        *job->out = res;  // valid = false
        xTaskNotifyGive(job->notify);
        vTaskDelete(nullptr);
        return;
    }

    Neuron* neurons = static_cast<Neuron*>(mem);
    typename Neuron::ConstructorArgs args = T::args(); 
    for (std::size_t i = 0; i < n; ++i) {
        new (&neurons[i]) Neuron(args);
        T::init(neurons[i]);
    }

    const float input = T::input();
    auto tick = [&]() {
        for (std::size_t i = 0; i < n; ++i) {
            neurons[i].add_synaptic_input(input);
            neurons[i].step(dt_ms);
        }
    };

    // Warm-up (caché de instrucciones desde flash), sin cronometrar.
    for (std::size_t s = 0; s < WARMUP_STEPS; ++s) tick();

    const uint32_t mhz = ESP.getCpuFreqMHz();
    const uint32_t period_cyc =
        static_cast<uint32_t>(static_cast<double>(dt_ms) * 1000.0 * mhz + 0.5);
    const uint32_t margin_cyc =
        static_cast<uint32_t>(static_cast<double>(period_cyc) * SAFETY_MARGIN);
    const uint32_t lag_limit_cyc = period_cyc * MAX_LAG_PERIODS;
    const uint32_t total_ticks = static_cast<uint32_t>(TEST_DURATION_MS / dt_ms);

    uint64_t sum_resp = 0, sum_jitter = 0;
    uint32_t max_resp = 0, max_jitter = 0;
    uint32_t margin_viol = 0, misses = 0, done = 0;
    bool aborted = false;
    float acc = 0.0f;

    // Primera liberación 1 ms en el futuro para que todo se asiente.
    uint32_t release = ESP.getCycleCount() + mhz * 1000u;

    for (uint32_t k = 0; k < total_ticks; ++k) {
        // Espera activa hasta la liberación (comparación segura ante wrap de 32 bits).
        while (static_cast<int32_t>(ESP.getCycleCount() - release) < 0) {
        }

        const uint32_t start = ESP.getCycleCount();

        for (std::size_t i = 0; i < n; ++i) {
            neurons[i].add_synaptic_input(input);
            neurons[i].step(dt_ms);
            acc += T::read(neurons[i]);
        }

        const uint32_t end = ESP.getCycleCount();

        const uint32_t jitter = start - release;
        const uint32_t resp = end - release;

        sum_jitter += jitter;
        sum_resp += resp;
        if (jitter > max_jitter) max_jitter = jitter;
        if (resp > max_resp) max_resp = resp;
        if (resp > margin_cyc) ++margin_viol;
        if (resp > period_cyc) ++misses;

        ++done;
        release += period_cyc;  // calendario absoluto: sin deriva

        // Si vamos demasiado atrasados, abortamos: la red no cabe.
        if (static_cast<int32_t>(end - release) > static_cast<int32_t>(lag_limit_cyc)) {
            aborted = true;
            break;
        }
    }

    g_sink = acc;

    for (std::size_t i = 0; i < n; ++i) neurons[i].~Neuron();
    std::free(mem);

    res.valid = true;
    res.aborted = aborted;
    res.ticks_planned = total_ticks;
    res.ticks_done = done;
    if (done > 0) {
        res.mean_resp_us = (static_cast<double>(sum_resp) / done) / mhz;
        res.mean_jitter_us = (static_cast<double>(sum_jitter) / done) / mhz;
    }
    res.max_resp_us = static_cast<double>(max_resp) / mhz;
    res.max_jitter_us = static_cast<double>(max_jitter) / mhz;
    res.margin_viol = margin_viol;
    res.misses = misses;

    *job->out = res;
    xTaskNotifyGive(job->notify);
    vTaskDelete(nullptr);
}

// ============================================================
// Lanzamiento y salida
// ============================================================

void printHeader() {
    Serial.println(
        "model,integrator,neurons,dt_ms,ticks_done,"
        "mean_resp_us,max_resp_us,mean_jitter_us,max_jitter_us,"
        "util_mean,util_worst,margin_viol,misses,aborted,verdict");
}

void printResult(const char* model, std::size_t n, float dt_ms, const RtResult& r) {
    if (!r.valid) {
        Serial.print("ERROR: fallo de reserva de memoria en ");
        Serial.println(model);
        return;
    }

    const double dt_us = static_cast<double>(dt_ms) * 1000.0;

    // PASS: sin fallos y sin violar el margen. WARN: sin fallos pero fuera de
    // margen. FAIL: algún plazo incumplido o prueba abortada.constexpr double WARN_MISS_FRACTION = 0.1;   // 10 %
    constexpr double WARN_MISS_FRACTION = 0.1;   // 10 %
    // Fracción de ticks con plazo incumplido (misses / ticks_done)
    const double miss_frac = (r.ticks_done > 0)
        ? static_cast<double>(r.misses) / static_cast<double>(r.ticks_done)
        : 1.0;                                   // sin ticks ejecutados: no es viable
    const double miss_pct = miss_frac * 100.0;

    const char* verdict = "PASS";
    if (r.aborted || miss_frac > WARN_MISS_FRACTION) verdict = "FAIL";
    else if (miss_frac > 0.0)                        verdict = "WARN";

    Serial.print(model);                  Serial.print(",");
    Serial.print(INTEGRATOR_NAME);        Serial.print(",");
    Serial.print(n);                      Serial.print(",");
    Serial.print(dt_ms, 4);               Serial.print(",");
    Serial.print(r.ticks_done);           Serial.print(",");
    Serial.print(r.mean_resp_us, 3);      Serial.print(",");
    Serial.print(r.max_resp_us, 3);       Serial.print(",");
    Serial.print(r.mean_jitter_us, 3);    Serial.print(",");
    Serial.print(r.max_jitter_us, 3);     Serial.print(",");
    Serial.print(r.mean_resp_us / dt_us, 4); Serial.print(",");
    Serial.print(r.max_resp_us / dt_us, 4);  Serial.print(",");
    Serial.print(r.margin_viol);          Serial.print(",");
    Serial.print(r.misses);               Serial.print(",");
    Serial.print(r.aborted ? 1 : 0);      Serial.print(",");
    Serial.print(miss_frac);               Serial.print(",");
    Serial.println(verdict);
}

template <typename T>
void runOne(std::size_t n, float dt_ms) {
    RtResult res{};
    RtJob job{n, dt_ms, &res, xTaskGetCurrentTaskHandle()};

    const BaseType_t ok = xTaskCreatePinnedToCore(
        rtTask<T>, "neun_rt", RT_STACK_BYTES, &job, RT_PRIORITY, nullptr, RT_CORE);

    if (ok != pdPASS) {
        Serial.println("ERROR: no se pudo crear la tarea RT.");
        return;
    }

    // Bloqueados (sin gastar CPU) hasta que la tarea RT termine.
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    printResult(T::name(), n, dt_ms, res);
}

template <typename T>
void runModel(const char* title) {
    Serial.println();
    Serial.print("=== ");
    Serial.print(title);
    Serial.println(" ===");

    for (std::size_t d = 0; d < NUM_DT; ++d) {
        for (std::size_t i = 0; i < NUM_NETWORK_SIZES; ++i) {
            if (NETWORK_SIZES[i] > MAX_NEURONS) continue;
            runOne<T>(NETWORK_SIZES[i], DT_VALUES_MS[d]);
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
    Serial.println("NEUN PERIODIC DEADLINE TEST");
    Serial.println("========================================");
    Serial.print("Chip: ");            Serial.println(ESP.getChipModel());
    Serial.print("CPU MHz: ");         Serial.println(ESP.getCpuFreqMHz());
    Serial.print("Integrator: ");      Serial.println(INTEGRATOR_NAME);
    Serial.print("Precision: ");       Serial.println("float");
    Serial.print("RT core: ");         Serial.println(RT_CORE);
    Serial.print("FreeRTOS tick Hz: ");Serial.println(configTICK_RATE_HZ);
    Serial.print("Test duration ms: ");Serial.println(TEST_DURATION_MS);
    Serial.print("Safety margin: ");   Serial.println(SAFETY_MARGIN, 2);
    Serial.println("Deadline: tick k must finish before release of tick k+1");
    Serial.println();

    printHeader();

    runModel<HHTraits>("HODGKIN-HUXLEY");
    runModel<HRTraits>("HINDMARSH-ROSE");
    runModel<IZHTraits>("IZHIKEVICH RS");

    Serial.println();
    Serial.println("========================================");
    Serial.println("TEST COMPLETE");
    Serial.println("========================================");
}

void loop() {
}
