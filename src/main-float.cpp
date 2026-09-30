#include <Arduino.h>

#ifndef LED_BUILTIN
#define LED_BUILTIN 21
#endif

#include <DifferentialNeuronWrapper.h>
#include <HodgkinHuxleyModel.h>
#include <RungeKutta6.h>
#include <SystemWrapper.h>

#include <new>
#include <cstddef>

// -----------------------------------------------------------------------------
// Neuron definition: exactly the same model as in the original code
// -----------------------------------------------------------------------------

typedef RungeKutta6 Integrator;

typedef DifferentialNeuronWrapper<
    SystemWrapper<HodgkinHuxleyModel<float>>,
    Integrator>
    Neuron;


// -----------------------------------------------------------------------------
// Benchmark configuration
// -----------------------------------------------------------------------------

// Number of integration steps for each benchmark.
constexpr std::size_t NUM_STEPS = 1000;

// Integration step, identical to the original example.
float STEP = 0.1;

// Population sizes to benchmark.
constexpr std::size_t NETWORK_SIZES[] = {
    1,
    10,
    100,
    200,
    500,
    1000
};


// -----------------------------------------------------------------------------
// Hodgkin-Huxley parameters
// -----------------------------------------------------------------------------

Neuron::ConstructorArgs makeNeuronArgs() {
    Neuron::ConstructorArgs args;

    args.params[Neuron::cm] = 1 * 7.854e-3;
    args.params[Neuron::vna] = 50;
    args.params[Neuron::vk] = -77;
    args.params[Neuron::vl] = -54.387;
    args.params[Neuron::gna] = 120 * 7.854e-3;
    args.params[Neuron::gk] = 36 * 7.854e-3;
    args.params[Neuron::gl] = 0.3 * 7.854e-3;

    return args;
}


// -----------------------------------------------------------------------------
// Initialize one neuron exactly as in the original example
// -----------------------------------------------------------------------------

void initializeNeuron(Neuron &n) {
    n.set(Neuron::v, -80);
    n.set(Neuron::m, 0.1);
    n.set(Neuron::n, 0.7);
    n.set(Neuron::h, 0.01);
}


// -----------------------------------------------------------------------------
// Benchmark one population
// -----------------------------------------------------------------------------

void benchmark(std::size_t neuron_count) {
    Serial.println();
    Serial.println("========================================");
    Serial.print("Neurons: ");
    Serial.println(neuron_count);
    Serial.println("========================================");

    const std::size_t heap_before = ESP.getFreeHeap();

    Serial.print("Free heap before allocation: ");
    Serial.print(heap_before);
    Serial.println(" bytes");

    // Allocate raw memory on the heap.
    //
    // We use placement new because Neuron is initialized with ConstructorArgs
    // and may not have a default constructor.
    const std::size_t neuron_memory =
        neuron_count * sizeof(Neuron);

    void *memory = std::malloc(neuron_memory);

    if (memory == nullptr) {
        Serial.println("ERROR: allocation failed.");
        return;
    }

    Neuron *neurons = static_cast<Neuron *>(memory);

    // Constructor parameters.
    Neuron::ConstructorArgs args = makeNeuronArgs();

    // Construct and initialize every neuron.
    for (std::size_t i = 0; i < neuron_count; ++i) {
        new (&neurons[i]) Neuron(args);
        initializeNeuron(neurons[i]);
    }

    const std::size_t heap_after = ESP.getFreeHeap();
    const std::size_t heap_used = heap_before - heap_after;

    Serial.print("Free heap after allocation:  ");
    Serial.print(heap_after);
    Serial.println(" bytes");

    Serial.print("Heap consumed:               ");
    Serial.print(heap_used);
    Serial.println(" bytes");

    Serial.print("sizeof(Neuron):              ");
    Serial.print(sizeof(Neuron));
    Serial.println(" bytes");

    Serial.print("Raw neuron memory:           ");
    Serial.print(neuron_memory);
    Serial.println(" bytes");

    // -------------------------------------------------------------------------
    // Simulation
    // -------------------------------------------------------------------------

    volatile double checksum = 0.0;

    const unsigned long start = micros();

    for (std::size_t step = 0; step < NUM_STEPS; ++step) {

        for (std::size_t i = 0; i < neuron_count; ++i) {
            neurons[i].step(STEP);

            // Read voltage so the computation has an observable result.
            checksum += neurons[i].get(Neuron::v);
        }
    }

    const unsigned long elapsed_us = micros() - start;

    // -------------------------------------------------------------------------
    // Results
    // -------------------------------------------------------------------------

    const std::size_t neuron_steps =
        neuron_count * NUM_STEPS;

    const double elapsed_seconds =
        static_cast<double>(elapsed_us) / 1'000'000.0;

    const double neuron_steps_per_second =
        static_cast<double>(neuron_steps) / elapsed_seconds;

    const double microseconds_per_neuron_step =
        static_cast<double>(elapsed_us) /
        static_cast<double>(neuron_steps);

    Serial.println();
    Serial.println("--- Results ---");

    Serial.print("Simulation steps:            ");
    Serial.println(NUM_STEPS);

    Serial.print("Neuron-steps:                 ");
    Serial.println(neuron_steps);

    Serial.print("Elapsed time:                 ");
    Serial.print(elapsed_us);
    Serial.println(" us");

    Serial.print("Elapsed time:                 ");
    Serial.print(elapsed_seconds, 6);
    Serial.println(" s");

    Serial.print("Neuron-steps / second:        ");
    Serial.println(neuron_steps_per_second, 2);

    Serial.print("us / neuron-step:             ");
    Serial.println(microseconds_per_neuron_step, 4);

    Serial.print("Checksum:                     ");
    Serial.println(checksum, 10);

    // -------------------------------------------------------------------------
    // Destruction and cleanup
    // -------------------------------------------------------------------------

    for (std::size_t i = 0; i < neuron_count; ++i) {
        neurons[i].~Neuron();
    }

    std::free(memory);

    const std::size_t heap_after_free = ESP.getFreeHeap();

    Serial.print("Free heap after cleanup:      ");
    Serial.print(heap_after_free);
    Serial.println(" bytes");

    Serial.println();
}


// -----------------------------------------------------------------------------
// Arduino setup
// -----------------------------------------------------------------------------

void setup() {
    Serial.begin(115200);

    pinMode(LED_BUILTIN, OUTPUT);
    // Give USB serial time to initialize.
    delay(10000);

    Serial.println();
    Serial.println("========================================");
    Serial.println("        NEUN ESP32-S3 BENCHMARK");
    Serial.println("========================================");

    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);

    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
    Serial.print("CPU frequency: ");
    Serial.print(getCpuFrequencyMhz());
    Serial.println(" MHz");

    Serial.print("Free heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");

    Serial.print("Free PSRAM: ");
    Serial.print(ESP.getFreePsram());
    Serial.println(" bytes");

    Serial.print("sizeof(Neuron): ");
    Serial.print(sizeof(Neuron));
    Serial.println(" bytes");

    Serial.print("Integration step: ");
    Serial.println(STEP, 6);

    Serial.print("Steps per benchmark: ");
    Serial.println(NUM_STEPS);

    // -------------------------------------------------------------------------
    // Run all benchmark sizes
    // -------------------------------------------------------------------------

    std::size_t NUM_NETWORK_SIZES =
        sizeof(NETWORK_SIZES) / sizeof(NETWORK_SIZES[0]);

    for (std::size_t i = 0; i < NUM_NETWORK_SIZES; ++i) {
        benchmark(NETWORK_SIZES[i]);

        // Small pause between tests.
        delay(500);
    }

    Serial.println();
    Serial.println("========================================");
    Serial.println("          BENCHMARK COMPLETE");
    Serial.println("========================================");
    
    
    // Give USB serial time to clean
    delay(10000);

    STEP = 0.01;

    Serial.println();
    Serial.println("========================================");
    Serial.println("        NEUN ESP32-S3 BENCHMARK");
    Serial.println("========================================");

    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);

    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
    Serial.print("CPU frequency: ");
    Serial.print(getCpuFrequencyMhz());
    Serial.println(" MHz");

    Serial.print("Free heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");

    Serial.print("Free PSRAM: ");
    Serial.print(ESP.getFreePsram());
    Serial.println(" bytes");

    Serial.print("sizeof(Neuron): ");
    Serial.print(sizeof(Neuron));
    Serial.println(" bytes");

    Serial.print("Integration step: ");
    Serial.println(STEP, 6);

    Serial.print("Steps per benchmark: ");
    Serial.println(NUM_STEPS);

    // -------------------------------------------------------------------------
    // Run all benchmark sizes
    // -------------------------------------------------------------------------

    NUM_NETWORK_SIZES =
        sizeof(NETWORK_SIZES) / sizeof(NETWORK_SIZES[0]);

    for (std::size_t i = 0; i < NUM_NETWORK_SIZES; ++i) {
        benchmark(NETWORK_SIZES[i]);

        // Small pause between tests.
        delay(500);
    }

    Serial.println();
    Serial.println("========================================");
    Serial.println("          BENCHMARK COMPLETE");
    Serial.println("========================================");
}


// -----------------------------------------------------------------------------
// Arduino loop
// -----------------------------------------------------------------------------

void loop() {
    // Nothing to do.
}