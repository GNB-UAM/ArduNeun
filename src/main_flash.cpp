#include <Arduino.h>

#include <DifferentialNeuronWrapper.h>
#include <HodgkinHuxleyModel.h>
#include <RungeKutta4.h>
#include <SystemWrapper.h>

typedef RungeKutta4 Integrator;

typedef DifferentialNeuronWrapper<
    SystemWrapper<HodgkinHuxleyModel<double>>,
    Integrator
> Neuron;

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

void setup() {
    Serial.begin(115200);

    delay(1000);

    Neuron::ConstructorArgs args = makeNeuronArgs();

    Neuron neuron(args);

    neuron.set(Neuron::v, -80);
    neuron.set(Neuron::m, 0.1);
    neuron.set(Neuron::n, 0.7);
    neuron.set(Neuron::h, 0.01);

    // Force the compiler to instantiate and retain the actual
    // neuron/integrator code.
    neuron.add_synaptic_input(10.0);
    neuron.step(0.001);

    double v = neuron.get(Neuron::v);

    Serial.print("Neun HH voltage: ");
    Serial.println(v, 10);
}

void loop() {
}