#include <Arduino.h>

#ifndef LED_BUILTIN
#define LED_BUILTIN 21
#endif

#include <DifferentialNeuronWrapper.h>
#include <HodgkinHuxleyModel.h>
#include <ChemicalSynapse.h>
#include <RungeKutta4.h>
#include <SystemWrapper.h>


typedef RungeKutta4 Integrator;

typedef DifferentialNeuronWrapper<
    SystemWrapper<HodgkinHuxleyModel<double>>,
    Integrator
> Neuron;

typedef DifferentialNeuronWrapper<SystemWrapper<HodgkinHuxleyModel<double>>, Integrator> HH;
typedef ChemicalSynapse<HH, HH, Integrator, double> Synapse;

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

Synapse::ConstructorArgs makeSynapseArgs() {
    Synapse::ConstructorArgs syn_args;
    syn_args.params[Synapse::gfast] = 0.015;
    syn_args.params[Synapse::Esyn] = -75;
    syn_args.params[Synapse::sfast] = 0.2;
    syn_args.params[Synapse::Vfast] = -50;
    syn_args.params[Synapse::gslow] = 0.025; //When 0, use only fast
    syn_args.params[Synapse::k1] = 1;
    syn_args.params[Synapse::k2] = 0.03;
    syn_args.params[Synapse::sslow] = 1;


    return syn_args;
}

Neuron::ConstructorArgs args = makeNeuronArgs();

// Initialize neuron models
Neuron h1(args), h2(args);

Synapse::ConstructorArgs syn_args = makeSynapseArgs();

// Initialize a synapse between the neurons
Synapse s(h1, Neuron::v, h2, Neuron::v, syn_args, 1);

double t=0;
// Set the integration step
static int plot_counter = 0;
const double step = 0.01;

void setup() {
    Serial.begin(921600);

    pinMode(LED_BUILTIN, OUTPUT);
    delay(10000);


    // Set initial value of V in neuron n1
    h1.set(Neuron::v, -75);
    h2.set(Neuron::v, -75);

    // Force the compiler to instantiate and retain the actual
    // neuron/integrator code.
    h1.add_synaptic_input(10.0);
    h1.step(step);
    h2.step(step);
    s.step(step);

    double v = h1.get(Neuron::v);
    double v2 = h2.get(Neuron::v);
    double i = s.get(Synapse::i);

}

void loop() {
    s.step(step, h1.get(HH::v), h2.get(HH::v));

    h1.add_synaptic_input(0.1);
    h2.add_synaptic_input(0.1);
    h2.add_synaptic_input(s.get(Synapse::i));

    h1.step(step);
    h2.step(step);

    if (++plot_counter >= 3) {
        plot_counter = 0;

        Serial.print(t, 4);
        Serial.print(" ");
        Serial.print(h1.get(HH::v), 6);
        Serial.print(" ");
        Serial.println(h2.get(HH::v), 6);
        // Serial.print(" ");
        // Serial.println(s.get(Synapse::i), 6);
    }
    // digitalWrite(LED_BUILTIN, HIGH);
    // delay(100);

    // digitalWrite(LED_BUILTIN, LOW);
    // delay(100);


    t += step;
}