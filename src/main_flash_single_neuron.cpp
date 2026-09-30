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

Neuron::ConstructorArgs args = makeNeuronArgs();

Neuron neuron(args);
void setup() {
    Serial.begin(115200);

    delay(1000);


    neuron.set(Neuron::v, -80);
    neuron.set(Neuron::m, 0.1);
    neuron.set(Neuron::n, 0.7);
    neuron.set(Neuron::h, 0.01);

}
// Set the integration step
static int plot_counter = 0;

double t = 0;
double step = 0.1;
void loop() {

    neuron.add_synaptic_input(0.1);
    neuron.step(step);

    double v = neuron.get(Neuron::v);

    // if (++plot_counter >= 3) {
    //     plot_counter = 0;
        // Serial.print("Neun HH voltage: ");
        Serial.print(t, 4);
        Serial.print(" ");
        Serial.println(v, 4);
    // }
    t+=step;

    // digitalWrite(LED_BUILTIN, HIGH);
    // delay(50);

    // digitalWrite(LED_BUILTIN, LOW);
    // delay(50);


}