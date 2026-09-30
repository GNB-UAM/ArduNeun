#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

// Biblioteca de neun
#include <DifferentialNeuronWrapper.h>
#include <HindmarshRoseModel.h>
#include <HodgkinHuxleyModel.h>
#include <IzhikevichModel.h>
#include <IzhikevichSystemWrapper.h>
#include "IntegratedSystemWrapper.h"
#include <RungeKutta4.h>
#include <RungeKutta6.h>
#include <Euler.h>
#include <SystemWrapper.h>

// ============================================================
// Elegir integrador
// ============================================================
//
// Cambia esta línea:
//
using Integrator = Euler;
// using Integrator = RungeKutta4;
// using Integrator = RungeKutta6;
//

// using Integrator = RungeKutta4;


// ============================================================
// Tipos de los tres modelos
// ============================================================

using HH = DifferentialNeuronWrapper<
    SystemWrapper<HodgkinHuxleyModel<float>>,
    Integrator
>;

using HR = DifferentialNeuronWrapper<
    SystemWrapper<HindmarshRoseModel<float>>,
    Integrator
>;

using IZH = IntegratedSystemWrapper<
    IzhikevichSystemWrapper<float>,
    Integrator
>;


// ============================================================
// Parámetros HH
// ============================================================

HH::ConstructorArgs hhArgs()
{
    HH::ConstructorArgs a;

    a.params[HH::cm]  = 1.0 * 7.854e-3;
    a.params[HH::vna] = 50.0;
    a.params[HH::vk]  = -77.0;
    a.params[HH::vl]  = -54.387;

    a.params[HH::gna] = 120.0 * 7.854e-3;
    a.params[HH::gk]  = 36.0 * 7.854e-3;
    a.params[HH::gl]  = 0.3 * 7.854e-3;

    return a;
}


// ============================================================
// Parámetros Hindmarsh-Rose
// ============================================================

HR::ConstructorArgs hrArgs()
{
    HR::ConstructorArgs a;

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


// ============================================================
// Parámetros Izhikevich
// ============================================================

IZH::ConstructorArgs izhArgs()
{
    IZH::ConstructorArgs a;

    a.params[IZH::a]         = 0.02;
    a.params[IZH::b]         = 0.20;
    a.params[IZH::c]         = -65.0;
    a.params[IZH::d]         = 8.0;
    a.params[IZH::threshold] = 30.0;

    return a;
}


// ============================================================
// Inicialización HH
// ============================================================

void initHH(HH& n)
{
    n.set(HH::v, -80.0f);
    n.set(HH::m, 0.1f);
    n.set(HH::n, 0.7f);
    n.set(HH::h, 0.01f);
}


// ============================================================
// Inicialización Hindmarsh-Rose
// ============================================================

void initHR(HR& n)
{
    n.set(HR::x, -0.712841f);
    n.set(HR::y, -1.93688f);
    n.set(HR::z, 3.16568f);
}


// ============================================================
// Inicialización Izhikevich
// ============================================================

void initIZH(IZH& n)
{
    n.set(IZH::v, -65.0f);
    n.set(IZH::u, 0.2f * -65.0f);
}


// ============================================================
// Simulación HH
// ============================================================

void simulateHH(
    float dt,
    float duration,
    const std::string& filename)
{
    HH::ConstructorArgs args = hhArgs();
    HH neuron(args);

    initHH(neuron);

    std::ofstream file(filename);

    if (!file)
    {
        std::cerr << "No se pudo abrir " << filename << '\n';
        return;
    }

    file << "time_ms,value\n";

    const int steps =
        static_cast<int>(duration / dt);

    const float input = 0.5f;

    for (int i = 0; i < steps; ++i)
    {
        float t = i * dt;

        neuron.add_synaptic_input(input);
        neuron.step(dt);

        float v = neuron.get(HH::v);

        file << t << "," << v << "\n";
    }

    file.close();

    std::cout
        << "HH   dt=" << dt
        << " ms -> " << filename << '\n';
}


// ============================================================
// Simulación Hindmarsh-Rose
// ============================================================

void simulateHR(
    float dt,
    float duration,
    const std::string& filename)
{
    HR::ConstructorArgs args = hrArgs();
    HR neuron(args);

    initHR(neuron);

    std::ofstream file(filename);

    if (!file)
    {
        std::cerr << "No se pudo abrir " << filename << '\n';
        return;
    }

    file << "time_ms,value\n";

    const int steps =
        static_cast<int>(duration / dt);

    const float input = 2.5f;

    for (int i = 0; i < steps; ++i)
    {
        float t = i * dt;

        neuron.add_synaptic_input(input);
        neuron.step(dt);

        float x = neuron.get(HR::x);

        file << t << "," << x << "\n";
    }

    file.close();

    std::cout
        << "HR   dt=" << dt
        << " ms -> " << filename << '\n';
}


// ============================================================
// Simulación Izhikevich
// ============================================================

void simulateIZH(
    float dt,
    float duration,
    const std::string& filename)
{
    IZH::ConstructorArgs args = izhArgs();
    IZH neuron(args);

    initIZH(neuron);

    std::ofstream file(filename);

    if (!file)
    {
        std::cerr << "No se pudo abrir " << filename << '\n';
        return;
    }

    file << "time_ms,value\n";

    const int steps =
        static_cast<int>(duration / dt);

    const float input = 10.0;

    for (int i = 0; i < steps; ++i)
    {
        float t = i * dt;

        neuron.add_synaptic_input(input);
        neuron.step(dt);

        float v = neuron.get(IZH::v);

        file << t << "," << v << "\n";
    }

    file.close();

    std::cout
        << "IZH  dt=" << dt
        << " ms -> " << filename << '\n';
}


// ============================================================
// MAIN
// ============================================================

int main()
{
    // Diferentes pasos de integración
    const std::vector<float> dtValues =
    {
        0.1f,
        0.05f,
        0.025f,
        0.01f,
        0.001f
    };

    // Duración de cada simulación
    const float duration = 1000.0f; // ms


    std::cout << "====================================\n";
    std::cout << " SIMULACION DE MODELOS NEURONALES\n";
    std::cout << "====================================\n";

    // std::cout << "Integrador: 4\n";
    std::cout << "Duracion:   " << duration << " ms\n\n";


    for (float dt : dtValues)
    {
        std::cout << "\n------------------------------------\n";
        std::cout << "dt = " << dt << " ms\n";
        std::cout << "------------------------------------\n";

        std::string suffix =
            std::to_string(dt);

        // Cambiar el punto decimal para que
        // sea cómodo usarlo en nombres de archivo.
        for (char& c : suffix)
        {
            if (c == '.')
                c = '_';
        }

        simulateHH(
            dt,
            duration,
            "HH_dt_" + suffix + ".csv"
        );

        simulateHR(
            dt,
            duration,
            "HR_dt_" + suffix + ".csv"
        );

        simulateIZH(
            dt,
            duration,
            "IZH_dt_" + suffix + ".csv"
        );
    }


    std::cout << "\n====================================\n";
    std::cout << " SIMULACION TERMINADA\n";
    std::cout << "====================================\n";

    return 0;
}