#include <Arduino.h>

#include <DifferentialNeuronWrapper.h>
#include <HodgkinHuxleyModel.h>
#include <BistableRulkovMapModel.h>
#include <RulkovMapModel.h>
#include <SimpleOscillatorModel.h>
#include <IzhikevichModel.h>
#include <VavoulisCGCModel.h>
#include <FerdoMapModel.h>
#include <MatsuokaModel.h>
#include <VavoulisCGCModelQ10.h>
#include <VavoulisModel.h>
#include <RowatSelverstonModel.h>

#include <ChemicalSynapse.h>
#include <STDPSynapse.h>
#include <DiffusionSynapse.h>
#include <LinskerSynapse.h>

#include <RungeKutta4.h>
#include <SystemWrapper.h>


// ============================================================
// Integrator
// ============================================================

typedef RungeKutta4 Integrator;


// ============================================================
// Define a type neuron for each model
// ============================================================

typedef DifferentialNeuronWrapper<SystemWrapper<HodgkinHuxleyModel<double>>,   Integrator> HH;
typedef DifferentialNeuronWrapper<SystemWrapper<BistableRulkovMapModel<double>>, Integrator> BistableRulkov;
typedef DifferentialNeuronWrapper<SystemWrapper<RulkovMapModel<double>>,       Integrator> Rulkov;
typedef DifferentialNeuronWrapper<SystemWrapper<SimpleOscillatorModel<double>>,Integrator> SimpleOscillator;
typedef DifferentialNeuronWrapper<SystemWrapper<IzhikevichModel<double>>,      Integrator> Izhikevich;
typedef DifferentialNeuronWrapper<SystemWrapper<VavoulisCGCModel<double>>,     Integrator> VavoulisCGC;
typedef DifferentialNeuronWrapper<SystemWrapper<FerdoMapModel<double>>,        Integrator> FerdoMap;
typedef DifferentialNeuronWrapper<SystemWrapper<MatsuokaModel<double>>,        Integrator> Matsuoka;
typedef DifferentialNeuronWrapper<SystemWrapper<VavoulisCGCModelQ10<double>>,  Integrator> VavoulisCGCQ10;
typedef DifferentialNeuronWrapper<SystemWrapper<VavoulisModel<double>>,        Integrator> Vavoulis;
typedef DifferentialNeuronWrapper<SystemWrapper<RowatSelverstonModel<double>>, Integrator> RowatSelverston;


// X-macro: unified list of neurons
#define NEURON_LIST(X) \
    X(HH) \
    X(BistableRulkov) \
    X(Rulkov) \
    X(SimpleOscillator) \
    X(Izhikevich) \
    X(VavoulisCGC) \
    X(FerdoMap) \
    X(Matsuoka) \
    X(VavoulisCGCQ10) \
    X(Vavoulis) \
    X(RowatSelverston)


// ============================================================
// Function to force the compiler
// ============================================================

volatile uint8_t g_checksum = 0;

// Reads object bytes and set them in checksum
// being volatile the compiler cannot discard its use
template<typename T>
__attribute__((noinline)) void checksum_object(const T& obj)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&obj);
    for (size_t i = 0; i < sizeof(T); ++i) {
        g_checksum ^= bytes[i];
    }
}


// ============================================================
// ENUM USED BY SYNAPSES
// ============================================================
//
// By default the parameter used for synapses will be v

template<typename T>
struct SynStateVar
{
    static constexpr auto value = T::v;
};

// For models without v it is explecitelly specified

template<>
struct SynStateVar<BistableRulkov>
{
    static constexpr auto value = BistableRulkov::x;
};

template<>
struct SynStateVar<Rulkov>
{
    static constexpr auto value = Rulkov::x;
};

template<>
struct SynStateVar<SimpleOscillator>
{
    static constexpr auto value = SimpleOscillator::x;
};

template<>
struct SynStateVar<FerdoMap>
{
    static constexpr auto value = FerdoMap::x;
};

#define SYN_VAR(T) SynStateVar<T>::value


// ============================================================
// EXERCISE A NEURON
// ============================================================

template<typename T>
__attribute__((noinline)) void exercise_neuron()
{
    typename T::ConstructorArgs args;

    T neuron(args);

    neuron.add_synaptic_input(1.0);
    neuron.step(0.001);

    checksum_object(neuron);
}

#define EXERCISE_ONE_NEURON(T) exercise_neuron<T>();

void exercise_all_neurons()
{
    NEURON_LIST(EXERCISE_ONE_NEURON)
}


// ============================================================
// EXERCISE A CHEMICAL SYNAPSE (PRE -> POST)
// ============================================================

template<typename PreT, typename PostT>
__attribute__((noinline)) void exercise_chemical_synapse()
{
    typedef ChemicalSynapse<PreT, PostT, Integrator, double> SynapseT;

    typename PreT::ConstructorArgs preArgs;
    typename PostT::ConstructorArgs postArgs;
   
    typename SynapseT::ConstructorArgs synArgs;

    PreT  pre(preArgs);
    PostT post(postArgs);

    SynapseT synapse(pre, SYN_VAR(PreT), post, SYN_VAR(PostT), synArgs, 1);

    pre.step(0.001);
    post.step(0.001);

    checksum_object(pre);
    checksum_object(post);
    checksum_object(synapse);
}

#define NEURON_LIST2(X, ARG) \
    X(ARG, HH) \
    X(ARG, BistableRulkov) \
    X(ARG, Rulkov) \
    X(ARG, SimpleOscillator) \
    X(ARG, Izhikevich) \
    X(ARG, VavoulisCGC) \
    X(ARG, FerdoMap) \
    X(ARG, Matsuoka) \
    X(ARG, VavoulisCGCQ10) \
    X(ARG, Vavoulis) \
    X(ARG, RowatSelverston)

#define EXERCISE_SYNAPSE_PAIR(PRE, POST) \
    exercise_chemical_synapse<PRE, POST>();

#define EXERCISE_SYNAPSE_ROW(PRE) NEURON_LIST2(EXERCISE_SYNAPSE_PAIR, PRE)

void exercise_all_chemical_synapses()
{
    // Complete matrix 11 x 11 = 121 combinations.
    NEURON_LIST(EXERCISE_SYNAPSE_ROW)
}


// ============================================================
// Exercise STDPSynapse / DiffusionSynapse / LinskerSynapse
// ============================================================

template<typename PreT, typename PostT>
__attribute__((noinline)) void exercise_stdp_synapse()
{
    typedef STDPSynapse<PreT, PostT, Integrator, double> SynapseT;

    typename PreT::ConstructorArgs preArgs;
    typename PostT::ConstructorArgs postArgs;
    typename SynapseT::ConstructorArgs synArgs;

    PreT  pre(preArgs);
    PostT post(postArgs);

    SynapseT synapse(pre, SYN_VAR(PreT), post, SYN_VAR(PostT), synArgs, 1);

    pre.step(0.001);
    post.step(0.001);
    synapse.step(0.001);

    checksum_object(pre);
    checksum_object(post);
    checksum_object(synapse);
}

#define EXERCISE_STDP_PAIR(PRE, POST) \
    exercise_stdp_synapse<PRE, POST>();

#define EXERCISE_STDP_ROW(PRE) NEURON_LIST2(EXERCISE_STDP_PAIR, PRE)

void exercise_all_stdp_synapses()
{
    NEURON_LIST(EXERCISE_STDP_ROW)
}


template<typename PreT, typename PostT>
__attribute__((noinline)) void exercise_diffusion_synapse()
{
    typedef DiffusionSynapse<PreT, PostT, Integrator, double> SynapseT;

    typename PreT::ConstructorArgs preArgs;
    typename PostT::ConstructorArgs postArgs;
    typename SynapseT::ConstructorArgs synArgs;

    PreT  pre(preArgs);
    PostT post(postArgs);

    SynapseT synapse(pre, SYN_VAR(PreT), post, SYN_VAR(PostT), synArgs, 1);

    pre.step(0.001);
    post.step(0.001);
    synapse.step(0.001);

    checksum_object(pre);
    checksum_object(post);
    checksum_object(synapse);
}

#define EXERCISE_DIFFUSION_PAIR(PRE, POST) \
    exercise_diffusion_synapse<PRE, POST>();

#define EXERCISE_DIFFUSION_ROW(PRE) NEURON_LIST2(EXERCISE_DIFFUSION_PAIR, PRE)

void exercise_all_diffusion_synapses()
{
    // Requiere el BUG 1 arreglado (ver cabecera del archivo).
    NEURON_LIST(EXERCISE_DIFFUSION_ROW)
}


template<typename PreT, typename PostT>
__attribute__((noinline)) void exercise_linsker_synapse()
{
    typedef LinskerSynapse<PreT, PostT, Integrator, double> SynapseT;

    typename PreT::ConstructorArgs preArgs;
    typename PostT::ConstructorArgs postArgs;
    typename SynapseT::ConstructorArgs synArgs;

    PreT  pre(preArgs);
    PostT post(postArgs);

    SynapseT synapse(pre, SYN_VAR(PreT), post, SYN_VAR(PostT), synArgs, 1);

    pre.step(0.001);
    post.step(0.001);
    synapse.step(0.001);

    checksum_object(pre);
    checksum_object(post);
    checksum_object(synapse);
}

#define EXERCISE_LINSKER_PAIR(PRE, POST) \
    exercise_linsker_synapse<PRE, POST>();

#define EXERCISE_LINSKER_ROW(PRE) NEURON_LIST2(EXERCISE_LINSKER_PAIR, PRE)

void exercise_all_linsker_synapses()
{
    NEURON_LIST(EXERCISE_LINSKER_ROW)
}



// ============================================================
// TEST with real values of HH
// ============================================================

HH::ConstructorArgs makeHHArgs()
{
    HH::ConstructorArgs args;

    args.params[HH::cm]  = 1 * 7.854e-3;
    args.params[HH::vna] = 50;
    args.params[HH::vk]  = -77;
    args.params[HH::vl]  = -54.387;

    args.params[HH::gna] = 120 * 7.854e-3;
    args.params[HH::gk]  = 36 * 7.854e-3;
    args.params[HH::gl]  = 0.3 * 7.854e-3;

    return args;
}

void run_real_neuron_test()
{
    HH::ConstructorArgs args = makeHHArgs();

    HH neuron(args);

    neuron.set(HH::v, -80);
    neuron.set(HH::m, 0.1);
    neuron.set(HH::n, 0.7);
    neuron.set(HH::h, 0.01);

    neuron.add_synaptic_input(10.0);
    neuron.step(0.001);

    double v = neuron.get(HH::v);

    Serial.print("HH voltage: ");
    Serial.println(v, 10);
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("========================================");
    Serial.println("Neun ALL-MODELS Flash Footprint Test");
    Serial.println("========================================");

    exercise_all_neurons();
    exercise_all_chemical_synapses();

    run_real_neuron_test();

    Serial.print("Checksum (must be random value != 0 if the script executed): ");
    Serial.println(g_checksum);

    Serial.println("All C++ types were instanciated and executed");
}

void loop()
{
}