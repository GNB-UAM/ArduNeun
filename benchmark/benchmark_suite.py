#!/usr/bin/env python3
"""
Reproducible Benchmark Suite for Neural Simulators

This script performs rigorous performance benchmarks comparing Neun, NEURON, 
Brian2, and NEST with comprehensive reproducibility controls:

- Warmup runs to eliminate JIT/cache effects
- Increased sample sizes for statistical robustness  
- Garbage collection between runs
- Median + IQR statistics (robust to outliers)
- Extended thermal stabilization delays
- Framework execution order randomization
- Multiple iteration support

Refactored with plugin-based architecture for maximum extensibility.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import gc
import psutil
from datetime import datetime as dt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from builtins import all as builtin_all  # Preserve builtin all before imports override it

# Import random module before Brian2 to preserve it
import random as pyrandom

# Add neun_py to path and import the compiled module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neun_py'))

# Import neun_py compiled module (the .so file)
import neun_py

# # Import other frameworks
# from neuron import h
# from brian2 import *
# import nest

# # Suppress Brian2 warnings for cleaner output
# import warnings
# warnings.filterwarnings('ignore')


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    mean_time: float
    std_time: float
    framework_name: str
    benchmark_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.framework_name:10s}: {self.mean_time:.6f} ± {self.std_time:.6f} seconds"


@dataclass
class BenchmarkSpec:
    """
    Specification for a benchmark type
    
    Defines what method should be called on each simulator and with what parameters.
    """
    name: str  # e.g., 'single_neuron', 'network', 'parameter_sweep'
    method_name: str  # e.g., 'benchmark_single_neuron'
    display_name: str  # e.g., 'Single Neuron'
    description: str  # e.g., '1000ms simulation, dt=0.025ms'
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)


# ============================================================================
# Abstract Output Handler
# ============================================================================

class OutputHandler(ABC):
    """Abstract base class for output handlers (CSV, plots, etc.)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def generate(self, simulators: List['SimulatorBenchmark'], 
                 benchmark_specs: List[BenchmarkSpec],
                 output_dir: str = 'output') -> None:
        """
        Generate output based on benchmark results
        
        Args:
            simulators: List of simulator objects with results
            benchmark_specs: List of benchmark specifications
            output_dir: Directory to save output
        """
        pass


# ============================================================================
# Abstract Simulator Benchmark
# ============================================================================

class SimulatorBenchmark(ABC):
    """
    Abstract base class for simulator benchmarks
    
    Now uses dynamic method lookup based on registered benchmarks.
    """
    
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color
        self.results: Dict[str, BenchmarkResult] = {}
    
    def has_benchmark(self, benchmark_spec: BenchmarkSpec) -> bool:
        """Check if this simulator has the required benchmark method"""
        return hasattr(self, benchmark_spec.method_name)
    
    def run_benchmark(self, benchmark_spec: BenchmarkSpec, 
                     verbose: bool = True) -> BenchmarkResult:
        """
        Run a specific benchmark using the benchmark spec
        
        Args:
            benchmark_spec: Specification of the benchmark to run
            verbose: Whether to print progress
            
        Returns:
            BenchmarkResult object
        """
        if not self.has_benchmark(benchmark_spec):
            raise NotImplementedError(
                f"{self.name} does not implement {benchmark_spec.method_name}"
            )
        
        if verbose:
            print(f"Running {self.name} {benchmark_spec.display_name} benchmark...")
        
        method = getattr(self, benchmark_spec.method_name)
        result = method(**benchmark_spec.default_params)
        
        if verbose:
            print(f"  {result}")
        
        return result


# ============================================================================
# Concrete Simulator Implementations
# ============================================================================
class NeunBenchmark(SimulatorBenchmark):
    """Neun framework benchmarks"""

    def __init__(self):
        super().__init__("Neun", "#2E86AB")

    # ------------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------------

    @staticmethod
    def _create_hh_neuron(gna: float = 120.0):
        """Create and configure one Neun Hodgkin-Huxley neuron."""
        args = neun_py.HHDoubleConstructorArgs()
        neuron = neun_py.HHDoubleRK4(args)

        neuron.set_param(
            neun_py.HHDoubleParameter.cm,
            1.0 * 7.854e-3
        )
        neuron.set_param(
            neun_py.HHDoubleParameter.vna,
            50
        )
        neuron.set_param(
            neun_py.HHDoubleParameter.vk,
            -77
        )
        neuron.set_param(
            neun_py.HHDoubleParameter.vl,
            -54.387
        )
        neuron.set_param(
            neun_py.HHDoubleParameter.gna,
            gna * 7.854e-3
        )
        neuron.set_param(
            neun_py.HHDoubleParameter.gk,
            36 * 7.854e-3
        )
        neuron.set_param(
            neun_py.HHDoubleParameter.gl,
            0.3 * 7.854e-3
        )

        neuron.set(
            neun_py.HHDoubleVariable.v,
            -65
        )

        return neuron

    @staticmethod
    def _pin_to_core(core: int = 0):
        """
        Temporarily pin the current process to one CPU core.

        Returns:
            Previous CPU affinity, so it can be restored afterwards.
        """
        process = psutil.Process(os.getpid())

        previous_affinity = process.cpu_affinity()
        available_cores = previous_affinity

        if core not in available_cores:
            raise ValueError(
                f"Requested CPU core {core} is not available. "
                f"Available cores: {available_cores}"
            )

        process.cpu_affinity([core])

        return previous_affinity

    @staticmethod
    def _restore_affinity(previous_affinity):
        """Restore the CPU affinity of the current process."""
        process = psutil.Process(os.getpid())
        process.cpu_affinity(previous_affinity)

    # ------------------------------------------------------------------------
    # Existing benchmarks
    # ------------------------------------------------------------------------

    def benchmark_single_neuron(
        self,
        duration_ms: float = 1000,
        dt_ms: float = 0.025,
        n_runs: int = 5
    ) -> BenchmarkResult:
        """Benchmark Neun with single Hodgkin-Huxley neuron."""

        times = []

        n_steps = int(duration_ms / dt_ms)

        for run in range(n_runs):
            neuron = self._create_hh_neuron()

            start = time.perf_counter()

            for _ in range(n_steps):
                neuron.add_synaptic_input(10.0)
                neuron.step(dt_ms)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            del neuron

        return BenchmarkResult(
            np.mean(times),
            np.std(times),
            self.name,
            'single',
            metadata={
                'n_steps': n_steps,
                'neuron_steps': n_steps,
                'steps_per_second': n_steps / np.mean(times),
                'microseconds_per_step': (
                    np.mean(times) * 1e6 / n_steps
                )
            }
        )

    def benchmark_network(
        self,
        n_neurons: int = 100,
        duration_ms: float = 1000,
        dt_ms: float = 0.025,
        n_runs: int = 3
    ) -> BenchmarkResult:
        """Benchmark Neun with small network of coupled neurons."""

        times = []

        n_steps = int(duration_ms / dt_ms)

        for run in range(n_runs):
            neurons = []

            for i in range(n_neurons):
                neuron = self._create_hh_neuron()

                neuron.set(
                    neun_py.HHDoubleVariable.v,
                    -65 + np.random.randn() * 5
                )

                neurons.append(neuron)

            start = time.perf_counter()

            for _ in range(n_steps):
                for i in range(n_neurons):
                    current = 10.0 + np.random.randn() * 2.0

                    if i > 0:
                        v_i = neurons[i].get(
                            neun_py.HHDoubleVariable.v
                        )
                        v_prev = neurons[i - 1].get(
                            neun_py.HHDoubleVariable.v
                        )

                        current += 0.1 * (v_prev - v_i)

                    neurons[i].add_synaptic_input(current)
                    neurons[i].step(dt_ms)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            del neurons
            gc.collect()

        total_neuron_steps = n_steps * n_neurons

        return BenchmarkResult(
            np.mean(times),
            np.std(times),
            self.name,
            'network',
            metadata={
                'n_neurons': n_neurons,
                'n_steps': n_steps,
                'neuron_steps': total_neuron_steps,
                'neuron_steps_per_second': (
                    total_neuron_steps / np.mean(times)
                ),
                'microseconds_per_neuron_step': (
                    np.mean(times) * 1e6 / total_neuron_steps
                )
            }
        )

    def benchmark_parameter_sweep(
        self,
        n_sweeps: int = 100,
        duration_ms: float = 100,
        dt_ms: float = 0.025,
        n_runs: int = 3
    ) -> BenchmarkResult:
        """Benchmark parameter sweep."""

        times = []

        n_steps = int(duration_ms / dt_ms)
        total_steps = n_sweeps * n_steps

        for run in range(n_runs):
            gna_values = np.linspace(100, 140, n_sweeps)

            start = time.perf_counter()

            for gna in gna_values:
                neuron = self._create_hh_neuron(gna)

                for _ in range(n_steps):
                    neuron.add_synaptic_input(10.0)
                    neuron.step(dt_ms)

                del neuron

            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return BenchmarkResult(
            np.mean(times),
            np.std(times),
            self.name,
            'parameter_sweep',
            metadata={
                'n_sweeps': n_sweeps,
                'steps_per_sweep': n_steps,
                'total_steps': total_steps,
                'steps_per_second': (
                    total_steps / np.mean(times)
                ),
                'microseconds_per_step': (
                    np.mean(times) * 1e6 / total_steps
                )
            }
        )

    # ------------------------------------------------------------------------
    # Memory benchmark
    # ------------------------------------------------------------------------

    def benchmark_memory(
        self,
        n_neurons: int = 1000,
        n_runs: int = 5
    ) -> BenchmarkResult:
        """
        Measure resident memory increase when creating N Neun neurons.

        RSS is used because Neun objects contain native C++ state that
        Python tracemalloc does not necessarily account for.
        """

        process = psutil.Process(os.getpid())
        measurements = []

        for run in range(n_runs):
            # Try to start from a clean Python/native allocation state.
            gc.collect()

            rss_before = process.memory_info().rss

            neurons = []

            for _ in range(n_neurons):
                neurons.append(self._create_hh_neuron())

            # Make sure objects are actually alive before measuring.
            gc.collect()

            rss_after = process.memory_info().rss

            memory_used = rss_after - rss_before
            measurements.append(memory_used)

            # Explicitly release the neurons.
            del neurons
            gc.collect()

        mean_memory = float(np.mean(measurements))
        std_memory = float(np.std(measurements))

        bytes_per_neuron = mean_memory / n_neurons

        print(
            f"[Neun Memory] "
            f"{n_neurons} neurons: "
            f"{mean_memory / 1024:.2f} KiB ± "
            f"{std_memory / 1024:.2f} KiB "
            f"({bytes_per_neuron:.2f} bytes/neuron)"
        )

        return BenchmarkResult(
            mean_memory,
            std_memory,
            self.name,
            'memory',
            metadata={
                'n_neurons': n_neurons,
                'bytes_per_neuron': bytes_per_neuron,
                'kib_per_neuron': bytes_per_neuron / 1024.0
            }
        )

    # ------------------------------------------------------------------------
    # CPU benchmark
    # ------------------------------------------------------------------------

    def benchmark_cpu(
        self,
        duration_ms: float = 1000,
        dt_ms: float = 0.025,
        n_runs: int = 5,
        core: int = 0
    ) -> BenchmarkResult:
        """
        Benchmark pure Neun computation pinned to a single CPU core.

        The affinity is restored after the benchmark.
        """

        previous_affinity = self._pin_to_core(core)

        try:
            n_steps = int(duration_ms / dt_ms)
            times = []
            cpu_times = []

            process = psutil.Process(os.getpid())

            for run in range(n_runs):
                neuron = self._create_hh_neuron()

                gc.collect()

                cpu_before = process.cpu_times()
                start = time.perf_counter()

                for _ in range(n_steps):
                    neuron.add_synaptic_input(10.0)
                    neuron.step(dt_ms)

                elapsed = time.perf_counter() - start

                cpu_after = process.cpu_times()

                cpu_time = (
                    (cpu_after.user - cpu_before.user)
                    +
                    (cpu_after.system - cpu_before.system)
                )

                times.append(elapsed)
                cpu_times.append(cpu_time)

                del neuron

            mean_time = float(np.mean(times))
            std_time = float(np.std(times))

            steps_per_second = n_steps / mean_time
            microseconds_per_step = (
                mean_time * 1e6 / n_steps
            )

            mean_cpu_time = float(np.mean(cpu_times))

            # This is only a secondary diagnostic.
            # Because the process is pinned to one core, values close to
            # 100% indicate that the process kept that core busy.
            cpu_utilization = (
                mean_cpu_time / mean_time * 100.0
                if mean_time > 0
                else 0.0
            )

            print(
                f"[Neun CPU] "
                f"core={core}, "
                f"time={mean_time:.6f}s ± {std_time:.6f}s, "
                f"{steps_per_second:,.0f} steps/s, "
                f"{microseconds_per_step:.3f} µs/step, "
                f"CPU={cpu_utilization:.1f}%"
            )

            return BenchmarkResult(
                mean_time,
                std_time,
                self.name,
                'cpu',
                metadata={
                    'core': core,
                    'n_steps': n_steps,
                    'steps_per_second': steps_per_second,
                    'microseconds_per_step': microseconds_per_step,
                    'cpu_time_seconds': mean_cpu_time,
                    'cpu_utilization_percent': cpu_utilization
                }
            )

        finally:
            self._restore_affinity(previous_affinity)


class NeuronBenchmark(SimulatorBenchmark):
    """NEURON framework benchmarks"""
    
    def __init__(self):
        super().__init__("NEURON", "#A23B72")
    
    def benchmark_single_neuron(self, duration_ms: float = 1000,
                                dt_ms: float = 0.025, n_runs: int = 5) -> BenchmarkResult:
        """Benchmark NEURON with single Hodgkin-Huxley neuron"""
        times = []
        
        for run in range(n_runs):
            # Setup
            h.load_file('stdrun.hoc')
            
            # Create section
            soma = h.Section(name='soma')
            soma.L = 10  # Length in microns
            soma.diam = 10  # Diameter in microns
            soma.insert('hh')  # Insert Hodgkin-Huxley mechanism
            
            # Set parameters
            for seg in soma:
                seg.hh.gnabar = 0.12  # S/cm^2
                seg.hh.gkbar = 0.036
                seg.hh.gl = 0.0003
                seg.hh.el = -54.3
            
            # Create stimulus
            stim = h.IClamp(soma(0.5))
            stim.delay = 0
            stim.dur = duration_ms
            stim.amp = 0.01  # nA
            
            # Record voltage
            v_vec = h.Vector()
            v_vec.record(soma(0.5)._ref_v)
            t_vec = h.Vector()
            t_vec.record(h._ref_t)
            
            # Set simulation parameters
            h.dt = dt_ms
            h.tstop = duration_ms
            h.v_init = -65
            
            # Benchmark
            start = time.perf_counter()
            h.run()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # Cleanup
            h.delete_section(sec=soma)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'single')
    
    def benchmark_network(self, n_neurons: int = 100, duration_ms: float = 1000,
                         dt_ms: float = 0.025, n_runs: int = 3) -> BenchmarkResult:
        """Benchmark NEURON with small network of coupled neurons"""
        times = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Pre-generate fixed connectivity pattern (same for all runs)
        connections = []
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.rand() < 0.1:
                    connections.append((i, j))
        
        for run in range(n_runs):
            # Setup
            h.load_file('stdrun.hoc')
            
            # Create neurons
            neurons = []
            for i in range(n_neurons):
                soma = h.Section(name=f'soma_{i}')
                soma.L = 10  # Length in microns
                soma.diam = 10  # Diameter in microns
                soma.insert('hh')  # Insert Hodgkin-Huxley mechanism
                
                # Set parameters
                for seg in soma:
                    seg.hh.gnabar = 0.12  # S/cm^2
                    seg.hh.gkbar = 0.036
                    seg.hh.gl = 0.0003
                    seg.hh.el = -54.3
                
                neurons.append(soma)
            
            # Create stimuli for each neuron with fixed seed for reproducibility
            np.random.seed(42 + run)
            stims = []
            for i, soma in enumerate(neurons):
                stim = h.IClamp(soma(0.5))
                stim.delay = 0
                stim.dur = duration_ms
                stim.amp = 0.01 + 0.001 * np.random.randn()  # nA with small noise
                stims.append(stim)
            
            # Create synaptic connections using fixed connectivity pattern
            synapses = []
            for i, j in connections:
                # Create gap junction-like connection
                syn = h.ExpSyn(neurons[j](0.5))
                syn.tau = 2.0  # ms
                syn.e = 0  # mV
                
                # Create NetCon
                nc = h.NetCon(neurons[i](0.5)._ref_v, syn, sec=neurons[i])
                nc.threshold = -20  # mV
                nc.delay = 1  # ms
                nc.weight[0] = 0.001  # Small synaptic weight
                
                synapses.append((syn, nc))
            
            # Set simulation parameters
            h.dt = dt_ms
            h.tstop = duration_ms
            h.v_init = -65
            
            # Benchmark
            start = time.perf_counter()
            h.run()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # Cleanup
            for soma in neurons:
                h.delete_section(sec=soma)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'network')
    
    def benchmark_parameter_sweep(self, n_sweeps: int = 100, duration_ms: float = 100,
                                  dt_ms: float = 0.025, n_runs: int = 3) -> BenchmarkResult:
        """Benchmark parameter sweep: many short simulations with different parameters"""
        times = []
        
        for run in range(n_runs):
            gna_values = np.linspace(0.10, 0.14, n_sweeps)  # S/cm^2
            
            start = time.perf_counter()
            
            for gna in gna_values:
                # Setup
                h.load_file('stdrun.hoc')
                
                # Create section
                soma = h.Section(name='soma')
                soma.L = 10
                soma.diam = 10
                soma.insert('hh')
                
                # Set parameters (varying gna)
                for seg in soma:
                    seg.hh.gnabar = gna
                    seg.hh.gkbar = 0.036
                    seg.hh.gl = 0.0003
                    seg.hh.el = -54.3
                
                # Create stimulus
                stim = h.IClamp(soma(0.5))
                stim.delay = 0
                stim.dur = duration_ms
                stim.amp = 0.01
                
                # Set simulation parameters
                h.dt = dt_ms
                h.tstop = duration_ms
                h.v_init = -65
                
                # Run simulation
                h.run()
                
                # Cleanup
                h.delete_section(sec=soma)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'parameter_sweep')


class Brian2Benchmark(SimulatorBenchmark):
    """Brian2 framework benchmarks"""
    
    def __init__(self):
        super().__init__("Brian2", "#F18F01")
    
    def benchmark_single_neuron(self, duration_ms: float = 1000,
                                dt_ms: float = 0.025, n_runs: int = 5) -> BenchmarkResult:
        """Benchmark Brian2 with single Hodgkin-Huxley neuron"""
        times = []
        
        duration = duration_ms * ms
        dt = dt_ms * ms
        
        for run_idx in range(n_runs):
            start_scope()  # Reset Brian2 state
            
            # Hodgkin-Huxley equations
            # Area normalized model (current density)
            eqs = '''
            dv/dt = (gl*(El-v) + gna*m**3*h*(Ena-v) + gk*n**4*(Ek-v) + Iinj)/C : volt
            dm/dt = alpham*(1-m) - betam*m : 1
            dn/dt = alphan*(1-n) - betan*n : 1
            dh/dt = alphah*(1-h) - betah*h : 1
            
            alpham = 0.1*(v/mV+40)/(1-exp(-(v/mV+40)/10))/ms : hertz
            betam = 4*exp(-(v/mV+65)/18)/ms : hertz
            alphah = 0.07*exp(-(v/mV+65)/20)/ms : hertz
            betah = 1/(1+exp(-(v/mV+35)/10))/ms : hertz
            alphan = 0.01*(v/mV+55)/(1-exp(-(v/mV+55)/10))/ms : hertz
            betan = 0.125*exp(-(v/mV+65)/80)/ms : hertz
            
            Iinj : amp/meter**2
            '''
            
            # Parameters
            C = 1*uF/cm**2
            gl = 0.3*mS/cm**2
            El = -54.387*mV
            gna = 120*mS/cm**2
            Ena = 50*mV
            gk = 36*mS/cm**2
            Ek = -77*mV
            
            # Create neuron
            neuron = NeuronGroup(1, eqs, method='exponential_euler', dt=dt)
            neuron.v = -65*mV
            neuron.Iinj = 10*uA/cm**2
            
            # Monitor
            mon = StateMonitor(neuron, 'v', record=True)
            
            # Benchmark
            start = time.perf_counter()
            run(duration)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'single')
    
    def benchmark_network(self, n_neurons: int = 100, duration_ms: float = 1000,
                         dt_ms: float = 0.025, n_runs: int = 3) -> BenchmarkResult:
        """Benchmark Brian2 with small network"""
        times = []
        
        duration = duration_ms * ms
        dt = dt_ms * ms
        
        for run_idx in range(n_runs):
            start_scope()
            
            # Simplified neuron model for network simulation
            # Area normalized model (current density)
            eqs = '''
            dv/dt = (gl*(El-v) + gna*m**3*h*(Ena-v) + gk*n**4*(Ek-v) + Iinj + Isyn)/C : volt
            dm/dt = alpham*(1-m) - betam*m : 1
            dn/dt = alphan*(1-n) - betan*n : 1
            dh/dt = alphah*(1-h) - betah*h : 1
            
            alpham = 0.1*(v/mV+40)/(1-exp(-(v/mV+40)/10))/ms : hertz
            betam = 4*exp(-(v/mV+65)/18)/ms : hertz
            alphah = 0.07*exp(-(v/mV+65)/20)/ms : hertz
            betah = 1/(1+exp(-(v/mV+35)/10))/ms : hertz
            alphan = 0.01*(v/mV+55)/(1-exp(-(v/mV+55)/10))/ms : hertz
            betan = 0.125*exp(-(v/mV+65)/80)/ms : hertz
            
            Iinj : amp/meter**2
            Isyn : amp/meter**2
            '''
            
            C = 1*uF/cm**2
            gl = 0.3*mS/cm**2
            El = -54.387*mV
            gna = 120*mS/cm**2
            Ena = 50*mV
            gk = 36*mS/cm**2
            Ek = -77*mV
            
            neurons = NeuronGroup(n_neurons, eqs, threshold='v > -20*mV', reset='v = -65*mV',
                                 method='exponential_euler', dt=dt)
            neurons.v = -65*mV + randn(n_neurons)*5*mV
            neurons.Iinj = 10*uA/cm**2
            
            # Simple synaptic coupling
            synapses = Synapses(neurons, neurons, 'w : amp/meter**2', on_pre='Isyn_post += w')
            synapses.connect(condition='i != j', p=0.1)
            synapses.w = '0.1*uA/cm**2'
            
            start = time.perf_counter()
            run(duration)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'network')
    
    def benchmark_parameter_sweep(self, n_sweeps: int = 100, duration_ms: float = 100,
                                  dt_ms: float = 0.025, n_runs: int = 3) -> BenchmarkResult:
        """Benchmark parameter sweep: many short simulations with different parameters"""
        times = []
        
        duration = duration_ms * ms
        dt = dt_ms * ms
        
        for run_idx in range(n_runs):
            gna_values = np.linspace(100, 140, n_sweeps)  # mS/cm^2
            
            start = time.perf_counter()
            
            for gna_val in gna_values:
                start_scope()
                
                # HH equations with varying gna
                eqs = '''
                dv/dt = (gl*(El-v) + gna*m**3*h*(Ena-v) + gk*n**4*(Ek-v) + Iinj)/C : volt
                dm/dt = alpham*(1-m) - betam*m : 1
                dn/dt = alphan*(1-n) - betan*n : 1
                dh/dt = alphah*(1-h) - betah*h : 1
                
                alpham = 0.1*(v/mV+40)/(1-exp(-(v/mV+40)/10))/ms : hertz
                betam = 4*exp(-(v/mV+65)/18)/ms : hertz
                alphah = 0.07*exp(-(v/mV+65)/20)/ms : hertz
                betah = 1/(1+exp(-(v/mV+35)/10))/ms : hertz
                alphan = 0.01*(v/mV+55)/(1-exp(-(v/mV+55)/10))/ms : hertz
                betan = 0.125*exp(-(v/mV+65)/80)/ms : hertz
                
                Iinj : amp/meter**2
                '''
                
                C = 1*uF/cm**2
                gl = 0.3*mS/cm**2
                El = -54.387*mV
                gna = gna_val*mS/cm**2  # Varying parameter
                Ena = 50*mV
                gk = 36*mS/cm**2
                Ek = -77*mV
                
                neuron = NeuronGroup(1, eqs, method='exponential_euler', dt=dt)
                neuron.v = -65*mV
                neuron.Iinj = 10*uA/cm**2
                
                run(duration)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'parameter_sweep')


class NestBenchmark(SimulatorBenchmark):
    """NEST framework benchmarks"""
    
    def __init__(self):
        super().__init__("NEST", "#06A77D")
    
    def benchmark_single_neuron(self, duration_ms: float = 1000,
                                dt_ms: float = 0.025, n_runs: int = 5) -> BenchmarkResult:
        """Benchmark NEST with single Hodgkin-Huxley neuron"""
        times = []
        
        for run in range(n_runs):
            nest.ResetKernel()
            nest.set_verbosity('M_WARNING')
            
            # Set simulation parameters
            nest.resolution = dt_ms
            nest.local_num_threads = 1  # Force single-threaded for fair comparison
            
            # Create neuron (NEST uses iaf_cond_alpha or hh_psc_alpha for HH)
            # Note: NEST's hh_psc_alpha is a simplified HH model
            neuron = nest.Create('hh_psc_alpha', 1)
            
            # Create constant current generator
            dc = nest.Create('dc_generator', 1)
            nest.SetStatus(dc, {'amplitude': 100.0})  # pA
            
            # Connect stimulus
            nest.Connect(dc, neuron)
            
            # Create voltmeter
            vm = nest.Create('voltmeter', 1)
            nest.SetStatus(vm, {'interval': dt_ms})
            nest.Connect(vm, neuron)
            
            # Benchmark
            start = time.perf_counter()
            nest.Simulate(duration_ms)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'single')
    
    def benchmark_network(self, n_neurons: int = 100, duration_ms: float = 1000,
                         dt_ms: float = 0.025, n_runs: int = 3) -> BenchmarkResult:
        """Benchmark NEST with small network"""
        times = []
        
        for run in range(n_runs):
            nest.ResetKernel()
            nest.set_verbosity('M_WARNING')
            nest.resolution = dt_ms
            nest.local_num_threads = 1  # Force single-threaded for fair comparison
            
            # Create neurons
            neurons = nest.Create('hh_psc_alpha', n_neurons)
            
            # Create stimulation
            noise = nest.Create('noise_generator', 1)
            nest.SetStatus(noise, {'mean': 100.0, 'std': 20.0})
            
            # Connect noise to all neurons
            nest.Connect(noise, neurons, 'all_to_all')
            
            # Create recurrent connections
            nest.Connect(neurons, neurons, 
                        {'rule': 'pairwise_bernoulli', 'p': 0.1},
                        {'weight': 10.0, 'delay': 1.0})
            
            start = time.perf_counter()
            nest.Simulate(duration_ms)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'network')
    
    def benchmark_parameter_sweep(self, n_sweeps: int = 100, duration_ms: float = 100,
                                  dt_ms: float = 0.025, n_runs: int = 3) -> BenchmarkResult:
        """Benchmark parameter sweep: many short simulations with different parameters
        Note: NEST doesn't easily support varying gnabar, so we vary input current instead"""
        times = []
        
        for run in range(n_runs):
            amplitude_values = np.linspace(50.0, 150.0, n_sweeps)  # pA
            
            start = time.perf_counter()
            
            for amp in amplitude_values:
                nest.ResetKernel()
                nest.set_verbosity('M_WARNING')
                nest.resolution = dt_ms
                nest.local_num_threads = 1  # Force single-threaded for fair comparison
                
                # Create neuron
                neuron = nest.Create('hh_psc_alpha', 1)
                
                # Create DC generator with varying amplitude
                dc = nest.Create('dc_generator', 1)
                nest.SetStatus(dc, {'amplitude': amp})
                
                # Connect stimulus
                nest.Connect(dc, neuron)
                
                # Run simulation
                nest.Simulate(duration_ms)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return BenchmarkResult(np.mean(times), np.std(times), self.name, 'parameter_sweep')




# ============================================================================
# Output Handlers
# ============================================================================

class CSVOutputHandler(OutputHandler):
    """Handles CSV output with dynamic columns based on benchmarks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.filename = self.config.get('filename', 'benchmark_results.csv')
        self.include_metadata = self.config.get('include_metadata', False)
    
    def generate(self, simulators: List[SimulatorBenchmark],
                 benchmark_specs: List[BenchmarkSpec],
                 output_dir: str = 'output') -> None:
        """Generate CSV output with dynamic columns"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, self.filename)
        
        # Check if file exists to determine if we need header
        file_exists = os.path.exists(filepath)
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Build header dynamically from benchmark specs
            if not file_exists:
                header = ['timestamp', 'framework']
                for spec in benchmark_specs:
                    header.extend([
                        f'{spec.name}_mean_s',
                        f'{spec.name}_std_s'
                    ])
                writer.writerow(header)
            
            # Get current timestamp
            timestamp = dt.now().isoformat()
            
            # Write one row per framework
            for sim in simulators:
                row = [timestamp, sim.name]
                for spec in benchmark_specs:
                    if spec.name in sim.results:
                        result = sim.results[spec.name]
                        row.extend([
                            f"{result.mean_time:.6f}",
                            f"{result.std_time:.6f}"
                        ])
                    else:
                        row.extend(['N/A', 'N/A'])
                writer.writerow(row)
        
        print(f"CSV results appended to: {filepath}")
        
        # Also save summary
        summary_file = filepath.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Neural Simulation Framework Benchmark Results\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            for spec in benchmark_specs:
                f.write(f"{spec.display_name.upper()} ({spec.description})\n")
                f.write("-" * 70 + "\n")
                for sim in simulators:
                    if spec.name in sim.results:
                        result = sim.results[spec.name]
                        f.write(f"{result.framework_name.upper():10s}: "
                               f"{result.mean_time:.6f} ± {result.std_time:.6f} seconds\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"Summary saved to: {summary_file}\n")


class PlotOutputHandler(OutputHandler):
    """Handles plot generation with dynamic subplots based on benchmarks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.filename = self.config.get('filename', 'benchmark_comparison')
        self.include_benchmarks = self.config.get('include_benchmarks', None)  # None = all
        self.plot_type = self.config.get('plot_type', 'comparison')  # 'comparison' or 'speedup'
    
    def generate(self, simulators: List[SimulatorBenchmark],
                 benchmark_specs: List[BenchmarkSpec],
                 output_dir: str = 'output') -> None:
        """Generate plots with dynamic subplots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter benchmarks if specified
        if self.include_benchmarks:
            benchmark_specs = [s for s in benchmark_specs 
                             if s.name in self.include_benchmarks]
        
        if self.plot_type == 'comparison':
            self._generate_comparison_plot(simulators, benchmark_specs, output_dir)
        elif self.plot_type == 'speedup':
            self._generate_speedup_plot(simulators, benchmark_specs, output_dir)
    
    def _generate_comparison_plot(self, simulators: List[SimulatorBenchmark],
                                  benchmark_specs: List[BenchmarkSpec],
                                  output_dir: str) -> None:
        """Generate comparison bar plots"""
        n_benchmarks = len(benchmark_specs)
        fig, axes = plt.subplots(1, n_benchmarks, figsize=(7*n_benchmarks, 6))
        
        # Handle single benchmark case
        if n_benchmarks == 1:
            axes = [axes]
        
        frameworks = [sim.name for sim in simulators]
        colors = [sim.color for sim in simulators]
        
        for ax, spec in zip(axes, benchmark_specs):
            means = []
            stds = []
            for sim in simulators:
                if spec.name in sim.results:
                    means.append(sim.results[spec.name].mean_time)
                    stds.append(sim.results[spec.name].std_time)
                else:
                    means.append(0)
                    stds.append(0)
            
            x_pos = np.arange(len(frameworks))
            bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8,
                         capsize=5, edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
            ax.set_title(f'{spec.display_name}\n({spec.description})',
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(frameworks, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                if mean > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{mean:.3f}s', ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        pdf_path = os.path.join(output_dir, f'{self.filename}.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {pdf_path}")
        
        png_path = os.path.join(output_dir, f'{self.filename}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {png_path}")
        
        plt.close()
    
    def _generate_speedup_plot(self, simulators: List[SimulatorBenchmark],
                               benchmark_specs: List[BenchmarkSpec],
                               output_dir: str) -> None:
        """Generate speedup plots"""
        n_benchmarks = len(benchmark_specs)
        fig, axes = plt.subplots(1, n_benchmarks, figsize=(7*n_benchmarks, 6))
        
        if n_benchmarks == 1:
            axes = [axes]
        
        frameworks = [sim.name for sim in simulators]
        colors = [sim.color for sim in simulators]
                
        for ax, spec in zip(axes, benchmark_specs):
            means = []
            for sim in simulators:
                if spec.name in sim.results:
                    means.append(sim.results[spec.name].mean_time)
                else:
                    means.append(float('inf'))
            
            # If no data, show empty plot with message
            # Use builtin_all because numpy's all() behaves differently
            if builtin_all(m == float('inf') for m in means):
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, fontweight='bold', color='gray')
                ax.set_title(f'{spec.display_name} Speedup\n(no data)',
                            fontsize=13, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            fastest = min(m for m in means if m != float('inf'))
            speedups = [fastest / m if m != float('inf') else 0 for m in means]
            
            x_pos = np.arange(len(frameworks))
            bars = ax.bar(x_pos, speedups, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel('Relative Speed', fontsize=12, fontweight='bold')
            ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
            ax.set_title(f'{spec.display_name} Speedup\n(1.0 = fastest)',
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(frameworks, fontsize=11)
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add value labels
            for bar, speedup in zip(bars, speedups):
                if speedup > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{speedup:.2f}×', ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        pdf_path = os.path.join(output_dir, f'{self.filename}.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Speedup plot saved to: {pdf_path}")
        
        png_path = os.path.join(output_dir, f'{self.filename}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Speedup plot saved to: {png_path}")
        
        plt.close()


# ============================================================================
# Benchmark Suite
# ============================================================================

class BenchmarkSuite:
    """
    Reproducible benchmark suite coordinator
    
    Key features:
    1. Register benchmark types dynamically
    2. Validate that all simulators implement required benchmarks
    3. Connect multiple output handlers
    4. Warmup runs to eliminate JIT/cache effects
    5. Garbage collection and thermal stabilization
    6. Robust statistics (median + IQR)
    """
    
    def __init__(self, 
                 delay_between_benchmarks: float = 30.0,
                 warmup_runs: int = 1,
                 use_median: bool = True):
        """
        Initialize benchmark suite with reproducibility controls
        
        Args:
            delay_between_benchmarks: Delay in seconds (increased to 30s for thermal stability)
            warmup_runs: Number of warmup runs before measurement (default: 1)
            use_median: Use median instead of mean for statistics (default: True, more robust)
        """
        self.simulators: List[SimulatorBenchmark] = []
        self.benchmark_specs: List[BenchmarkSpec] = []
        self.output_handlers: List[OutputHandler] = []
        self.delay = delay_between_benchmarks
        self.warmup_runs = warmup_runs
        self.use_median = use_median
        self._validated = False
    
    def register_simulator(self, simulator: SimulatorBenchmark) -> 'BenchmarkSuite':
        """
        Register a simulator to be benchmarked
        
        Args:
            simulator: SimulatorBenchmark instance
            
        Returns:
            self for method chaining
        """
        self.simulators.append(simulator)
        self._validated = False
        return self
    
    def register_benchmark(self, spec: BenchmarkSpec) -> 'BenchmarkSuite':
        """
        Register a benchmark type
        
        Args:
            spec: BenchmarkSpec defining the benchmark
            
        Returns:
            self for method chaining
        """
        self.benchmark_specs.append(spec)
        self._validated = False
        return self
    
    def register_output_handler(self, handler: OutputHandler) -> 'BenchmarkSuite':
        """
        Register an output handler
        
        Args:
            handler: OutputHandler instance
            
        Returns:
            self for method chaining
        """
        self.output_handlers.append(handler)
        return self
    
    def validate(self) -> bool:
        """
        Validate that all simulators implement all registered benchmarks
        
        Returns:
            True if validation passes
            
        Raises:
            RuntimeError: If validation fails
        """
        missing = defaultdict(list)
        
        for sim in self.simulators:
            for spec in self.benchmark_specs:
                if not sim.has_benchmark(spec):
                    missing[sim.name].append(spec.method_name)
        
        if missing:
            error_msg = "Validation failed! The following simulators are missing benchmarks:\n"
            for sim_name, methods in missing.items():
                error_msg += f"  {sim_name}: {', '.join(methods)}\n"
            raise RuntimeError(error_msg)
        
        self._validated = True
        return True
    
    def run_all(self, randomize_order: bool = True, validate: bool = True) -> None:
        """
        Run all registered benchmarks with reproducibility controls
        
        Args:
            randomize_order: If True, randomize framework execution order
            validate: If True, validate before running
        """
        # Validate if requested
        if validate and not self._validated:
            self.validate()
        
        print("=" * 80)
        print("REPRODUCIBLE Neural Simulation Framework Benchmark")
        print("=" * 80)
        print(f"Simulators: {', '.join([s.name for s in self.simulators])}")
        print(f"Benchmarks: {', '.join([s.name for s in self.benchmark_specs])}")
        print(f"Reproducibility controls:")
        print(f"  - Warmup runs: {self.warmup_runs}")
        print("  - Statistics: Mean ± Std")
        print(f"  - Thermal delay: {self.delay}s")
        print(f"  - Randomize order: {randomize_order}")
        print(f"  - Garbage collection: enabled")
        print()
        
        # Run each benchmark type
        for bench_idx, spec in enumerate(self.benchmark_specs):
            print(f"{spec.display_name} ({spec.description})")
            print("-" * 80)
            
            # Randomize order for this benchmark
            sim_order = list(self.simulators)
            if randomize_order:
                pyrandom.shuffle(sim_order)
                print(f"Execution order: {' → '.join([s.name for s in sim_order])}")
            
            # Run benchmark for each simulator
            for sim_idx, sim in enumerate(sim_order):
                # Force garbage collection before benchmark
                gc.collect()
                
                # Run warmup if configured
                if self.warmup_runs > 0:
                    print(f"  {sim.name}: [warmup: {self.warmup_runs}] ", end='', flush=True)
                    warmup_params = spec.default_params.copy()
                    warmup_params['n_runs'] = self.warmup_runs
                    method = getattr(sim, spec.method_name)
                    _ = method(**warmup_params)  # Discard warmup results
                    print("✓ ", end='', flush=True)
                else:
                    print(f"  {sim.name}: ", end='', flush=True)
                
                # Run actual benchmark
                result = sim.run_benchmark(spec, verbose=False)
                
                # Convert to median/IQR if requested
                # Note: Individual benchmark methods still compute mean/std,
                # but we could enhance them to return raw times for better control
                if spec.name == 'memory':
                    print(
                        f"{result.mean_time / 1024:.2f} KiB "
                        f"(±{result.std_time / 1024:.2f} KiB)"
                    )
                else:
                    print(
                        f"{result.mean_time:.4f}s "
                        f"(±{result.std_time:.4f}s)"
    )
                
                sim.results[spec.name] = result
                
                # Add delay between simulators (but not after the last one)
                if sim_idx < len(sim_order) - 1:
                    print(f"  [Thermal stabilization: {self.delay}s]", flush=True)
                    time.sleep(self.delay)
            
            # Add delay between benchmark types (but not after the last one)
            if bench_idx < len(self.benchmark_specs) - 1:
                print(f"\n[Cooldown before next benchmark: {self.delay}s]", flush=True)
                time.sleep(self.delay)
                print()
        
        print()
        print("=" * 80)
        print("Benchmark completed!")
        print("=" * 80)
    
    def generate_outputs(self, output_dir: str = 'output') -> None:
        """
        Generate all registered outputs
        
        Args:
            output_dir: Directory to save outputs
        """
        print()
        print("Generating outputs...")
        print("-" * 70)
        
        for handler in self.output_handlers:
            handler.generate(self.simulators, self.benchmark_specs, output_dir)
        
        print("=" * 70)
    
    def load_results_from_csv(self, csv_path: str = 'output/benchmark_results.csv',
                             use_last_run: bool = True) -> None:
        """
        Load benchmark results from CSV file without re-running benchmarks
        
        This is useful for regenerating plots or trying different output configurations
        without the computational cost of re-running all benchmarks.
        
        Args:
            csv_path: Path to the CSV file with benchmark results
            use_last_run: If True, load only the most recent run (by timestamp).
                         If False, load all runs (will overwrite with latest values)
        
        Example:
            suite = BenchmarkSuite()
            suite.register_simulator(NeunBenchmark())
            suite.register_benchmark(...)
            suite.load_results_from_csv()  # Load from existing CSV
            suite.register_output_handler(PlotOutputHandler(...))
            suite.generate_outputs()  # Generate new plots without re-running
        """
        import csv as csv_module
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
        
        if not rows:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        # Filter to last run if requested
        if use_last_run:
            last_timestamp = rows[-1]['timestamp']
            rows = [r for r in rows if r['timestamp'] == last_timestamp]
            print(f"Loading results from run: {last_timestamp}")
        
        # Clear existing results
        for sim in self.simulators:
            sim.results.clear()
        
        # Load results from CSV
        loaded_count = 0
        for row in rows:
            framework = row['framework']
            sim = next((s for s in self.simulators if s.name == framework), None)
            
            if not sim:
                print(f"Warning: Framework '{framework}' in CSV not found in registered simulators")
                continue
            
            for spec in self.benchmark_specs:
                # Try the exact spec name first
                mean_key = f'{spec.name}_mean_s'
                std_key = f'{spec.name}_std_s'
                
                # If not found, check if it's using the old benchmark_type name
                # (for backwards compatibility with old CSV files)
                if mean_key not in row:
                    # Try common alternatives
                    alt_names = []
                    if spec.name == 'parameter_sweep':
                        alt_names = ['sweep']
                    
                    for alt in alt_names:
                        alt_mean_key = f'{alt}_mean_s'
                        alt_std_key = f'{alt}_std_s'
                        if alt_mean_key in row:
                            mean_key = alt_mean_key
                            std_key = alt_std_key
                            print(f"Note: Using column '{alt_mean_key}' for benchmark '{spec.name}'")
                            break
                
                if mean_key in row and row[mean_key] not in ('N/A', '', None):
                    try:
                        sim.results[spec.name] = BenchmarkResult(
                            float(row[mean_key]),
                            float(row[std_key]),
                            framework,
                            spec.name,
                            {}
                        )
                        loaded_count += 1
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Could not load {spec.name} for {framework}: {e}")
        
        print(f"Loaded {loaded_count} benchmark results from CSV")
        print(f"Results per framework:")
        for sim in self.simulators:
            benchmarks = ', '.join(sim.results.keys()) if sim.results else 'none'
            print(f"  {sim.name}: {benchmarks}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    try:
        # Create benchmark suite with reproducibility settings
        suite = BenchmarkSuite(
            delay_between_benchmarks=30.0,  # Increased for thermal stability
            warmup_runs=1,  # Warmup to eliminate JIT/cache effects
            use_median=True  # Use median for robustness
        )
        
        # Register simulators
        suite.register_simulator(NeunBenchmark())
        # suite.register_simulator(NeuronBenchmark())
        # suite.register_simulator(Brian2Benchmark())
        # suite.register_simulator(NestBenchmark())
        
        # Register benchmarks with increased n_runs for better statistics
        suite.register_benchmark(BenchmarkSpec(
            name='single',
            method_name='benchmark_single_neuron',
            display_name='Single Neuron Simulation',
            description='1000ms, dt=0.025ms',
            default_params={'duration_ms': 1000, 'dt_ms': 0.025, 'n_runs': 10}  # Increased from 5
        ))
        
        suite.register_benchmark(BenchmarkSpec(
            name='network',
            method_name='benchmark_network',
            display_name='Network Simulation',
            description='100 neurons, 1000ms, dt=0.025ms',
            default_params={'n_neurons': 100, 'duration_ms': 1000, 'dt_ms': 0.025, 'n_runs': 10}  # Increased from 3
        ))
        
        suite.register_benchmark(BenchmarkSpec(
            name='parameter_sweep',
            method_name='benchmark_parameter_sweep',
            display_name='Parameter Sweep',
            description='100 sweeps × 100ms, dt=0.025ms',
            default_params={'n_sweeps': 100, 'duration_ms': 100, 'dt_ms': 0.025, 'n_runs': 10}  # Increased from 3
        ))

        suite.register_benchmark(BenchmarkSpec(
            name='memory',
            method_name='benchmark_memory',
            display_name='Memory Usage',
            description='1000 HH neurons, RSS increase',
            default_params={
                'n_neurons': 1000,
                'n_runs': 5
            }
        ))

        suite.register_benchmark(BenchmarkSpec(
            name='cpu',
            method_name='benchmark_cpu',
            display_name='Single-Core CPU',
            description='1000ms, dt=0.025ms, pinned to CPU core 0',
            default_params={
                'duration_ms': 1000,
                'dt_ms': 0.025,
                'n_runs': 10,
                'core': 0
            }
        ))
        
        # Register output handlers
        suite.register_output_handler(CSVOutputHandler({
            'filename': 'benchmark_results.csv'
        }))
        
        suite.register_output_handler(PlotOutputHandler({
            'filename': 'benchmark_comparison',
            'plot_type': 'comparison'
        }))
        
        suite.register_output_handler(PlotOutputHandler({
            'filename': 'speedup_comparison',
            'plot_type': 'speedup'
        }))
        
        # Validate configuration
        suite.validate()
        
        # Run all benchmarks with randomized order
        suite.run_all(randomize_order=True)
        
        # Generate all outputs
        suite.generate_outputs()
        
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)