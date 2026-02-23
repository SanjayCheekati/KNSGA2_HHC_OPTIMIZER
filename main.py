"""
K-NSGA-II: Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm
==============================================================================

A novel hybrid optimization framework combining K-means clustering with 
NSGA-II for solving the Home Health Care Multi-Objective Vehicle Routing 
Problem with Time Windows (HHC-MOVRPTW).

Algorithm Architecture:
    Stage 1 - DECOMPOSITION: K-means clustering partitions customers geographically
    Stage 2 - OPTIMIZATION: NSGA-II optimizes each cluster independently  
    Stage 3 - COMBINATION: Pareto fronts are merged into global optimal front

Objectives:
    F1: Minimize total service time (travel + service duration)
    F2: Minimize total tardiness (patient preference violations)

Version: 2.1.0
"""

import sys
import os
import time
import random

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance, list_available_instances
from src.hybrid_knsga2 import KNSGAII

# Benchmark instances for evaluation
BENCHMARK_INSTANCES = ['C101.25', 'C101.100', 'C107.100', 'C206.50', 'R109.25', 'RC106.50']

# Algorithm parameter presets
PARAM_PRESETS = {
    'fast': {'population': 50, 'generations': 100, 'desc': 'Quick evaluation (~2-3s)'},
    'standard': {'population': 100, 'generations': 500, 'desc': 'Balanced (~30s)'},
    'research': {'population': 100, 'generations': 1000, 'desc': 'Full convergence (~60s)'}
}


def run_optimization(instance_name: str, population: int, generations: int, 
                     num_runs: int = 1, verbose: bool = True):
    """
    Execute K-NSGA-II optimization on a problem instance.
    
    Args:
        instance_name: Name of the benchmark instance (e.g., 'C101.25')
        population: Population size for NSGA-II
        generations: Maximum number of generations
        num_runs: Number of independent runs for statistical analysis
        verbose: Print detailed progress
    
    Returns:
        Dictionary containing optimization results and metrics
    """
    instance = load_instance(instance_name)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  K-NSGA-II Optimization: {instance_name}")
        print(f"{'='*70}")
        print(f"  Problem Size: {instance.num_customers} customers, {instance.num_vehicles} vehicles")
        print(f"  Parameters:   population={population}, generations={generations}")
        print(f"  Runs:         {num_runs}")
        print(f"{'-'*70}")
    
    all_hvs = []
    all_sps = []
    all_times = []
    all_pareto_sizes = []
    
    for run in range(num_runs):
        start = time.time()
        
        knsga2 = KNSGAII(
            instance=instance,
            population_size=population,
            max_generations=generations,
            crossover_rate=0.7,
            mutation_rate=0.2,
            random_state=run * 7 + 42
        )
        
        knsga2.run(verbose=(verbose and run == 0 and num_runs == 1))
        metrics = knsga2.get_performance_metrics()
        elapsed = time.time() - start
        
        all_hvs.append(metrics['hypervolume'])
        all_sps.append(metrics['spacing'])
        all_times.append(elapsed)
        all_pareto_sizes.append(metrics['pareto_size'])
        
        if num_runs > 1 and verbose:
            print(f"  Run {run+1:2d}/{num_runs}: Hv={metrics['hypervolume']:.4f}, "
                  f"SP={metrics['spacing']:.4f}, Pareto={metrics['pareto_size']:3d}, "
                  f"Time={elapsed:.2f}s")
    
    # Calculate statistics
    results = {
        'instance': instance_name,
        'avg_hv': sum(all_hvs) / len(all_hvs),
        'min_hv': min(all_hvs),
        'max_hv': max(all_hvs),
        'avg_sp': sum(all_sps) / len(all_sps),
        'avg_time': sum(all_times) / len(all_times),
        'avg_pareto_size': sum(all_pareto_sizes) / len(all_pareto_sizes),
        'runs': num_runs
    }
    
    if verbose:
        print(f"{'-'*70}")
        if num_runs > 1:
            print(f"  STATISTICAL SUMMARY ({num_runs} runs):")
            print(f"    Hypervolume:  {results['avg_hv']:.4f} "
                  f"(range: {results['min_hv']:.4f} - {results['max_hv']:.4f})")
            print(f"    Spacing:      {results['avg_sp']:.4f}")
            print(f"    Pareto Size:  {results['avg_pareto_size']:.1f} solutions")
            print(f"    Avg Time:     {results['avg_time']:.2f}s per run")
        else:
            print(f"  RESULTS:")
            print(f"    Hypervolume:  {results['avg_hv']:.4f}")
            print(f"    Spacing:      {results['avg_sp']:.4f}")
            print(f"    Pareto Size:  {int(results['avg_pareto_size'])} solutions")
            print(f"    Time:         {results['avg_time']:.2f}s")
        print(f"{'='*70}")
    
    return results


def run_with_params(instance_name: str):
    """Select parameters and run optimization."""
    
    print(f"\n  Selected: {instance_name}")
    print("\n  Parameter Presets:")
    print("    1. Fast     (pop=50,  gen=100)   - Quick evaluation")
    print("    2. Standard (pop=100, gen=500)   - Balanced")
    print("    3. Research (pop=100, gen=1000)  - Full convergence")
    print("    4. Custom parameters")
    
    preset = input("\n  Choice (1-4) [default=1]: ").strip() or '1'
    
    if preset == '1':
        pop, gen = 50, 100
    elif preset == '2':
        pop, gen = 100, 500
    elif preset == '3':
        pop, gen = 100, 1000
    elif preset == '4':
        pop = int(input("  Population size [100]: ").strip() or '100')
        gen = int(input("  Generations [500]: ").strip() or '500')
    else:
        pop, gen = 50, 100
    
    runs_input = input("  Number of runs [1]: ").strip() or '1'
    num_runs = int(runs_input)
    
    run_optimization(instance_name, pop, gen, num_runs, verbose=True)


def interactive_menu():
    """Interactive command-line interface for algorithm execution."""
    
    while True:
        print("\n" + "=" * 70)
        print("  K-NSGA-II: Multi-Objective Home Health Care Optimization")
        print("=" * 70)
        print("\n  Select Option:")
        print("  " + "-" * 50)
        
        for i, inst in enumerate(BENCHMARK_INSTANCES, 1):
            parts = inst.split('.')
            customers = parts[1] if len(parts) > 1 else '?'
            print(f"    {i}. {inst:<15} ({customers} customers)")
        
        n = len(BENCHMARK_INSTANCES)
        # custom instance follows the last benchmark
        print(f"\n    {n + 1}. Custom instance")
        print(f"    0. Exit")
        
        print("  " + "-" * 50)
        
        try:
            choice = input(f"\n  Enter choice (0-{n + 2}): ").strip()
            
            if choice == '0':
                print("\n  Thank you for using K-NSGA-II. Goodbye!\n")
                break
            
            elif choice.isdigit() and 1 <= int(choice) <= n:
                selected_instance = BENCHMARK_INSTANCES[int(choice) - 1]
                run_with_params(selected_instance)
            
            elif choice == str(n + 1):
                custom = input("  Enter instance name (e.g., C101.50): ").strip().upper()
                try:
                    run_with_params(custom)
                except FileNotFoundError:
                    print(f"\n  Error: Instance '{custom}' not found!")
                    print(f"  Available: {', '.join(list_available_instances()[:10])}...")
            
            else:
                print(f"  Invalid choice. Please enter 0-{n + 1}.")
                
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print(__doc__)
            print("\nUsage:")
            print("  python main.py                     # Interactive mode")
            print("  python main.py <instance>          # Quick run")
            print("  python main.py <instance> <pop> <gen> <runs>")
            print("\nExamples:")
            print("  python main.py C101.25")
            print("  python main.py C101.100 100 500 5")
        else:
            instance = sys.argv[1].upper()
            pop = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            gen = int(sys.argv[3]) if len(sys.argv) > 3 else 500
            runs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
            run_optimization(instance, pop, gen, runs)
    else:
        interactive_menu()
