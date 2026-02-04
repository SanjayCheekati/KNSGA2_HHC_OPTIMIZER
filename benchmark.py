"""
Benchmark Evaluation Script
============================
Runs statistical analysis across all benchmark instances.
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII

# Benchmark configurations
INSTANCES = ['C101.25', 'C101.100', 'C107.100', 'C206.50', 'RC106.50']
NUM_RUNS = 5
POPULATION = 100
GENERATIONS = 500


def run_benchmark():
    """Execute comprehensive benchmark evaluation."""
    
    print("=" * 80)
    print("K-NSGA-II BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Configuration: population={POPULATION}, generations={GENERATIONS}, runs={NUM_RUNS}")
    print("-" * 80)
    
    all_results = {}
    total_start = time.time()
    
    for instance_name in INSTANCES:
        print(f"\nEvaluating {instance_name}...")
        instance = load_instance(instance_name)
        
        hvs, sps, times, pareto_sizes = [], [], [], []
        
        for run in range(NUM_RUNS):
            start = time.time()
            
            knsga2 = KNSGAII(
                instance=instance,
                population_size=POPULATION,
                max_generations=GENERATIONS,
                crossover_rate=0.7,
                mutation_rate=0.2,
                random_state=run * 7 + 42
            )
            
            knsga2.run(verbose=False)
            metrics = knsga2.get_performance_metrics()
            
            hvs.append(metrics['hypervolume'])
            sps.append(metrics['spacing'])
            pareto_sizes.append(metrics['pareto_size'])
            times.append(time.time() - start)
            
            print(f"  Run {run+1}/{NUM_RUNS}: Hv={metrics['hypervolume']:.4f}, "
                  f"SP={metrics['spacing']:.4f}, Time={times[-1]:.2f}s")
        
        all_results[instance_name] = {
            'avg_hv': sum(hvs) / len(hvs),
            'std_hv': (sum((h - sum(hvs)/len(hvs))**2 for h in hvs) / len(hvs)) ** 0.5,
            'min_hv': min(hvs),
            'max_hv': max(hvs),
            'avg_sp': sum(sps) / len(sps),
            'avg_pareto': sum(pareto_sizes) / len(pareto_sizes),
            'avg_time': sum(times) / len(times)
        }
    
    # Summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Instance':<12} {'Avg Hv':>10} {'Std Hv':>10} {'Min Hv':>10} "
          f"{'Max Hv':>10} {'Avg SP':>10} {'Time':>8}")
    print("-" * 80)
    
    for inst, r in all_results.items():
        print(f"{inst:<12} {r['avg_hv']:>10.4f} {r['std_hv']:>10.4f} "
              f"{r['min_hv']:>10.4f} {r['max_hv']:>10.4f} "
              f"{r['avg_sp']:>10.4f} {r['avg_time']:>7.2f}s")
    
    print("-" * 80)
    print(f"Total evaluation time: {time.time() - total_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
