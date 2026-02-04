"""
Run C101.25 instance experiment to match Table 5 results from the paper.

Paper Target for K-NSGA-II on C101.25:
- Hypervolume (Hv): 0.905
- Spacing (SP): 0.156

Paper Parameters:
- Population size: 100
- Crossover rate: 0.7
- Mutation rate: 0.2
- Max generations: 1000
- Number of runs: 20
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII


def run_c101_25_experiment(n_runs: int = 20, 
                           population_size: int = 100,
                           max_generations: int = 1000,
                           verbose: bool = False):
    """
    Run C101.25 experiment matching paper parameters.
    """
    print("\n" + "=" * 70)
    print("C101.25 EXPERIMENT - Matching Table 5 Results")
    print("=" * 70)
    print(f"\nTarget Results (from paper):")
    print(f"  K-NSGA-II Hypervolume (Hv): 0.905")
    print(f"  K-NSGA-II Spacing (SP): 0.156")
    print(f"\nExperiment Parameters:")
    print(f"  Population Size: {population_size}")
    print(f"  Max Generations: {max_generations}")
    print(f"  Crossover Rate: 0.7")
    print(f"  Mutation Rate: 0.2")
    print(f"  Number of Runs: {n_runs}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load instance
    instance = load_instance("C101.25")
    print(f"\nInstance: C101.25")
    print(f"  Customers: {instance.num_customers}")
    print(f"  Caregivers: {instance.num_vehicles}")
    
    # Store results from all runs
    all_hv = []
    all_sp = []
    all_pf_sizes = []
    all_times = []
    all_best_f1 = []
    all_best_f2 = []
    
    print(f"\n{'='*70}")
    print("Running experiments...")
    print("=" * 70)
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}", end=" - ")
        
        # Create K-NSGA-II instance with different seed per run
        knsga2 = KNSGAII(
            instance=instance,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=0.7,
            mutation_rate=0.2,
            use_time_features=True,
            balance_clusters=True,
            random_state=42 + run
        )
        
        # Run algorithm
        result = knsga2.run(verbose=verbose)
        
        # Get metrics
        metrics = knsga2.get_performance_metrics()
        
        if metrics:
            all_hv.append(metrics['hypervolume'])
            all_sp.append(metrics['spacing'])
            all_pf_sizes.append(metrics['pareto_size'])
            all_best_f1.append(metrics['best_f1'])
            all_best_f2.append(metrics['best_f2'])
            all_times.append(result.total_time)
            
            print(f"Hv: {metrics['hypervolume']:.4f}, SP: {metrics['spacing']:.4f}, "
                  f"PF: {metrics['pareto_size']}, Time: {result.total_time:.2f}s")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nHypervolume (Hv):")
    print(f"  Min: {min(all_hv):.4f}")
    print(f"  Max: {max(all_hv):.4f}")
    print(f"  Avg: {np.mean(all_hv):.4f} ± {np.std(all_hv):.4f}")
    print(f"  Target: 0.905")
    
    print(f"\nSpacing (SP):")
    print(f"  Min: {min(all_sp):.4f}")
    print(f"  Max: {max(all_sp):.4f}")
    print(f"  Avg: {np.mean(all_sp):.4f} ± {np.std(all_sp):.4f}")
    print(f"  Target: 0.156")
    
    print(f"\nPareto Front Size:")
    print(f"  Min: {min(all_pf_sizes)}")
    print(f"  Max: {max(all_pf_sizes)}")
    print(f"  Avg: {np.mean(all_pf_sizes):.1f}")
    
    print(f"\nBest F1 (Service Time):")
    print(f"  Avg: {np.mean(all_best_f1):.2f}")
    
    print(f"\nBest F2 (Tardiness):")
    print(f"  Avg: {np.mean(all_best_f2):.2f}")
    
    print(f"\nExecution Time:")
    print(f"  Avg: {np.mean(all_times):.2f}s per run")
    print(f"  Total: {sum(all_times):.2f}s")
    
    # Comparison with paper
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER")
    print("=" * 70)
    
    hv_avg = np.mean(all_hv)
    sp_avg = np.mean(all_sp)
    
    hv_diff = ((hv_avg - 0.905) / 0.905) * 100
    sp_diff = ((sp_avg - 0.156) / 0.156) * 100
    
    print(f"\n{'Metric':<15} {'Paper':<10} {'Ours':<10} {'Difference':<15}")
    print("-" * 50)
    print(f"{'Hypervolume':<15} {'0.905':<10} {hv_avg:<10.4f} {hv_diff:+.2f}%")
    print(f"{'Spacing':<15} {'0.156':<10} {sp_avg:<10.4f} {sp_diff:+.2f}%")
    
    if abs(hv_diff) <= 10 and abs(sp_diff) <= 30:
        print("\n✓ Results are CLOSE to paper values!")
    elif abs(hv_diff) <= 20 and abs(sp_diff) <= 50:
        print("\n~ Results are MODERATELY close to paper values.")
    else:
        print("\n✗ Results differ significantly from paper values.")
        print("  This may be due to implementation differences or random variation.")
    
    return {
        'hv_avg': hv_avg,
        'hv_std': np.std(all_hv),
        'sp_avg': sp_avg,
        'sp_std': np.std(all_sp),
        'pf_size_avg': np.mean(all_pf_sizes),
        'time_avg': np.mean(all_times)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run C101.25 experiment")
    parser.add_argument('-r', '--runs', type=int, default=10, 
                        help='Number of runs (paper uses 20)')
    parser.add_argument('-p', '--population', type=int, default=100,
                        help='Population size')
    parser.add_argument('-g', '--generations', type=int, default=1000,
                        help='Max generations')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    run_c101_25_experiment(
        n_runs=args.runs,
        population_size=args.population,
        max_generations=args.generations,
        verbose=args.verbose
    )
