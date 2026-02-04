"""
Main Execution Script for K-NSGA-II
Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance, ProblemInstance
from src.problem import HHCProblem
from src.hybrid_knsga2 import KNSGAII, KNSGAIIResult


def run_experiment(instance_name: str,
                   population_size: int = 100,
                   max_generations: int = 1000,
                   crossover_rate: float = 0.7,
                   mutation_rate: float = 0.2,
                   n_runs: int = 1,
                   verbose: bool = True) -> Dict:
    """
    Run K-NSGA-II experiment on a single instance.
    
    Args:
        instance_name: Instance name (e.g., "C101.25")
        population_size: Population size
        max_generations: Maximum generations
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        n_runs: Number of runs (for statistical analysis)
        verbose: Print detailed output
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {instance_name}")
    print(f"{'='*60}")
    
    # Load instance
    try:
        instance = load_instance(instance_name)
        print(f"Instance loaded: {instance.num_customers} customers, {instance.num_vehicles} caregivers")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    all_results = []
    all_metrics = []
    
    for run in range(n_runs):
        if n_runs > 1:
            print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # Create K-NSGA-II instance
        knsga2 = KNSGAII(
            instance=instance,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            use_time_features=True,
            balance_clusters=True,
            random_state=42 + run  # Different seed per run
        )
        
        # Run algorithm
        result = knsga2.run(verbose=verbose and n_runs == 1)
        all_results.append(result)
        
        # Calculate metrics
        metrics = knsga2.get_performance_metrics()
        all_metrics.append(metrics)
        
        if n_runs > 1 and metrics:
            print(f"  Pareto size: {metrics['pareto_size']}, "
                  f"Best F1: {metrics['best_f1']:.2f}, Best F2: {metrics['best_f2']:.2f}")
    
    # Aggregate results
    if all_metrics and all_metrics[0]:
        avg_metrics = {
            'instance': instance_name,
            'pareto_size': np.mean([m['pareto_size'] for m in all_metrics]),
            'pareto_size_std': np.std([m['pareto_size'] for m in all_metrics]),
            'best_f1': np.mean([m['best_f1'] for m in all_metrics]),
            'best_f1_std': np.std([m['best_f1'] for m in all_metrics]),
            'best_f2': np.mean([m['best_f2'] for m in all_metrics]),
            'best_f2_std': np.std([m['best_f2'] for m in all_metrics]),
            'spacing': np.mean([m['spacing'] for m in all_metrics]),
            'hypervolume': np.mean([m['hypervolume'] for m in all_metrics]),
            'avg_time': np.mean([r.total_time for r in all_results]),
        }
    else:
        avg_metrics = {}
    
    return {
        'results': all_results,
        'metrics': all_metrics,
        'avg_metrics': avg_metrics
    }


def run_table5_experiments(population_size: int = 100,
                           max_generations: int = 1000,
                           n_runs: int = 10,
                           verbose: bool = False):
    """
    Run experiments replicating Table 5 from the paper.
    
    Instances: C101.25, C101.100, C107.100, C206.50, R109.25, RC106.50
    """
    print("\n" + "=" * 80)
    print("TABLE 5 REPLICATION EXPERIMENT")
    print("K-NSGA-II: Hybrid K-means + NSGA-II Algorithm")
    print("=" * 80)
    print(f"\nExperiment Settings:")
    print(f"  Population Size: {population_size}")
    print(f"  Max Generations: {max_generations}")
    print(f"  Number of Runs: {n_runs}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Table 5 instances from the paper
    instances = [
        "C101.25",    # 25 customers, clustered
        "C101.100",   # 100 customers, clustered
        "C107.100",   # 100 customers, clustered
        "C206.50",    # 50 customers, clustered
        "R109.25",    # 25 customers, random
        "RC106.50"    # 50 customers, mixed
    ]
    
    all_experiments = {}
    
    for instance_name in instances:
        result = run_experiment(
            instance_name=instance_name,
            population_size=population_size,
            max_generations=max_generations,
            n_runs=n_runs,
            verbose=verbose
        )
        
        if result:
            all_experiments[instance_name] = result
    
    # Print summary table
    print_results_table(all_experiments)
    
    return all_experiments


def print_results_table(experiments: Dict):
    """Print results in a formatted table similar to Table 5"""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (K-NSGA-II)")
    print("=" * 80)
    
    # Header
    print(f"\n{'Instance':<12} | {'Pareto':<8} | {'Best F1':<15} | {'Best F2':<15} | {'Time(s)':<10}")
    print("-" * 80)
    
    for instance, data in experiments.items():
        metrics = data.get('avg_metrics', {})
        if metrics:
            pareto = f"{metrics['pareto_size']:.1f}"
            f1 = f"{metrics['best_f1']:.2f} ± {metrics['best_f1_std']:.2f}"
            f2 = f"{metrics['best_f2']:.2f} ± {metrics['best_f2_std']:.2f}"
            time_s = f"{metrics['avg_time']:.2f}"
            print(f"{instance:<12} | {pareto:<8} | {f1:<15} | {f2:<15} | {time_s:<10}")
    
    print("-" * 80)


def quick_test():
    """Quick test with small instance and few generations"""
    print("\n" + "=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)
    
    instance_name = "C101.25"
    
    result = run_experiment(
        instance_name=instance_name,
        population_size=30,
        max_generations=50,
        n_runs=1,
        verbose=True
    )
    
    if result and result['avg_metrics']:
        print("\nQuick Test Results:")
        for key, value in result['avg_metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def main():
    """Main entry point with command-line arguments"""
    parser = argparse.ArgumentParser(
        description="K-NSGA-II for Home Health Care Routing and Scheduling"
    )
    
    parser.add_argument('--instance', '-i', type=str, default=None,
                        help='Instance name (e.g., C101.25)')
    parser.add_argument('--population', '-p', type=int, default=100,
                        help='Population size')
    parser.add_argument('--generations', '-g', type=int, default=1000,
                        help='Maximum generations')
    parser.add_argument('--runs', '-r', type=int, default=1,
                        help='Number of runs')
    parser.add_argument('--table5', action='store_true',
                        help='Run Table 5 replication experiment')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    elif args.table5:
        run_table5_experiments(
            population_size=args.population,
            max_generations=args.generations,
            n_runs=args.runs,
            verbose=args.verbose
        )
    elif args.instance:
        run_experiment(
            instance_name=args.instance,
            population_size=args.population,
            max_generations=args.generations,
            n_runs=args.runs,
            verbose=True
        )
    else:
        # Default: run quick test
        print("No arguments provided. Running quick test...")
        print("Use --help for command-line options")
        quick_test()


if __name__ == "__main__":
    main()
