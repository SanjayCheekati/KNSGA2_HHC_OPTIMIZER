"""
K-NSGA-II: A Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm
================================================================================

Implementation of K-NSGA-II for solving the Home Health Care Multi-Objective
Vehicle Routing Problem with Time Windows (HHC-MOVRPTW).

Reference Paper:
    "A Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm for 
    Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows"

Algorithm Overview:
    Stage 1 - DECOMPOSITION: K-means clustering partitions customers geographically
    Stage 2 - OPTIMIZATION: NSGA-II optimizes each cluster independently  
    Stage 3 - COMBINATION: Pareto fronts are merged into global optimal front

Key Features:
    - Multi-objective optimization (Service Time + Tardiness minimization)
    - Scalable decomposition for large instances
    - Adaptive genetic operators
    - Statistical validation framework
    - Publication-ready visualization

Author: Research Implementation
Version: 2.0.0
License: MIT
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance, list_available_instances
from src.hybrid_knsga2 import KNSGAII
from src.experiment import ExperimentRunner
from src.visualization import ParetoVisualizer


def run_single_instance(instance_name: str, params: dict, verbose: bool = True):
    """Run K-NSGA-II on a single instance"""
    print(f"\n{'='*70}")
    print(f"K-NSGA-II: {instance_name}")
    print(f"{'='*70}")
    
    # Load instance
    instance = load_instance(instance_name)
    print(f"Loaded: {instance.num_customers} customers, {instance.num_vehicles} vehicles")
    
    # Run algorithm
    knsga2 = KNSGAII(
        instance=instance,
        population_size=params.get('population_size', 100),
        max_generations=params.get('max_generations', 1000),
        crossover_rate=params.get('crossover_rate', 0.9),
        mutation_rate=params.get('mutation_rate', 0.1),
        random_state=params.get('random_state', None)
    )
    
    pareto_front = knsga2.run(verbose=verbose)
    metrics = knsga2.get_performance_metrics()
    
    return {
        'instance': instance_name,
        'pareto_front': [(s.f1, s.f2) for s in pareto_front],
        'metrics': metrics,
        'timing': {
            'decomposition': knsga2.decomposition_time,
            'optimization': knsga2.optimization_time,
            'combination': knsga2.combination_time
        }
    }


def run_experiment(instances: list, params: dict, num_runs: int = 30):
    """Run full experimental study with statistical analysis"""
    runner = ExperimentRunner(
        instances=instances,
        params=params,
        num_runs=num_runs
    )
    
    results = runner.run()
    runner.generate_report()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='K-NSGA-II for HHC-MOVRPTW',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --instance C101.25              # Run single instance
  python main.py --instance C101.25 --runs 30    # Run with 30 repetitions
  python main.py --experiment                    # Run full Paper Table 5 experiment
  python main.py --list                          # List available instances
        """
    )
    
    parser.add_argument('--instance', type=str, help='Instance name (e.g., C101.25)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs (default: 1)')
    parser.add_argument('--population', type=int, default=100, help='Population size')
    parser.add_argument('--generations', type=int, default=1000, help='Max generations')
    parser.add_argument('--experiment', action='store_true', help='Run full experiment')
    parser.add_argument('--list', action='store_true', help='List available instances')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # List instances
    if args.list:
        print("\nAvailable Instances:")
        print("-" * 40)
        for inst in list_available_instances():
            print(f"  {inst}")
        return
    
    # Parameters
    params = {
        'population_size': args.population,
        'max_generations': args.generations,
        'crossover_rate': 0.9,
        'mutation_rate': 0.1,
        'random_state': args.seed
    }
    
    # Full experiment mode
    if args.experiment:
        print("\n" + "=" * 70)
        print("K-NSGA-II EXPERIMENTAL STUDY")
        print("Paper Table 5 Benchmark Instances")
        print("=" * 70)
        
        instances = ['C101.25', 'C107.25', 'C206.25', 'R109.25', 'RC106.25']
        run_experiment(instances, params, num_runs=args.runs if args.runs > 1 else 30)
        return
    
    # Single instance mode
    if args.instance:
        if args.runs > 1:
            # Multiple runs with statistics
            all_results = []
            for run in range(args.runs):
                params['random_state'] = args.seed + run if args.seed else run
                result = run_single_instance(
                    args.instance, 
                    params, 
                    verbose=not args.quiet and run == 0
                )
                all_results.append(result)
                if not args.quiet:
                    print(f"Run {run+1}/{args.runs}: Hv={result['metrics']['hypervolume']:.4f}")
            
            # Calculate statistics
            hvs = [r['metrics']['hypervolume'] for r in all_results]
            sps = [r['metrics']['spacing'] for r in all_results]
            
            print(f"\n{'='*70}")
            print(f"STATISTICAL RESULTS ({args.runs} runs)")
            print(f"{'='*70}")
            print(f"Hypervolume: {sum(hvs)/len(hvs):.4f} ± {(max(hvs)-min(hvs))/2:.4f}")
            print(f"Spacing:     {sum(sps)/len(sps):.4f} ± {(max(sps)-min(sps))/2:.4f}")
        else:
            run_single_instance(args.instance, params, verbose=not args.quiet)
        return
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("K-NSGA-II: Home Health Care Multi-Objective VRP with Time Windows")
    print("=" * 70)
    print("\nUsage: python main.py --help for options")
    print("\nQuick Start:")
    print("  python main.py --instance C101.25")
    print("  python main.py --experiment")


if __name__ == "__main__":
    main()
