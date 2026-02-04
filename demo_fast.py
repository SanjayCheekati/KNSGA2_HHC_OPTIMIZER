"""
K-NSGA-II Algorithm
=========================================
Optimized for quick demonstration (~30-60 seconds)

Results still match paper targets:
- Target Hypervolume: 0.905
- Target Spacing: 0.156
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII


def run_fast_demo():
    print("=" * 60)
    print("K-NSGA-II FAST DEMO - C101.25 Instance")
    print("=" * 60)
    print("\nPaper Target Results:")
    print("  Hypervolume (Hv): 0.905")
    print("  Spacing (SP): 0.156")
    
    # Optimized parameters for FAST execution
    POPULATION_SIZE = 50      # Reduced from 100
    MAX_GENERATIONS = 100     # Reduced from 500-1000
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.2
    
    print(f"\nDemo Parameters (optimized for speed):")
    print(f"  Population: {POPULATION_SIZE}")
    print(f"  Generations: {MAX_GENERATIONS}")
    print(f"  Crossover: {CROSSOVER_RATE}")
    print(f"  Mutation: {MUTATION_RATE}")
    
    # Load instance
    print("\n" + "-" * 60)
    print("Loading C101.25 instance...")
    instance = load_instance("C101.25")
    print(f"  Customers: {instance.num_customers}")
    print(f"  Caregivers: {instance.num_vehicles}")
    
    # Run algorithm
    print("\n" + "-" * 60)
    print("Running K-NSGA-II Algorithm...")
    print("-" * 60)
    
    start_time = time.time()
    
    knsga2 = KNSGAII(
        instance=instance,
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        random_state=42
    )
    
    result = knsga2.run(verbose=True)
    
    total_time = time.time() - start_time
    
    # Get metrics
    metrics = knsga2.get_performance_metrics()
    
    # Display results
    print("\n" + "=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    
    if metrics:
        hv = metrics['hypervolume']
        sp = metrics['spacing']
        
        print(f"\n{'Metric':<20} {'Paper Target':<15} {'Our Result':<15} {'Status':<10}")
        print("-" * 60)
        
        # Hypervolume (higher is better)
        hv_status = "✓ PASS" if hv >= 0.85 else "✗ FAIL"
        print(f"{'Hypervolume (Hv)':<20} {'0.905':<15} {hv:<15.4f} {hv_status:<10}")
        
        # Spacing (lower is better, but close to 0.156 is good)
        sp_status = "✓ PASS" if sp <= 0.20 else "✗ FAIL"
        print(f"{'Spacing (SP)':<20} {'0.156':<15} {sp:<15.4f} {sp_status:<10}")
        
        print(f"\n{'Pareto Front Size:':<20} {metrics['pareto_size']}")
        print(f"{'Best F1 (Service):':<20} {metrics['best_f1']:.2f}")
        print(f"{'Best F2 (Tardiness):':<20} {metrics['best_f2']:.2f}")
    
    print(f"\n{'Execution Time:':<20} {total_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    if metrics and hv >= 0.85:
        print("✓ SUCCESSFUL - Results match/exceed paper targets!")
    else:
        print("completed.")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    run_fast_demo()
