"""Quick test to verify Hypervolume calculation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII
import time

print("=" * 60)
print("K-NSGA-II VERIFICATION TEST")
print("=" * 60)

# Test instances with paper targets
test_cases = [
    ('C101.25', 0.905, 0.156),
    ('C101.100', 0.810, 0.193),
    ('C107.100', 0.815, 0.133),
    ('C206.50', 0.865, 0.199),
    ('RC106.50', 0.802, 0.203),
]

for instance_name, target_hv, target_sp in test_cases:
    print(f"\nTesting {instance_name}...")
    print(f"Paper Targets: Hv={target_hv}, SP={target_sp}")
    
    instance = load_instance(instance_name)
    optimizer = KNSGAII(
        instance=instance,
        population_size=50,
        max_generations=50,  # Very fast test
        crossover_rate=0.7,
        mutation_rate=0.2
    )
    
    start = time.time()
    pareto_front = optimizer.run(verbose=False)
    elapsed = time.time() - start
    
    metrics = optimizer.get_performance_metrics()
    
    print(f"\nResults for {instance_name}:")
    print(f"  Hypervolume: {metrics['hypervolume']:.4f} (target: {target_hv})")
    print(f"  Spacing:     {metrics['spacing']:.4f} (target: {target_sp})")
    print(f"  Pareto Size: {metrics['pareto_size']}")
    print(f"  Time:        {elapsed:.2f}s")
    
    # Check if within reasonable range
    hv_ok = metrics['hypervolume'] >= target_hv * 0.9
    print(f"  Status:      {'✓ PASS' if hv_ok else '✗ BELOW TARGET'}")

print("\n" + "=" * 60)
