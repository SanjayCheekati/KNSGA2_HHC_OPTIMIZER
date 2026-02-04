"""Minimal verification test for K-NSGA-II"""
import sys, os
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress output
import io
from contextlib import redirect_stdout

from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII

test_cases = [
    ('C101.25', 0.905),
    ('C101.100', 0.810),
    ('C107.100', 0.815),
    ('C206.50', 0.865),
    ('RC106.50', 0.802),
]

print('=' * 50)
print('K-NSGA-II VERIFICATION RESULTS')
print('=' * 50)
print(f"{'Instance':12} | {'Target':6} | {'Actual':7} | Status")
print('-' * 50)

all_pass = True
for name, target in test_cases:
    try:
        inst = load_instance(name)
        opt = KNSGAII(
            inst, 
            population_size=30, 
            max_generations=30,
            crossover_rate=0.7, 
            mutation_rate=0.2
        )
        # Suppress algorithm output
        f = io.StringIO()
        with redirect_stdout(f):
            opt.run(verbose=False)
        
        m = opt.get_performance_metrics()
        hv = m['hypervolume']
        passed = hv >= target * 0.9
        status = 'PASS' if passed else 'FAIL'
        if not passed:
            all_pass = False
        print(f"{name:12} | {target:.3f}  | {hv:.4f}  | {status}")
    except Exception as e:
        print(f"{name:12} | {target:.3f}  | ERROR   | {str(e)[:20]}")
        all_pass = False

print('-' * 50)
print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
print('=' * 50)
