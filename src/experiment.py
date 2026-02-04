"""
Experiment Runner for K-NSGA-II Statistical Validation
========================================================

Provides comprehensive experimental framework for:
- Multiple independent runs
- Statistical significance testing
- Performance metric calculation
- Result aggregation and reporting

Based on experimental methodology from published research.
"""

import os
import json
import math
import random
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from .data_parser import load_instance
from .hybrid_knsga2 import KNSGAII


@dataclass
class RunResult:
    """Results from a single algorithm run"""
    run_id: int
    instance: str
    hypervolume: float
    spacing: float
    pareto_size: int
    best_f1: float
    best_f2: float
    decomposition_time: float
    optimization_time: float
    combination_time: float
    total_time: float
    pareto_front: List[Tuple[float, float]]


@dataclass  
class InstanceStatistics:
    """Aggregated statistics for an instance across multiple runs"""
    instance: str
    num_runs: int
    
    # Hypervolume statistics
    hv_mean: float
    hv_std: float
    hv_min: float
    hv_max: float
    hv_median: float
    
    # Spacing statistics
    sp_mean: float
    sp_std: float
    sp_min: float
    sp_max: float
    sp_median: float
    
    # Pareto size statistics
    size_mean: float
    size_std: float
    
    # Time statistics
    time_mean: float
    time_std: float
    
    # Best solutions found
    best_f1_overall: float
    best_f2_overall: float


class StatisticalAnalyzer:
    """Statistical analysis utilities for experimental results"""
    
    @staticmethod
    def mean(values: List[float]) -> float:
        """Calculate arithmetic mean"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    @staticmethod
    def std(values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean_val = StatisticalAnalyzer.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    @staticmethod
    def median(values: List[float]) -> float:
        """Calculate median"""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        return sorted_vals[n//2]
    
    @staticmethod
    def wilcoxon_signed_rank(x: List[float], y: List[float]) -> Tuple[float, float]:
        """
        Wilcoxon signed-rank test for paired samples
        Returns (statistic, approximate p-value)
        """
        if len(x) != len(y) or len(x) < 5:
            return 0.0, 1.0
        
        differences = [xi - yi for xi, yi in zip(x, y)]
        abs_diffs = [(abs(d), i, d > 0) for i, d in enumerate(differences) if d != 0]
        
        if not abs_diffs:
            return 0.0, 1.0
        
        # Rank by absolute difference
        abs_diffs.sort(key=lambda x: x[0])
        
        # Calculate W+ and W-
        w_plus = 0
        w_minus = 0
        
        for rank, (_, _, is_positive) in enumerate(abs_diffs, 1):
            if is_positive:
                w_plus += rank
            else:
                w_minus += rank
        
        w = min(w_plus, w_minus)
        n = len(abs_diffs)
        
        # Approximate p-value using normal approximation
        mean_w = n * (n + 1) / 4
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        
        if std_w == 0:
            return w, 1.0
        
        z = (w - mean_w) / std_w
        # Two-tailed p-value approximation
        p_value = 2 * (1 - StatisticalAnalyzer._norm_cdf(abs(z)))
        
        return w, p_value
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Approximate standard normal CDF"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class ExperimentRunner:
    """
    Comprehensive experiment runner for K-NSGA-II validation
    
    Features:
    - Multiple independent runs per instance
    - Statistical analysis (mean, std, median)
    - Significance testing
    - JSON result export
    - LaTeX table generation
    """
    
    def __init__(
        self,
        instances: List[str],
        params: Dict,
        num_runs: int = 30,
        output_dir: str = "results"
    ):
        """
        Initialize experiment runner
        
        Args:
            instances: List of instance names to test
            params: Algorithm parameters
            num_runs: Number of independent runs per instance
            output_dir: Directory for output files
        """
        self.instances = instances
        self.params = params
        self.num_runs = num_runs
        self.output_dir = output_dir
        
        self.results: Dict[str, List[RunResult]] = {}
        self.statistics: Dict[str, InstanceStatistics] = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, verbose: bool = True) -> Dict[str, InstanceStatistics]:
        """
        Execute the complete experimental study
        
        Args:
            verbose: Print progress information
        
        Returns:
            Dictionary of statistics per instance
        """
        start_time = datetime.now()
        
        if verbose:
            print("\n" + "=" * 70)
            print("K-NSGA-II EXPERIMENTAL STUDY")
            print("=" * 70)
            print(f"Instances: {len(self.instances)}")
            print(f"Runs per instance: {self.num_runs}")
            print(f"Total runs: {len(self.instances) * self.num_runs}")
            print(f"Parameters: pop={self.params.get('population_size', 100)}, "
                  f"gen={self.params.get('max_generations', 1000)}")
            print("=" * 70)
        
        for inst_idx, instance_name in enumerate(self.instances):
            if verbose:
                print(f"\n[{inst_idx+1}/{len(self.instances)}] {instance_name}")
                print("-" * 50)
            
            self.results[instance_name] = []
            
            # Load instance once
            instance = load_instance(instance_name)
            
            for run in range(self.num_runs):
                # Set different seed for each run
                run_seed = (self.params.get('random_state', 0) or 0) + run
                
                run_start = time.time()
                
                # Create and run algorithm
                knsga2 = KNSGAII(
                    instance=instance,
                    population_size=self.params.get('population_size', 100),
                    max_generations=self.params.get('max_generations', 1000),
                    crossover_rate=self.params.get('crossover_rate', 0.9),
                    mutation_rate=self.params.get('mutation_rate', 0.1),
                    random_state=run_seed
                )
                
                pareto_front = knsga2.run(verbose=False)
                metrics = knsga2.get_performance_metrics()
                
                total_time = time.time() - run_start
                
                # Store result
                result = RunResult(
                    run_id=run + 1,
                    instance=instance_name,
                    hypervolume=metrics['hypervolume'],
                    spacing=metrics['spacing'],
                    pareto_size=metrics['pareto_size'],
                    best_f1=metrics['best_f1'],
                    best_f2=metrics['best_f2'],
                    decomposition_time=knsga2.decomposition_time,
                    optimization_time=knsga2.optimization_time,
                    combination_time=knsga2.combination_time,
                    total_time=total_time,
                    pareto_front=[(s.f1, s.f2) for s in pareto_front]
                )
                
                self.results[instance_name].append(result)
                
                if verbose:
                    print(f"  Run {run+1:2d}/{self.num_runs}: "
                          f"Hv={metrics['hypervolume']:.4f}, "
                          f"SP={metrics['spacing']:.4f}, "
                          f"Size={metrics['pareto_size']:2d}, "
                          f"Time={total_time:.2f}s")
            
            # Calculate statistics for this instance
            self.statistics[instance_name] = self._calculate_statistics(instance_name)
            
            if verbose:
                stats = self.statistics[instance_name]
                print(f"\n  Statistics ({self.num_runs} runs):")
                print(f"    Hv:   {stats.hv_mean:.4f} ± {stats.hv_std:.4f} "
                      f"[{stats.hv_min:.4f}, {stats.hv_max:.4f}]")
                print(f"    SP:   {stats.sp_mean:.4f} ± {stats.sp_std:.4f}")
                print(f"    Time: {stats.time_mean:.2f}s ± {stats.time_std:.2f}s")
        
        # Save results
        self._save_results()
        
        if verbose:
            elapsed = datetime.now() - start_time
            print("\n" + "=" * 70)
            print(f"Experiment completed in {elapsed}")
            print(f"Results saved to: {self.output_dir}/")
            print("=" * 70)
        
        return self.statistics
    
    def _calculate_statistics(self, instance_name: str) -> InstanceStatistics:
        """Calculate aggregate statistics for an instance"""
        results = self.results[instance_name]
        
        hvs = [r.hypervolume for r in results]
        sps = [r.spacing for r in results]
        sizes = [r.pareto_size for r in results]
        times = [r.total_time for r in results]
        
        return InstanceStatistics(
            instance=instance_name,
            num_runs=len(results),
            hv_mean=StatisticalAnalyzer.mean(hvs),
            hv_std=StatisticalAnalyzer.std(hvs),
            hv_min=min(hvs),
            hv_max=max(hvs),
            hv_median=StatisticalAnalyzer.median(hvs),
            sp_mean=StatisticalAnalyzer.mean(sps),
            sp_std=StatisticalAnalyzer.std(sps),
            sp_min=min(sps),
            sp_max=max(sps),
            sp_median=StatisticalAnalyzer.median(sps),
            size_mean=StatisticalAnalyzer.mean(sizes),
            size_std=StatisticalAnalyzer.std(sizes),
            time_mean=StatisticalAnalyzer.mean(times),
            time_std=StatisticalAnalyzer.std(times),
            best_f1_overall=min(r.best_f1 for r in results),
            best_f2_overall=min(r.best_f2 for r in results)
        )
    
    def _save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        detailed_results = {
            'experiment_info': {
                'timestamp': timestamp,
                'num_runs': self.num_runs,
                'params': self.params,
                'instances': self.instances
            },
            'statistics': {k: asdict(v) for k, v in self.statistics.items()},
            'detailed_runs': {
                k: [asdict(r) for r in v] 
                for k, v in self.results.items()
            }
        }
        
        json_path = os.path.join(self.output_dir, f'results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save summary CSV
        csv_path = os.path.join(self.output_dir, f'summary_{timestamp}.csv')
        with open(csv_path, 'w') as f:
            f.write("Instance,Hv_mean,Hv_std,SP_mean,SP_std,Size_mean,Time_mean\n")
            for inst, stats in self.statistics.items():
                f.write(f"{inst},{stats.hv_mean:.4f},{stats.hv_std:.4f},"
                       f"{stats.sp_mean:.4f},{stats.sp_std:.4f},"
                       f"{stats.size_mean:.1f},{stats.time_mean:.2f}\n")
        
        # Save LaTeX table
        self._save_latex_table(timestamp)
    
    def _save_latex_table(self, timestamp: str):
        """Generate LaTeX table for publication"""
        latex_path = os.path.join(self.output_dir, f'table_{timestamp}.tex')
        
        with open(latex_path, 'w') as f:
            f.write("% Auto-generated LaTeX table for K-NSGA-II results\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{K-NSGA-II Performance Results}\n")
            f.write("\\label{tab:knsga2_results}\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\hline\n")
            f.write("Instance & $Hv_{avg}$ & $Hv_{std}$ & $SP_{avg}$ & $SP_{std}$ & $|PF|$ & Time(s) \\\\\n")
            f.write("\\hline\n")
            
            for inst, stats in self.statistics.items():
                f.write(f"{inst} & {stats.hv_mean:.3f} & {stats.hv_std:.3f} & "
                       f"{stats.sp_mean:.3f} & {stats.sp_std:.3f} & "
                       f"{stats.size_mean:.0f} & {stats.time_mean:.1f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        report_path = os.path.join(self.output_dir, 'experiment_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("K-NSGA-II EXPERIMENTAL REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Instances: {', '.join(self.instances)}\n")
            f.write(f"Independent runs: {self.num_runs}\n")
            f.write(f"Population size: {self.params.get('population_size', 100)}\n")
            f.write(f"Max generations: {self.params.get('max_generations', 1000)}\n")
            f.write(f"Crossover rate: {self.params.get('crossover_rate', 0.9)}\n")
            f.write(f"Mutation rate: {self.params.get('mutation_rate', 0.1)}\n\n")
            
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Instance':<12} {'Hv_avg':>10} {'Hv_std':>10} "
                   f"{'SP_avg':>10} {'SP_std':>10} {'Time':>10}\n")
            f.write("-" * 70 + "\n")
            
            for inst, stats in self.statistics.items():
                f.write(f"{inst:<12} {stats.hv_mean:>10.4f} {stats.hv_std:>10.4f} "
                       f"{stats.sp_mean:>10.4f} {stats.sp_std:>10.4f} "
                       f"{stats.time_mean:>10.2f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
        
        print(f"\nReport saved to: {report_path}")


# Paper Table 5 target values for comparison
PAPER_TARGETS = {
    'C101.25': {'hv': 0.905, 'sp': 0.156},
    'C101.100': {'hv': 0.81, 'sp': 0.193},
    'C107.100': {'hv': 0.815, 'sp': 0.133},
    'C206.50': {'hv': 0.865, 'sp': 0.199},
    'R109.25': {'hv': 0.799, 'sp': 0.239},
    'RC106.50': {'hv': 0.802, 'sp': 0.203}
}
