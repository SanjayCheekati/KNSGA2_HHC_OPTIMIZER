"""
Algorithm Comparison Utilities
==============================

Runs side-by-side experiments for NSGA-II and K-NSGA-II on the same
benchmark instances, then exports summary artifacts and boxplot visuals.
"""

import csv
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

try:
    from .data_parser import load_instance
    from .hybrid_knsga2 import KNSGAII
    from .nsga2 import NSGA2
except ImportError:  # Allows running as a script from the src directory.
    from data_parser import load_instance
    from hybrid_knsga2 import KNSGAII
    from nsga2 import NSGA2


BENCHMARK_INSTANCES = [
    "C101.25",
    "C101.100",
    "C107.100",
    "C206.50",
    "R109.25",
    "RC106.50",
]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _unique_front_by_objectives(pareto_front: List, decimals: int = 6) -> List:
    """Drop duplicate objective points to avoid degenerate spacing values."""
    unique = []
    seen = set()

    for sol in pareto_front:
        key = (round(sol.f1, decimals), round(sol.f2, decimals))
        if key not in seen:
            seen.add(key)
            unique.append(sol)

    return unique


def _compute_pareto_metrics(pareto_front: List, reference_pool: List = None) -> Dict[str, float]:
    """
    Compute hypervolume and spacing using the same methodology as KNSGA-II.

    Args:
        pareto_front: Non-dominated solutions.
        reference_pool: Population used for normalization bounds.
    """
    # De-duplicate objective pairs first so spacing is meaningful.
    pareto_front = _unique_front_by_objectives(pareto_front)

    if not pareto_front:
        return {
            "hypervolume": 0.0,
            "spacing": 0.0,
            "pareto_size": 0,
            "best_f1": float("inf"),
            "best_f2": float("inf"),
        }

    f1_values = [s.f1 for s in pareto_front]
    f2_values = [s.f2 for s in pareto_front]

    if reference_pool:
        all_f1 = [s.f1 for s in reference_pool]
        all_f2 = [s.f2 for s in reference_pool]
    else:
        all_f1 = f1_values
        all_f2 = f2_values

    f1_min_pop, f1_max_pop = min(all_f1), max(all_f1)
    f2_min_pop, f2_max_pop = min(all_f2), max(all_f2)

    f1_range = f1_max_pop - f1_min_pop
    f2_range = f2_max_pop - f2_min_pop

    if f1_range < 1e-10:
        f1_range = max(f1_max_pop * 0.01, 1.0)
    if f2_range < 1e-10:
        f2_range = max(f2_max_pop * 0.01, 1.0)

    normalized_points = []
    for f1, f2 in zip(f1_values, f2_values):
        norm_f1 = (f1 - f1_min_pop) / f1_range
        norm_f2 = (f2 - f2_min_pop) / f2_range
        normalized_points.append((norm_f1, norm_f2))

    normalized_points.sort(key=lambda p: p[0])

    ref_f1 = 1.1
    ref_f2 = 1.1

    hypervolume = 0.0
    n = len(normalized_points)
    for i in range(n):
        x_i, y_i = normalized_points[i]
        x_next = normalized_points[i + 1][0] if i < n - 1 else ref_f1
        width = x_next - x_i
        height = ref_f2 - y_i
        if width > 0 and height > 0:
            hypervolume += width * height

    hypervolume = hypervolume / (ref_f1 * ref_f2)

    spacing = 0.0
    if len(pareto_front) > 1:
        pf_f1_min, pf_f1_max = min(f1_values), max(f1_values)
        pf_f2_min, pf_f2_max = min(f2_values), max(f2_values)
        pf_f1_range = pf_f1_max - pf_f1_min if pf_f1_max > pf_f1_min else 1.0
        pf_f2_range = pf_f2_max - pf_f2_min if pf_f2_max > pf_f2_min else 1.0

        ds = []
        for i in range(len(f1_values)):
            d1 = min(abs(f1_values[i] - f1_values[j]) for j in range(len(f1_values)) if j != i)
            d2 = min(abs(f2_values[i] - f2_values[j]) for j in range(len(f2_values)) if j != i)
            ds.append(d1 / pf_f1_range + d2 / pf_f2_range)

        if ds:
            mean_d = _mean(ds)
            spacing = math.sqrt(sum((d - mean_d) ** 2 for d in ds) / len(ds))

    return {
        "hypervolume": hypervolume,
        "spacing": spacing,
        "pareto_size": len(pareto_front),
        "best_f1": min(f1_values),
        "best_f2": min(f2_values),
    }


def _summarize_runs(runs: List[Dict]) -> Dict[str, float]:
    hv = [r["hypervolume"] for r in runs]
    sp = [r["spacing"] for r in runs]
    sizes = [r["pareto_size"] for r in runs]
    times = [r["runtime"] for r in runs]

    return {
        "hv_mean": _mean(hv),
        "hv_std": _std(hv),
        "sp_mean": _mean(sp),
        "sp_std": _std(sp),
        "size_mean": _mean(sizes),
        "size_std": _std(sizes),
        "time_mean": _mean(times),
        "time_std": _std(times),
    }


def _save_summary_csv(summary: Dict, filepath: str, instances: List[str]) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance",
            "Algorithm",
            "Hv_mean",
            "Hv_std",
            "SP_mean",
            "SP_std",
            "Pareto_size_mean",
            "Time_mean",
        ])

        for instance_name in instances:
            for algorithm in ["NSGA-II", "K-NSGA-II"]:
                stats = summary[instance_name][algorithm]
                writer.writerow([
                    instance_name,
                    algorithm,
                    f"{stats['hv_mean']:.6f}",
                    f"{stats['hv_std']:.6f}",
                    f"{stats['sp_mean']:.6f}",
                    f"{stats['sp_std']:.6f}",
                    f"{stats['size_mean']:.3f}",
                    f"{stats['time_mean']:.4f}",
                ])


def _save_runs_csv(detailed: Dict, filepath: str, instances: List[str]) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance",
            "Algorithm",
            "Run",
            "Hypervolume",
            "Spacing",
            "Pareto_size",
            "Best_F1",
            "Best_F2",
            "Runtime",
        ])

        for instance_name in instances:
            for algorithm in ["NSGA-II", "K-NSGA-II"]:
                for run in detailed[instance_name][algorithm]:
                    writer.writerow([
                        instance_name,
                        algorithm,
                        run["run_id"],
                        f"{run['hypervolume']:.6f}",
                        f"{run['spacing']:.6f}",
                        run["pareto_size"],
                        f"{run['best_f1']:.6f}",
                        f"{run['best_f2']:.6f}",
                        f"{run['runtime']:.4f}",
                    ])


def _save_algorithm_only_csv(
    detailed: Dict,
    summary: Dict,
    instances: List[str],
    algorithm: str,
    runs_path: str,
    summary_path: str,
) -> None:
    with open(runs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance",
            "Run",
            "Hypervolume",
            "Spacing",
            "Pareto_size",
            "Best_F1",
            "Best_F2",
            "Runtime",
        ])

        for instance_name in instances:
            for run in detailed[instance_name][algorithm]:
                writer.writerow([
                    instance_name,
                    run["run_id"],
                    f"{run['hypervolume']:.6f}",
                    f"{run['spacing']:.6f}",
                    run["pareto_size"],
                    f"{run['best_f1']:.6f}",
                    f"{run['best_f2']:.6f}",
                    f"{run['runtime']:.4f}",
                ])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance",
            "Hv_mean",
            "Hv_std",
            "SP_mean",
            "SP_std",
            "Pareto_size_mean",
            "Time_mean",
        ])

        for instance_name in instances:
            stats = summary[instance_name][algorithm]
            writer.writerow([
                instance_name,
                f"{stats['hv_mean']:.6f}",
                f"{stats['hv_std']:.6f}",
                f"{stats['sp_mean']:.6f}",
                f"{stats['sp_std']:.6f}",
                f"{stats['size_mean']:.3f}",
                f"{stats['time_mean']:.4f}",
            ])


def _boxplot_metric_per_instance(
    detailed: Dict,
    instances: List[str],
    metric_key: str,
    y_label: str,
    output_dir: str,
    timestamp: str,
) -> Dict[str, str]:
    output_paths: Dict[str, str] = {}

    for instance_name in instances:
        nsga_values = [r[metric_key] for r in detailed[instance_name]["NSGA-II"]]
        knsga_values = [r[metric_key] for r in detailed[instance_name]["K-NSGA-II"]]

        fig, ax = plt.subplots(figsize=(7, 5))
        boxes = ax.boxplot(
            [nsga_values, knsga_values],
            labels=["NSGA-II", "K-NSGA-II"],
            patch_artist=True,
            showfliers=True,
        )

        if len(boxes["boxes"]) >= 2:
            boxes["boxes"][0].set_facecolor("#1f77b4")
            boxes["boxes"][0].set_alpha(0.7)
            boxes["boxes"][1].set_facecolor("#ff7f0e")
            boxes["boxes"][1].set_alpha(0.7)

        ax.set_title(f"{instance_name}: {y_label} (NSGA-II vs K-NSGA-II)")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        safe_instance_name = instance_name.replace(".", "_")
        output_path = os.path.join(
            output_dir,
            f"boxplot_{metric_key}_{safe_instance_name}_{timestamp}.png",
        )
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        output_paths[instance_name] = output_path

    return output_paths


def _boxplot_metric_all_instances(
    detailed: Dict,
    instances: List[str],
    metric_key: str,
    y_label: str,
    output_dir: str,
    timestamp: str,
) -> str:
    """Create one comparative boxplot across all instances."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    positions = []
    labels = []
    colors = []

    pos = 1
    for instance_name in instances:
        nsga_values = [r[metric_key] for r in detailed[instance_name]["NSGA-II"]]
        knsga_values = [r[metric_key] for r in detailed[instance_name]["K-NSGA-II"]]

        data.extend([nsga_values, knsga_values])
        positions.extend([pos, pos + 0.35])
        labels.append(instance_name)
        colors.extend(["#1f77b4", "#ff7f0e"])
        pos += 1.1

    boxes = ax.boxplot(
        data,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=True,
    )

    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    tick_positions = [i * 1.1 + 1.175 for i in range(len(instances))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Instance")
    ax.set_title(f"{y_label} Distribution by Instance")
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", alpha=0.7, label="NSGA-II"),
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", alpha=0.7, label="K-NSGA-II"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    output_path = os.path.join(output_dir, f"boxplot_{metric_key}_{timestamp}.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def _boxplot_hypervolume_dataset_grid(
    detailed: Dict,
    summary: Dict,
    instances: List[str],
    output_dir: str,
    timestamp: str,
) -> str:
    """Create a 2x3 dataset-wise grid of hypervolume boxplots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    # Determine a common y-range for better visual comparison.
    all_hv_values = []
    for instance_name in instances:
        all_hv_values.extend([r["hypervolume"] for r in detailed[instance_name]["NSGA-II"]])
        all_hv_values.extend([r["hypervolume"] for r in detailed[instance_name]["K-NSGA-II"]])

    y_min = max(0.0, min(all_hv_values) - 0.02) if all_hv_values else 0.0
    y_max = min(1.05, max(all_hv_values) + 0.02) if all_hv_values else 1.0

    for idx, instance_name in enumerate(instances):
        ax = axes[idx]
        nsga_values = [r["hypervolume"] for r in detailed[instance_name]["NSGA-II"]]
        knsga_values = [r["hypervolume"] for r in detailed[instance_name]["K-NSGA-II"]]

        boxes = ax.boxplot(
            [nsga_values, knsga_values],
            labels=["NSGA-II", "K-NSGA-II"],
            patch_artist=True,
            showfliers=True,
        )

        if len(boxes["boxes"]) >= 2:
            boxes["boxes"][0].set_facecolor("#1f77b4")
            boxes["boxes"][0].set_alpha(0.55)
            boxes["boxes"][1].set_facecolor("#ff7f0e")
            boxes["boxes"][1].set_alpha(0.55)

        ax.set_title(instance_name, fontsize=12)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.3)

        # Show mean values in each subplot for report readability.
        nsga_mean = summary[instance_name]["NSGA-II"]["hv_mean"]
        knsga_mean = summary[instance_name]["K-NSGA-II"]["hv_mean"]
        ax.text(0.5, y_max - 0.03, f"N={nsga_mean:.3f}", ha="center", va="top", fontsize=9)
        ax.text(1.5, y_max - 0.03, f"K={knsga_mean:.3f}", ha="center", va="top", fontsize=9)

    # Hide any unused axes if instances are fewer than 6.
    for j in range(len(instances), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Dataset-wise Box-Plot Comparison (Hypervolume)", fontsize=16, y=0.98)
    fig.text(0.04, 0.5, "Hypervolume", va="center", rotation="vertical", fontsize=12)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", alpha=0.55, label="NSGA-II"),
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", alpha=0.55, label="K-NSGA-II"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0.05, 0.03, 0.98, 0.95])
    output_path = os.path.join(output_dir, f"boxplot_hypervolume_grid_{timestamp}.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def _save_instance_delta_csv(summary: Dict, filepath: str, instances: List[str]) -> None:
    """Save one-row-per-instance comparison deltas for report writing."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance",
            "Hv_NSGAII",
            "Hv_KNSGAII",
            "Hv_Delta",
            "Hv_Delta_Percent",
            "SP_NSGAII",
            "SP_KNSGAII",
            "SP_Delta",
            "Time_NSGAII_s",
            "Time_KNSGAII_s",
            "Time_Ratio",
            "Hv_Winner",
        ])

        for instance_name in instances:
            nsga = summary[instance_name]["NSGA-II"]
            knsga = summary[instance_name]["K-NSGA-II"]

            hv_delta = knsga["hv_mean"] - nsga["hv_mean"]
            hv_delta_pct = (hv_delta / nsga["hv_mean"] * 100.0) if nsga["hv_mean"] else 0.0
            sp_delta = knsga["sp_mean"] - nsga["sp_mean"]
            time_ratio = (knsga["time_mean"] / nsga["time_mean"]) if nsga["time_mean"] else 0.0

            writer.writerow([
                instance_name,
                f"{nsga['hv_mean']:.6f}",
                f"{knsga['hv_mean']:.6f}",
                f"{hv_delta:.6f}",
                f"{hv_delta_pct:.2f}",
                f"{nsga['sp_mean']:.6f}",
                f"{knsga['sp_mean']:.6f}",
                f"{sp_delta:.6f}",
                f"{nsga['time_mean']:.4f}",
                f"{knsga['time_mean']:.4f}",
                f"{time_ratio:.2f}",
                "K-NSGA-II" if hv_delta > 0 else ("NSGA-II" if hv_delta < 0 else "Tie"),
            ])


def run_algorithm_comparison(
    instances: List[str] = None,
    population_size: int = 100,
    max_generations: int = 500,
    num_runs: int = 5,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict:
    """
    Run NSGA-II and K-NSGA-II side by side for each benchmark instance.

    Returns a dictionary with output paths and computed summaries.
    """
    if instances is None:
        instances = BENCHMARK_INSTANCES

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_results = {
        instance_name: {
            "NSGA-II": [],
            "K-NSGA-II": [],
        }
        for instance_name in instances
    }

    if verbose:
        print("\n" + "=" * 72)
        print("  NSGA-II vs K-NSGA-II COMPARISON")
        print("=" * 72)
        print(
            f"  Parameters: population={population_size}, generations={max_generations}, runs={num_runs}"
        )
        print(f"  Instances: {', '.join(instances)}")
        print("-" * 72)

    for idx, instance_name in enumerate(instances, 1):
        instance = load_instance(instance_name)

        if verbose:
            print(f"\n[{idx}/{len(instances)}] {instance_name}")

        for run_id in range(1, num_runs + 1):
            seed = 42 + run_id * 7

            # Baseline NSGA-II on the full instance
            nsga_start = time.time()
            nsga = NSGA2(
                instance=instance,
                population_size=population_size,
                max_generations=max_generations,
                crossover_rate=0.7,
                mutation_rate=0.2,
                random_state=seed,
            )
            nsga_front = nsga.run(verbose=False)
            nsga_runtime = time.time() - nsga_start
            nsga_metrics = _compute_pareto_metrics(nsga_front, reference_pool=nsga.population)
            nsga_metrics["run_id"] = run_id
            nsga_metrics["runtime"] = nsga_runtime
            detailed_results[instance_name]["NSGA-II"].append(nsga_metrics)

            # K-NSGA-II
            knsga_start = time.time()
            knsga = KNSGAII(
                instance=instance,
                population_size=population_size,
                max_generations=max_generations,
                crossover_rate=0.7,
                mutation_rate=0.2,
                random_state=seed,
            )
            knsga.run(verbose=False)
            knsga_runtime = time.time() - knsga_start
            knsga_metrics = knsga.get_performance_metrics()
            knsga_metrics["run_id"] = run_id
            knsga_metrics["runtime"] = knsga_runtime
            detailed_results[instance_name]["K-NSGA-II"].append(knsga_metrics)

            if verbose:
                print(
                    f"  Run {run_id:2d}/{num_runs}: "
                    f"NSGA Hv={nsga_metrics['hypervolume']:.4f}, "
                    f"K-NSGA Hv={knsga_metrics['hypervolume']:.4f}"
                )

    summary = {}
    for instance_name in instances:
        summary[instance_name] = {
            "NSGA-II": _summarize_runs(detailed_results[instance_name]["NSGA-II"]),
            "K-NSGA-II": _summarize_runs(detailed_results[instance_name]["K-NSGA-II"]),
        }

    json_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    summary_csv_path = os.path.join(output_dir, f"comparison_summary_{timestamp}.csv")
    runs_csv_path = os.path.join(output_dir, f"comparison_runs_{timestamp}.csv")
    delta_csv_path = os.path.join(output_dir, f"comparison_instance_delta_{timestamp}.csv")
    nsga_runs_csv_path = os.path.join(output_dir, f"nsga2_pure_runs_{timestamp}.csv")
    nsga_summary_csv_path = os.path.join(output_dir, f"nsga2_pure_summary_{timestamp}.csv")
    hv_all_plot_path = _boxplot_metric_all_instances(
        detailed_results,
        instances,
        metric_key="hypervolume",
        y_label="Hypervolume",
        output_dir=output_dir,
        timestamp=timestamp,
    )
    sp_all_plot_path = _boxplot_metric_all_instances(
        detailed_results,
        instances,
        metric_key="spacing",
        y_label="Spacing",
        output_dir=output_dir,
        timestamp=timestamp,
    )
    hv_grid_plot_path = _boxplot_hypervolume_dataset_grid(
        detailed_results,
        summary,
        instances,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    hv_plot_paths = _boxplot_metric_per_instance(
        detailed_results,
        instances,
        metric_key="hypervolume",
        y_label="Hypervolume",
        output_dir=output_dir,
        timestamp=timestamp,
    )
    sp_plot_paths = _boxplot_metric_per_instance(
        detailed_results,
        instances,
        metric_key="spacing",
        y_label="Spacing",
        output_dir=output_dir,
        timestamp=timestamp,
    )

    with open(json_path, "w") as f:
        json.dump(
            {
                "experiment_info": {
                    "timestamp": timestamp,
                    "instances": instances,
                    "population_size": population_size,
                    "max_generations": max_generations,
                    "num_runs": num_runs,
                },
                "summary": summary,
                "detailed_runs": detailed_results,
            },
            f,
            indent=2,
            default=str,
        )

    _save_summary_csv(summary, summary_csv_path, instances)
    _save_runs_csv(detailed_results, runs_csv_path, instances)
    _save_instance_delta_csv(summary, delta_csv_path, instances)
    _save_algorithm_only_csv(
        detailed_results,
        summary,
        instances,
        algorithm="NSGA-II",
        runs_path=nsga_runs_csv_path,
        summary_path=nsga_summary_csv_path,
    )
    if verbose:
        print("\n" + "=" * 72)
        print("  COMPARISON SUMMARY (MEAN VALUES)")
        print("=" * 72)
        print(f"  {'Instance':<10} {'Alg':<10} {'Hv':>10} {'SP':>10} {'Time(s)':>10}")
        print("-" * 72)
        for instance_name in instances:
            for algorithm in ["NSGA-II", "K-NSGA-II"]:
                stats = summary[instance_name][algorithm]
                print(
                    f"  {instance_name:<10} {algorithm:<10} "
                    f"{stats['hv_mean']:>10.4f} {stats['sp_mean']:>10.4f} {stats['time_mean']:>10.3f}"
                )
        print("=" * 72)
        print("  Files generated:")
        print(f"    - {json_path}")
        print(f"    - {summary_csv_path}")
        print(f"    - {runs_csv_path}")
        print(f"    - {delta_csv_path} (instance-wise delta report)")
        print(f"    - {nsga_runs_csv_path} (pure NSGA-II runs)")
        print(f"    - {nsga_summary_csv_path} (pure NSGA-II summary)")
        print(f"    - {hv_all_plot_path} (all-instance hypervolume boxplot)")
        print(f"    - {sp_all_plot_path} (all-instance spacing boxplot)")
        print(f"    - {hv_grid_plot_path} (dataset-wise hypervolume grid)")
        print("    - Per-instance hypervolume boxplots:")
        for instance_name in instances:
            print(f"      * {instance_name}: {hv_plot_paths[instance_name]}")
        print("    - Per-instance spacing boxplots:")
        for instance_name in instances:
            print(f"      * {instance_name}: {sp_plot_paths[instance_name]}")

    return {
        "summary": summary,
        "detailed_runs": detailed_results,
        "paths": {
            "json": json_path,
            "summary_csv": summary_csv_path,
            "runs_csv": runs_csv_path,
            "instance_delta_csv": delta_csv_path,
            "nsga_pure_runs_csv": nsga_runs_csv_path,
            "nsga_pure_summary_csv": nsga_summary_csv_path,
            "hypervolume_boxplot_all": hv_all_plot_path,
            "spacing_boxplot_all": sp_all_plot_path,
            "hypervolume_boxplot_grid": hv_grid_plot_path,
            "hypervolume_boxplots": hv_plot_paths,
            "spacing_boxplots": sp_plot_paths,
        },
    }