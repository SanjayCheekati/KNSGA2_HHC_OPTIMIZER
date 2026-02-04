"""
Publication-Quality Visualization for K-NSGA-II Results
=========================================================

Generates research-grade plots for:
- Pareto front visualization
- Convergence analysis
- Statistical comparison charts
- Multi-instance comparison

Designed for academic publications and presentations.
"""

import os
import math
from typing import List, Dict, Tuple, Optional


class ParetoVisualizer:
    """
    Visualization tools for Pareto fronts and algorithm analysis
    
    Note: Uses ASCII art for terminal display.
    For publication plots, export data and use matplotlib externally.
    """
    
    def __init__(self, width: int = 60, height: int = 20):
        """Initialize visualizer with display dimensions"""
        self.width = width
        self.height = height
    
    def plot_pareto_ascii(
        self,
        pareto_front: List[Tuple[float, float]],
        title: str = "Pareto Front",
        x_label: str = "F1 (Service Time)",
        y_label: str = "F2 (Tardiness)"
    ) -> str:
        """
        Create ASCII art visualization of Pareto front
        
        Args:
            pareto_front: List of (f1, f2) tuples
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
        
        Returns:
            ASCII string representation of plot
        """
        if not pareto_front:
            return "No solutions to plot"
        
        # Extract values
        f1_vals = [p[0] for p in pareto_front]
        f2_vals = [p[1] for p in pareto_front]
        
        # Calculate ranges
        f1_min, f1_max = min(f1_vals), max(f1_vals)
        f2_min, f2_max = min(f2_vals), max(f2_vals)
        
        # Handle edge cases
        f1_range = f1_max - f1_min if f1_max > f1_min else 1
        f2_range = f2_max - f2_min if f2_max > f2_min else 1
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Plot points
        for f1, f2 in pareto_front:
            x = int((f1 - f1_min) / f1_range * (self.width - 1))
            y = int((f2 - f2_min) / f2_range * (self.height - 1))
            y = self.height - 1 - y  # Invert Y axis
            
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            grid[y][x] = '●'
        
        # Build output string
        lines = []
        lines.append(f"╔{'═' * (self.width + 2)}╗")
        lines.append(f"║ {title:^{self.width}} ║")
        lines.append(f"╠{'═' * (self.width + 2)}╣")
        
        for row in grid:
            lines.append(f"║ {''.join(row)} ║")
        
        lines.append(f"╠{'═' * (self.width + 2)}╣")
        lines.append(f"║ {x_label:^{self.width}} ║")
        lines.append(f"╚{'═' * (self.width + 2)}╝")
        
        # Add statistics
        lines.append(f"\nPoints: {len(pareto_front)}")
        lines.append(f"F1 range: [{f1_min:.2f}, {f1_max:.2f}]")
        lines.append(f"F2 range: [{f2_min:.2f}, {f2_max:.2f}]")
        
        return '\n'.join(lines)
    
    def plot_convergence_ascii(
        self,
        history: List[float],
        title: str = "Convergence Plot",
        metric: str = "Hypervolume"
    ) -> str:
        """Create ASCII convergence plot"""
        if not history:
            return "No history to plot"
        
        height = 15
        width = min(60, len(history))
        
        # Normalize values
        min_val = min(history)
        max_val = max(history)
        val_range = max_val - min_val if max_val > min_val else 1
        
        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Sample points if history is longer than width
        step = max(1, len(history) // width)
        sampled = history[::step][:width]
        
        # Plot line
        for i, val in enumerate(sampled):
            y = int((val - min_val) / val_range * (height - 1))
            y = height - 1 - y
            y = max(0, min(height - 1, y))
            
            if i < width:
                grid[y][i] = '█'
        
        # Build output
        lines = []
        lines.append(f"\n{title}")
        lines.append("─" * (width + 5))
        
        for i, row in enumerate(grid):
            if i == 0:
                label = f"{max_val:.3f}"
            elif i == height - 1:
                label = f"{min_val:.3f}"
            else:
                label = "     "
            lines.append(f"{label[:5]:>5}│{''.join(row)}")
        
        lines.append(f"     └{'─' * width}")
        lines.append(f"      Generation →")
        
        return '\n'.join(lines)
    
    def create_comparison_table(
        self,
        results: Dict[str, Dict],
        paper_targets: Dict[str, Dict]
    ) -> str:
        """
        Create comparison table between results and paper targets
        
        Args:
            results: Dictionary of instance results
            paper_targets: Dictionary of paper target values
        
        Returns:
            Formatted comparison table
        """
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("COMPARISON WITH PAPER RESULTS")
        lines.append("=" * 80)
        lines.append(f"{'Instance':<12} {'Our Hv':>10} {'Paper Hv':>10} {'Δ Hv':>10} "
                    f"{'Our SP':>10} {'Paper SP':>10} {'Status':>10}")
        lines.append("-" * 80)
        
        for inst, res in results.items():
            our_hv = res.get('hypervolume', 0)
            our_sp = res.get('spacing', 0)
            
            if inst in paper_targets:
                paper_hv = paper_targets[inst]['hv']
                paper_sp = paper_targets[inst]['sp']
                delta_hv = our_hv - paper_hv
                status = "✓ PASS" if our_hv >= paper_hv else "✗ FAIL"
            else:
                paper_hv = paper_sp = delta_hv = 0
                status = "N/A"
            
            lines.append(f"{inst:<12} {our_hv:>10.4f} {paper_hv:>10.3f} "
                        f"{delta_hv:>+10.4f} {our_sp:>10.4f} {paper_sp:>10.3f} {status:>10}")
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def export_for_matplotlib(
        self,
        pareto_front: List[Tuple[float, float]],
        filepath: str
    ):
        """
        Export Pareto front data for external plotting with matplotlib
        
        Args:
            pareto_front: List of (f1, f2) tuples
            filepath: Output CSV file path
        """
        with open(filepath, 'w') as f:
            f.write("f1,f2\n")
            for f1, f2 in pareto_front:
                f.write(f"{f1},{f2}\n")
    
    def export_latex_tikz(
        self,
        pareto_front: List[Tuple[float, float]],
        filepath: str,
        title: str = "Pareto Front"
    ):
        """
        Export Pareto front as TikZ code for LaTeX
        
        Args:
            pareto_front: List of (f1, f2) tuples
            filepath: Output .tex file path
            title: Plot title
        """
        with open(filepath, 'w') as f:
            f.write("% TikZ code for Pareto Front visualization\n")
            f.write("% Include in LaTeX with: \\usepackage{pgfplots}\n\n")
            f.write("\\begin{tikzpicture}\n")
            f.write("\\begin{axis}[\n")
            f.write(f"    title={{{title}}},\n")
            f.write("    xlabel={$F_1$ (Service Time)},\n")
            f.write("    ylabel={$F_2$ (Tardiness)},\n")
            f.write("    grid=major,\n")
            f.write("    legend pos=north east\n")
            f.write("]\n")
            f.write("\\addplot[only marks, mark=*, blue] coordinates {\n")
            
            for f1, f2 in pareto_front:
                f.write(f"    ({f1:.2f}, {f2:.2f})\n")
            
            f.write("};\n")
            f.write("\\legend{K-NSGA-II}\n")
            f.write("\\end{axis}\n")
            f.write("\\end{tikzpicture}\n")


def generate_publication_figures(
    results_dir: str,
    output_dir: str = "figures"
):
    """
    Generate all publication-ready figures
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory for output figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = ParetoVisualizer()
    
    # Generate matplotlib script
    script_path = os.path.join(output_dir, "generate_plots.py")
    
    with open(script_path, 'w') as f:
        f.write('''"""
Publication-Ready Plots for K-NSGA-II Results
Run this script with matplotlib installed:
    pip install matplotlib
    python generate_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_pareto_front(f1_vals, f2_vals, title, filename):
    """Create publication-quality Pareto front plot"""
    fig, ax = plt.subplots()
    
    ax.scatter(f1_vals, f2_vals, c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Connect points with line (sorted by f1)
    sorted_indices = np.argsort(f1_vals)
    ax.plot(np.array(f1_vals)[sorted_indices], np.array(f2_vals)[sorted_indices], 
            'b--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('$F_1$ (Total Service Time)')
    ax.set_ylabel('$F_2$ (Total Tardiness)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_comparison_bar(instances, our_hv, paper_hv, filename):
    """Create comparison bar chart"""
    x = np.arange(len(instances))
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, our_hv, width, label='K-NSGA-II (Ours)', color='steelblue')
    bars2 = ax.bar(x + width/2, paper_hv, width, label='Paper Results', color='coral')
    
    ax.set_ylabel('Hypervolume')
    ax.set_title('Hypervolume Comparison: Our Results vs Paper')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

if __name__ == "__main__":
    print("Generating publication figures...")
    # Add your data here or load from results JSON
    print("Done!")
''')
    
    print(f"Matplotlib script created: {script_path}")
    print("Run 'pip install matplotlib' then 'python generate_plots.py' to create figures")
