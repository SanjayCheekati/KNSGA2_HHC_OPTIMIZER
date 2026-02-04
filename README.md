<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-K--NSGA--II-green.svg" alt="Algorithm">
  <img src="https://img.shields.io/badge/Domain-Healthcare%20Optimization-red.svg" alt="Domain">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<h1 align="center">K-NSGA-II</h1>
<h3 align="center">Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm<br>for Home Health Care Optimization</h3>

<p align="center">
  <i>A novel three-stage hybrid optimization framework combining unsupervised learning<br>with evolutionary computation for solving complex vehicle routing problems.</i>
</p>

---

## 📋 Overview

**K-NSGA-II** is an advanced multi-objective optimization algorithm designed to solve the **Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows (HHC-MOVRPTW)**. This framework addresses the critical challenge of efficiently scheduling healthcare workers to visit patients at their homes while respecting time preferences and operational constraints.

### The Problem

Home Health Care (HHC) organizations face a complex daily challenge:
- **Multiple caregivers** must visit **multiple patients** at their homes
- Each patient has **preferred time windows** for visits
- Caregivers have **capacity constraints** (workload limits)
- Goals are conflicting: **minimize travel time** vs **maximize patient satisfaction**

This is an NP-hard combinatorial optimization problem combining:
- Vehicle Routing Problem (VRP)
- Personnel Scheduling Problem
- Multi-Objective Optimization

### Our Solution

K-NSGA-II tackles this problem using a **decomposition-based hybrid approach**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        K-NSGA-II Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Stage 1: DECOMPOSITION                                             │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  K-means clustering partitions patients geographically       │   │
│   │  into k clusters (where k = number of caregivers)           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│   Stage 2: OPTIMIZATION                                              │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  NSGA-II runs independently on each cluster                  │   │
│   │  → Produces k local Pareto fronts                           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│   Stage 3: COMBINATION                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Local Pareto fronts are merged into global optimal front    │   │
│   │  → Non-dominated sorting extracts best solutions            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Objectives

The algorithm simultaneously optimizes two competing objectives:

| Objective | Symbol | Description | Goal |
|-----------|--------|-------------|------|
| **Service Time** | F₁ | Total travel time + service duration | Minimize |
| **Tardiness** | F₂ | Deviation from patient time preferences | Minimize |

$$F_1 = \sum_{c \in C} \sum_{(i,j) \in routes} (t_{ij} + s_i)$$

$$F_2 = \sum_{i \in N} \max(0, arrival_i - due_i)$$

---

## 🚀 Key Features

### Algorithmic Innovations
- **Hybrid Architecture**: Combines unsupervised learning (K-means) with evolutionary optimization (NSGA-II)
- **Problem Decomposition**: Reduces computational complexity by partitioning into subproblems
- **Parallelizable**: Cluster optimizations are independent and can run concurrently
- **Scalable**: Handles instances from 25 to 100+ customers efficiently

### Technical Features
- **K-means++ Initialization**: Superior centroid selection for better clustering
- **Fast Non-dominated Sorting**: O(MN²) complexity for Pareto ranking
- **Crowding Distance**: Maintains solution diversity across Pareto front
- **Elitist Selection**: Preserves best solutions across generations

### Constraint Handling
- ✅ Vehicle capacity constraints
- ✅ Time window constraints (soft)
- ✅ Workload balancing
- ✅ Route continuity (depot start/end)

---

## 📁 Project Structure

```
K-NSGA-II/
├── main.py                 # Interactive CLI interface
├── demo.py                 # Quick demonstration script
├── benchmark.py            # Statistical benchmarking
├── requirements.txt        # Python dependencies
│
├── src/                    # Core algorithm modules
│   ├── __init__.py
│   ├── hybrid_knsga2.py    # Main K-NSGA-II implementation
│   ├── nsga2.py            # NSGA-II optimizer
│   ├── kmeans.py           # K-means clustering
│   ├── problem.py          # Problem definition & solution encoding
│   ├── data_parser.py      # Benchmark instance parser
│   ├── experiment.py       # Statistical analysis framework
│   └── visualization.py    # Result visualization
│
├── datasets/               # Solomon benchmark instances
│   ├── C_type/             # Clustered customer distribution
│   ├── R_type/             # Random customer distribution
│   └── RC_type/            # Mixed distribution
│
└── results/                # Output directory
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/K-NSGA-II.git
cd K-NSGA-II

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
```

---

## 🖥️ Usage

### Interactive Mode

```bash
python main.py
```

This launches an interactive menu where you can:
- Select benchmark instances
- Choose parameter presets (Fast/Standard/Research)
- Run multiple trials for statistical analysis

### Quick Demo

```bash
python demo.py C101.25
```

### Benchmark Suite

```bash
python benchmark.py
```

Runs statistical evaluation across all benchmark instances with multiple runs.

### Command Line

```bash
# Basic usage
python main.py <instance> <population> <generations> <runs>

# Examples
python main.py C101.25
python main.py C101.100 100 500 5
python main.py C206.50 100 1000 20
```

---

## 📊 Performance Metrics

### Hypervolume (Hv)
Measures the volume of objective space dominated by the Pareto front. **Higher is better**.

$$Hv(PF, ref) = \sum_{i=1}^{|PF|} |F_1(i+1) - F_1(i)| \cdot |F_{2,ref} - F_2(i)|$$

### Spacing (SP)
Measures uniformity of solution distribution. **Lower is better**.

$$SP = \sqrt{\frac{1}{|PF|} \sum_{i=1}^{|PF|} (d_i - \bar{d})^2}$$

---

## 🔬 Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Number of solutions in population |
| `max_generations` | 1000 | Maximum evolutionary iterations |
| `crossover_rate` | 0.7 | Probability of crossover |
| `mutation_rate` | 0.2 | Probability of mutation |
| `k_clusters` | auto | Number of clusters (= num_vehicles) |

### Parameter Presets

| Preset | Population | Generations | Use Case |
|--------|------------|-------------|----------|
| Fast | 50 | 100 | Quick testing, demos |
| Standard | 100 | 500 | Balanced performance |
| Research | 100 | 1000 | Publication-quality results |

---

## 📈 Benchmark Instances

We use the **Solomon Benchmark** instances, the gold standard for VRPTW research:

| Instance | Customers | Type | Characteristics |
|----------|-----------|------|-----------------|
| C101.25 | 25 | Clustered | Tight time windows |
| C101.100 | 100 | Clustered | Large-scale |
| C107.100 | 100 | Clustered | Varied time windows |
| C206.50 | 50 | Clustered | Wide time windows |
| RC106.50 | 50 | Mixed | Challenging distribution |

---

## 🧪 Experimental Framework

The `experiment.py` module provides a comprehensive statistical analysis framework:

```python
from src.experiment import ExperimentRunner

# Configure experiment
runner = ExperimentRunner(
    instances=['C101.25', 'C101.100'],
    num_runs=30,
    population_size=100,
    max_generations=1000
)

# Run and analyze
results = runner.run()
runner.statistical_analysis()  # Wilcoxon signed-rank test
runner.export_latex_table()    # Publication-ready tables
```

---

## 📉 Visualization

The `visualization.py` module generates publication-quality figures:

```python
from src.visualization import ParetoVisualizer

viz = ParetoVisualizer(pareto_front)
viz.plot_pareto_front()         # 2D Pareto front
viz.plot_convergence()          # Hypervolume over generations
viz.export_tikz()               # LaTeX/TikZ export
```

---

## 🔮 Future Enhancements

- [ ] Local search operators (2-opt, Or-opt)
- [ ] Parallel cluster optimization
- [ ] Adaptive parameter control
- [ ] Real-time re-optimization
- [ ] Integration with mapping APIs

---

## 📚 Theoretical Background

### Multi-Objective Optimization

In multi-objective optimization, we seek solutions that represent optimal trade-offs between competing objectives. A solution **x** dominates solution **y** if:

$$\forall i: f_i(x) \leq f_i(y) \land \exists j: f_j(x) < f_j(y)$$

The **Pareto front** contains all non-dominated solutions.

### NSGA-II

The Non-dominated Sorting Genetic Algorithm II uses:
1. **Fast non-dominated sorting** for ranking
2. **Crowding distance** for diversity preservation
3. **Elitist selection** for convergence

### K-means Clustering

Partitions data into k clusters by minimizing within-cluster sum of squares:

$$\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@software{knsga2_hhc,
  title = {K-NSGA-II: Hybrid Decomposition-Based Multi-Objective 
           Evolutionary Algorithm for Home Health Care Optimization},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/K-NSGA-II}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<p align="center">
  <b>K-NSGA-II</b> — Optimizing Healthcare, One Route at a Time
</p>
