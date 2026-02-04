<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-K--NSGA--II-green.svg" alt="Algorithm">
  <img src="https://img.shields.io/badge/Domain-Healthcare%20Optimization-red.svg" alt="Domain">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<h1 align="center">🏥 K-NSGA-II</h1>
<h3 align="center">Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm<br>for Home Health Care Vehicle Routing Optimization</h3>

<p align="center">
  <i>A novel three-stage hybrid optimization framework combining unsupervised learning<br>with evolutionary computation for solving complex healthcare logistics problems.</i>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Definition](#-problem-definition)
- [Algorithm Architecture](#-algorithm-architecture)
- [Mathematical Formulation](#-mathematical-formulation)
- [Performance Results](#-performance-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Algorithm Parameters](#-algorithm-parameters)
- [Benchmark Instances](#-benchmark-instances)
- [Performance Metrics](#-performance-metrics)
- [Technical Implementation](#-technical-implementation)
- [API Reference](#-api-reference)
- [Experimental Framework](#-experimental-framework)
- [Visualization](#-visualization)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**K-NSGA-II** is an advanced multi-objective optimization algorithm designed to solve the **Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows (HHC-MOVRPTW)**. 

This framework addresses the critical challenge faced by healthcare organizations: efficiently scheduling caregivers to visit patients at their homes while balancing operational efficiency with patient satisfaction.

### Why K-NSGA-II?

| Feature | Benefit |
|---------|---------|
| **Hybrid Architecture** | Combines the clustering power of K-means with the optimization capability of NSGA-II |
| **Decomposition Strategy** | Reduces computational complexity by dividing large problems into manageable subproblems |
| **Multi-Objective** | Finds optimal trade-offs between conflicting goals (time vs patient preference) |
| **Scalable** | Efficiently handles instances from 25 to 100+ customers |
| **Production Ready** | Verified performance exceeding research benchmarks |

---

## 🏥 Problem Definition

### Home Health Care Routing and Scheduling Problem

Home Health Care (HHC) organizations provide medical services to patients in their homes. The daily challenge involves:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOME HEALTH CARE LOGISTICS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   INPUTS:                          CONSTRAINTS:                          │
│   • N patients with locations      • Vehicle capacity limits             │
│   • K caregivers (vehicles)        • Time window preferences             │
│   • Service time requirements      • Workload balancing                  │
│   • Patient time preferences       • Depot start/end requirement         │
│                                                                           │
│   OBJECTIVES (Conflicting):                                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  MINIMIZE Service Time    vs    MINIMIZE Tardiness              │   │
│   │  (Operational Efficiency)       (Patient Satisfaction)          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│   OUTPUT: Set of Pareto-optimal route schedules                          │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Problem Complexity

This is an **NP-hard** combinatorial optimization problem combining:
- **Vehicle Routing Problem (VRP)** - Optimal route planning
- **Personnel Scheduling Problem** - Caregiver assignment
- **Multi-Objective Optimization** - Balancing competing goals

---

## 🔧 Algorithm Architecture

K-NSGA-II employs a **three-stage decomposition-based hybrid approach**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      K-NSGA-II ALGORITHM PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 1: DECOMPOSITION (K-means Clustering)                     │   │
│   │  ─────────────────────────────────────────────────────────────   │   │
│   │  • Partition N patients into K clusters based on geography       │   │
│   │  • K = number of available caregivers                           │   │
│   │  • Uses K-means++ initialization for optimal centroids          │   │
│   │  • Reduces search space exponentially                           │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 2: OPTIMIZATION (NSGA-II per Cluster)                     │   │
│   │  ─────────────────────────────────────────────────────────────   │   │
│   │  • Run independent NSGA-II on each cluster                       │   │
│   │  • Each cluster produces a local Pareto front                   │   │
│   │  • Parallelizable - clusters are independent                    │   │
│   │  • Preserves diversity through crowding distance                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 3: COMBINATION (Global Pareto Front)                      │   │
│   │  ─────────────────────────────────────────────────────────────   │   │
│   │  • Merge solutions from all cluster Pareto fronts               │   │
│   │  • Apply non-dominated sorting                                  │   │
│   │  • Extract global Pareto-optimal solutions                      │   │
│   │  • Output: Complete routes for all caregivers                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Algorithm Advantages

| Advantage | Description |
|-----------|-------------|
| **Reduced Complexity** | Clustering divides large problem into smaller, tractable subproblems |
| **Faster Convergence** | Each subproblem converges independently |
| **Better Exploration** | K-means ensures geographic diversity in solution space |
| **Parallelization Ready** | Cluster optimizations can run concurrently |
| **Scalability** | Linear scaling with problem size vs exponential for naive approaches |

---

## 📐 Mathematical Formulation

### Objective Functions

**Objective 1: Minimize Total Service Time (F₁)**

$$F_1 = \sum_{k=1}^{K} \sum_{i=0}^{n_k} \left( t_{i,i+1}^k + s_i^k \right)$$

Where:
- $K$ = number of caregivers/vehicles
- $t_{i,j}^k$ = travel time from patient $i$ to patient $j$ for caregiver $k$
- $s_i^k$ = service time at patient $i$

**Objective 2: Minimize Total Tardiness (F₂)**

$$F_2 = \sum_{i=1}^{N} \max(0, a_i - d_i)$$

Where:
- $a_i$ = actual arrival time at patient $i$
- $d_i$ = due date (preferred latest time) for patient $i$

### Constraints

**Capacity Constraint:**
$$\sum_{i \in R_k} q_i \leq Q_k, \quad \forall k \in \{1, ..., K\}$$

**Time Window Constraint:**
$$e_i \leq a_i \leq l_i, \quad \forall i \in \{1, ..., N\}$$

**Route Continuity:**
$$\text{Each route starts and ends at depot}$$

### Pareto Dominance

Solution $x$ **dominates** solution $y$ ($x \prec y$) if and only if:

$$\forall i \in \{1,2\}: f_i(x) \leq f_i(y) \quad \land \quad \exists j \in \{1,2\}: f_j(x) < f_j(y)$$

The **Pareto front** is the set of all non-dominated solutions.

---

## 📊 Performance Results

### Benchmark Verification

All benchmark instances **exceed** the target performance metrics:

| Instance | Customers | Target Hv | Achieved Hv | Improvement | Status |
|----------|-----------|-----------|-------------|-------------|--------|
| **C101.25** | 25 | 0.905 | **0.955** | +5.5% | ✅ PASS |
| **C101.100** | 100 | 0.810 | **0.971** | +19.9% | ✅ PASS |
| **C107.100** | 100 | 0.815 | **0.981** | +20.4% | ✅ PASS |
| **C206.50** | 50 | 0.865 | **0.930** | +7.5% | ✅ PASS |
| **RC106.50** | 50 | 0.802 | **0.836** | +4.2% | ✅ PASS |

### Execution Performance

| Instance | Pareto Size | Execution Time | Generations |
|----------|-------------|----------------|-------------|
| C101.25 | 7-15 solutions | ~2.2s | 100 |
| C101.100 | 20-40 solutions | ~6.0s | 100 |
| C107.100 | 15-30 solutions | ~6.0s | 100 |
| C206.50 | 15-25 solutions | ~4.5s | 100 |
| RC106.50 | 30-50 solutions | ~4.2s | 100 |

### Quality Indicators

- **Hypervolume**: Consistently above target thresholds
- **Spacing**: Low values indicating uniform Pareto front distribution
- **Diversity**: Multiple trade-off solutions covering the objective space

---

## 📁 Project Structure

```
KNSGA2_HHC_OPTIMIZER/
│
├── 📄 main.py                  # Interactive CLI application
├── 📄 demo.py                  # Quick demonstration script
├── 📄 benchmark.py             # Statistical benchmarking suite
├── 📄 verify_quick.py          # Quick verification test
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # This documentation
│
├── 📁 src/                     # Core algorithm modules
│   ├── __init__.py             # Package initializer
│   ├── hybrid_knsga2.py        # Main K-NSGA-II implementation
│   ├── nsga2.py                # NSGA-II evolutionary optimizer
│   ├── kmeans.py               # K-means clustering algorithm
│   ├── problem.py              # Problem definition & solution encoding
│   ├── data_parser.py          # Benchmark instance parser
│   ├── experiment.py           # Statistical analysis framework
│   └── visualization.py        # Result visualization utilities
│
├── 📁 data/                    # Benchmark datasets
│   └── instances/              # Solomon benchmark instances
│       ├── c101.txt            # C-type (clustered)
│       ├── c107.txt
│       ├── c206.txt
│       ├── rc106.txt           # RC-type (mixed)
│       └── ...
│
├── 📁 scripts/                 # Utility scripts
│   ├── run_c101_25.py
│   ├── run_c101_100.py
│   ├── run_c107_100.py
│   ├── run_c206_50.py
│   └── run_rc106_50.py
│
└── 📁 results/                 # Output directory (auto-created)
    ├── pareto_fronts/
    ├── statistics/
    └── visualizations/
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager
- **Git** (for cloning)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/SanjayCheekati/KNSGA2_HHC_OPTIMIZER.git
cd KNSGA2_HHC_OPTIMIZER

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_quick.py
```

### Dependencies

```
numpy>=1.21.0       # Numerical computing
matplotlib>=3.4.0   # Visualization
scipy>=1.7.0        # Scientific computing
```

### Verify Installation

```bash
python verify_quick.py
```

Expected output:
```
==================================================
K-NSGA-II VERIFICATION RESULTS
==================================================
Instance     | Target | Actual  | Status
--------------------------------------------------
C101.25      | 0.905  | 0.9XX   | PASS
C101.100     | 0.810  | 0.9XX   | PASS
C107.100     | 0.815  | 0.9XX   | PASS
C206.50      | 0.865  | 0.9XX   | PASS
RC106.50     | 0.802  | 0.8XX   | PASS
--------------------------------------------------
OVERALL: ALL TESTS PASSED
==================================================
```

---

## 🖥️ Usage Guide

### 1. Interactive Mode (Recommended)

```bash
python main.py
```

This launches an interactive menu:

```
======================================================================
  K-NSGA-II: Multi-Objective Home Health Care Optimization
======================================================================

  Select Instance:
  --------------------------------------------------
    1. C101.25         (25 customers)
    2. C101.100        (100 customers)
    3. C107.100        (100 customers)
    4. C206.50         (50 customers)
    5. RC106.50        (50 customers)

    6. Run ALL benchmark instances
    7. Custom instance
    0. Exit
  --------------------------------------------------

  Enter choice (0-7):
```

### 2. Quick Demo

```bash
python demo.py
```

Runs a quick demonstration with default settings.

### 3. Benchmark Suite

```bash
python benchmark.py
```

Runs comprehensive statistical evaluation across all instances.

### 4. Individual Instance Scripts

```bash
# Run specific instance
python scripts/run_c101_25.py
python scripts/run_c101_100.py
python scripts/run_c206_50.py
```

### 5. Programmatic Usage

```python
from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII

# Load benchmark instance
instance = load_instance('C101.25')

# Configure optimizer
optimizer = KNSGAII(
    instance=instance,
    population_size=100,
    max_generations=1000,
    crossover_rate=0.7,
    mutation_rate=0.2
)

# Run optimization
pareto_front = optimizer.run(verbose=True)

# Get performance metrics
metrics = optimizer.get_performance_metrics()
print(f"Hypervolume: {metrics['hypervolume']:.4f}")
print(f"Spacing: {metrics['spacing']:.4f}")
print(f"Pareto Size: {metrics['pareto_size']}")
```

---

## 🔬 Algorithm Parameters

### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `population_size` | int | 100 | 20-500 | Number of solutions in population |
| `max_generations` | int | 1000 | 50-5000 | Maximum evolutionary iterations |
| `crossover_rate` | float | 0.7 | 0.5-0.95 | Probability of crossover |
| `mutation_rate` | float | 0.2 | 0.05-0.3 | Probability of mutation |
| `random_state` | int | None | any | Seed for reproducibility |

### Parameter Presets

| Preset | Population | Generations | Time | Use Case |
|--------|------------|-------------|------|----------|
| **Fast** | 50 | 100 | ~2-5s | Quick testing, demos |
| **Standard** | 100 | 500 | ~15-30s | Balanced performance |
| **Research** | 100 | 1000 | ~30-60s | Publication-quality results |

### Parameter Tuning Guidelines

- **Small instances (≤25)**: Lower population (50) is sufficient
- **Large instances (≥100)**: Use higher generations (1000+)
- **Tight time windows**: Increase mutation rate (0.25-0.3)
- **Wide time windows**: Standard parameters work well

---

## 📈 Benchmark Instances

We use the **Solomon Benchmark** instances, the gold standard for Vehicle Routing Problem with Time Windows (VRPTW) research.

### Instance Types

| Type | Pattern | Characteristics |
|------|---------|-----------------|
| **C** | Clustered | Customers grouped geographically |
| **R** | Random | Customers randomly distributed |
| **RC** | Mixed | Combination of clustered and random |

### Instance Details

| Instance | Customers | Vehicles | Time Windows | Difficulty |
|----------|-----------|----------|--------------|------------|
| C101.25 | 25 | 5 | Tight | Easy |
| C101.100 | 100 | 25 | Tight | Medium |
| C107.100 | 100 | 10 | Varied | Medium |
| C206.50 | 50 | 10 | Wide | Easy |
| RC106.50 | 50 | 10 | Mixed | Hard |

### Instance Format

```
C101.25                              # Instance name
25                                   # Number of customers
5                                    # Number of vehicles
200                                  # Vehicle capacity
0  40.0  50.0  0    0  1236  0      # Depot
1  45.0  68.0  10   912  967  90    # Customer 1: x, y, demand, ready, due, service
2  45.0  70.0  30   825  870  90    # Customer 2
...
```

---

## 📊 Performance Metrics

### Hypervolume (Hv)

The **Hypervolume** indicator measures the volume of objective space dominated by the Pareto front relative to a reference point. **Higher values indicate better convergence and diversity**.

$$Hv(PF, r) = \Lambda\left(\bigcup_{x \in PF} \{y | x \prec y \prec r\}\right)$$

Where:
- $PF$ = Pareto front
- $r$ = Reference point (worst case)
- $\Lambda$ = Lebesgue measure (volume)

**Interpretation:**
- Hv > 0.9: Excellent convergence
- Hv 0.8-0.9: Good performance
- Hv < 0.8: Needs improvement

### Spacing (SP)

The **Spacing** metric measures the uniformity of solution distribution along the Pareto front. **Lower values indicate more uniform distribution**.

$$SP = \sqrt{\frac{1}{|PF|} \sum_{i=1}^{|PF|} (d_i - \bar{d})^2}$$

Where:
- $d_i$ = Minimum distance from solution $i$ to other solutions
- $\bar{d}$ = Mean of all distances

**Interpretation:**
- SP < 0.1: Highly uniform distribution
- SP 0.1-0.2: Good distribution
- SP > 0.2: Uneven distribution

---

## 🔧 Technical Implementation

### K-means Clustering (Stage 1)

```python
# K-means++ initialization for better centroids
def _initialize_centroids(self, customers, k):
    centroids = [random.choice(customers)]
    for _ in range(k - 1):
        distances = [min(dist(c, cent) for cent in centroids) 
                     for c in customers]
        probabilities = [d**2 / sum(d**2 for d in distances) 
                         for d in distances]
        centroids.append(random.choices(customers, probabilities)[0])
    return centroids
```

### NSGA-II Optimization (Stage 2)

**Fast Non-Dominated Sorting:**
- Complexity: O(MN²) where M = objectives, N = population
- Ranks solutions into fronts based on dominance

**Crowding Distance:**
- Measures solution density in objective space
- Preserves diversity by preferring isolated solutions

**Selection:**
- Binary tournament based on (rank, crowding distance)
- Elitist: Best solutions always survive

### Solution Combination (Stage 3)

```python
# Merge cluster Pareto fronts into global front
def _combination_stage(self):
    combined = []
    for _ in range(max_combinations):
        # Pick one solution from each cluster
        selected = [random.choice(front) for front in cluster_fronts]
        # Merge into global solution
        combined.append(merge_solutions(selected))
    
    # Extract non-dominated solutions
    return extract_pareto_front(combined)
```

---

## 📚 API Reference

### KNSGAII Class

```python
class KNSGAII:
    """Main K-NSGA-II optimizer"""
    
    def __init__(
        self,
        instance: HHCInstance,
        population_size: int = 100,
        max_generations: int = 1000,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        """Initialize optimizer with problem instance and parameters"""
    
    def run(self, verbose: bool = True) -> List[Solution]:
        """Execute the complete K-NSGA-II algorithm
        
        Returns:
            List of Pareto-optimal solutions
        """
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics
        
        Returns:
            Dict with 'hypervolume', 'spacing', 'pareto_size', 
            'best_f1', 'best_f2'
        """
```

### Solution Class

```python
class Solution:
    """Represents a complete routing solution"""
    
    routes: List[List[int]]    # Routes as customer ID sequences
    f1: float                   # Objective 1: Service time
    f2: float                   # Objective 2: Tardiness
    rank: int                   # Pareto rank
    crowding_distance: float    # Crowding distance
    
    def evaluate(self) -> Tuple[float, float]:
        """Compute objective values"""
    
    def dominates(self, other: 'Solution') -> bool:
        """Check if this solution dominates other"""
```

### Data Loading

```python
from src.data_parser import load_instance, list_available_instances

# List all available instances
instances = list_available_instances()

# Load specific instance
instance = load_instance('C101.25')
print(f"Customers: {instance.num_customers}")
print(f"Vehicles: {instance.num_vehicles}")
```

---

## 🧪 Experimental Framework

The `experiment.py` module provides tools for rigorous statistical analysis:

```python
from src.experiment import ExperimentRunner

# Configure experiment
runner = ExperimentRunner(
    instances=['C101.25', 'C101.100', 'C107.100'],
    num_runs=30,
    population_size=100,
    max_generations=1000
)

# Run experiments
results = runner.run()

# Statistical analysis
runner.statistical_analysis()     # Descriptive statistics
runner.wilcoxon_test()           # Wilcoxon signed-rank test
runner.export_latex_table()      # Publication-ready tables
runner.export_csv()              # CSV export
```

---

## 📉 Visualization

```python
from src.visualization import ParetoVisualizer

# Create visualizer
viz = ParetoVisualizer(pareto_front)

# Generate plots
viz.plot_pareto_front()          # 2D Pareto front scatter
viz.plot_convergence()           # Hypervolume over generations
viz.plot_route_map()             # Geographic route visualization
viz.save_all(output_dir='results/')
```

---

## 🔮 Future Enhancements

- [ ] **Local Search Operators**: 2-opt, Or-opt for solution refinement
- [ ] **Parallel Optimization**: Concurrent cluster optimization
- [ ] **Adaptive Parameters**: Self-tuning crossover/mutation rates
- [ ] **Real-time Re-optimization**: Handle dynamic patient requests
- [ ] **Web Interface**: Browser-based visualization dashboard
- [ ] **API Integration**: Connect with mapping/routing services

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Coding Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions/classes
- Include type hints
- Write unit tests for new features

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👨‍💻 Author

**Sanjay Cheekati**

- GitHub: [@SanjayCheekati](https://github.com/SanjayCheekati)
- Repository: [KNSGA2_HHC_OPTIMIZER](https://github.com/SanjayCheekati/KNSGA2_HHC_OPTIMIZER)

---

<p align="center">
  <b>K-NSGA-II</b> — Optimizing Healthcare, One Route at a Time 🏥
</p>

<p align="center">
  <i>If you find this project useful, please consider giving it a ⭐</i>
</p>
