<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-K--NSGA--II-2563EB?style=flat-square" alt="Algorithm">
  <img src="https://img.shields.io/badge/Domain-Healthcare%20Optimization-DC2626?style=flat-square" alt="Domain">
  <img src="https://img.shields.io/badge/Problem-HHC--MOVRPTW-16A34A?style=flat-square" alt="Problem">
  <img src="https://img.shields.io/badge/License-MIT-9333EA?style=flat-square" alt="License">
</p>

<h1 align="center">K-NSGA-II</h1>
<h3 align="center">A Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm<br>for the Home Health Care Vehicle Routing Problem with Time Windows</h3>

<p align="center">
  <i>Three-stage optimization pipeline: K-means decomposition → NSGA-II per cluster → Pareto front combination</i>
</p>

---

## Abstract

This repository implements **K-NSGA-II**, a hybrid multi-objective optimization algorithm for the **Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows (HHC-MOVRPTW)**. The algorithm decomposes the NP-hard combinatorial problem using K-means clustering, independently optimizes each subproblem with NSGA-II, and merges the resulting Pareto fronts into a global non-dominated solution set. The framework is implemented in pure Python with zero external dependencies for the core algorithm, and has been validated on Solomon VRPTW benchmark instances across three distribution types (C, R, RC) with 25–100 customers.

---

## Table of Contents

- [Problem Formulation](#problem-formulation)
- [Algorithm Design](#algorithm-design)
- [Computational Experiments](#computational-experiments)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [API Reference](#api-reference)
- [Authors](#authors)
- [License](#license)

---

## Problem Formulation

### HHC-MOVRPTW

Home Health Care (HHC) organizations assign caregivers to visit geographically distributed patients within preferred time windows. This gives rise to a bi-objective optimization problem:

**Objective 1 — Minimize Total Service Time ($F_1$)**

$$F_1 = \sum_{k=1}^{K} \sum_{i=0}^{n_k} \left( t_{i,i+1}^k + s_i^k \right)$$

**Objective 2 — Minimize Total Tardiness ($F_2$)**

$$F_2 = \sum_{i=1}^{N} \max(0,\; a_i - d_i)$$

Where $t_{i,j}^k$ is the travel time on arc $(i,j)$ for vehicle $k$, $s_i^k$ is the service duration, $a_i$ is the arrival time, and $d_i$ is the due date of patient $i$.

### Constraints

| Constraint | Formulation |
|---|---|
| Vehicle capacity | $\sum_{i \in R_k} q_i \leq Q_k \quad \forall k$ |
| Time windows | $e_i \leq a_i \leq l_i \quad \forall i$ |
| Depot origin/return | Each route starts and ends at the depot |

### Pareto Dominance

Solution $\mathbf{x}$ dominates $\mathbf{y}$ ($\mathbf{x} \prec \mathbf{y}$) iff:

$$\forall\, i \in \{1,2\}:\; f_i(\mathbf{x}) \leq f_i(\mathbf{y}) \;\;\land\;\; \exists\, j \in \{1,2\}:\; f_j(\mathbf{x}) < f_j(\mathbf{y})$$

The output is a set of **Pareto-optimal** solutions representing the efficient trade-off frontier between operational cost and patient satisfaction.

---

## Algorithm Design

### Three-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1: DECOMPOSITION                                          │
│  K-means++ partitions N patients into K geographic clusters      │
│  Search space: O(N!) → K independent O((N/K)!) subproblems       │
├──────────────────────────────────────────────────────────────────┤
│  STAGE 2: OPTIMIZATION                                           │
│  NSGA-II runs independently on each cluster                      │
│  Operators: OX crossover, swap/insert mutation                   │
│  Selection: Binary tournament on (rank, crowding distance)       │
├──────────────────────────────────────────────────────────────────┤
│  STAGE 3: COMBINATION                                            │
│  Cluster-level Pareto fronts are merged combinatorially          │
│  Non-dominated sorting extracts the global Pareto front          │
└──────────────────────────────────────────────────────────────────┘
```

### NSGA-II Components

| Component | Method |
|---|---|
| **Initialization** | Random permutation encoding |
| **Crossover** | Order Crossover (OX), rate $P_c$ |
| **Mutation** | Swap + Insert, rate $P_m$ |
| **Selection** | Binary tournament (rank, then crowding) |
| **Ranking** | Fast non-dominated sorting, $O(MN^2)$ |
| **Diversity** | Crowding distance in objective space |
| **Elitism** | Parent+offspring merge, truncate to $N$ |

### K-means Clustering

- **Initialization**: K-means++ for deterministic, well-spread centroids
- **Distance metric**: Euclidean on $(x, y)$ patient coordinates
- **K selection**: Experimentally optimized per instance type and size

---

## Computational Experiments

### Benchmark Instances

Experiments use adapted **Solomon VRPTW** instances across three spatial distributions:

| Instance | Type | Customers | Vehicles | K (clusters) | Time Windows |
|---|---|---|---|---|---|
| C101.25 | Clustered | 25 | 5 | 2 | Tight |
| C101.100 | Clustered | 100 | 25 | 4 | Tight |
| C107.100 | Clustered | 100 | 10 | 4 | Varied |
| C206.50 | Clustered | 50 | 10 | 3 | Wide |
| R109.25 | Random | 25 | 5 | 2 | Tight |
| RC106.50 | Mixed | 50 | 10 | 3 | Mixed |

### Experimental Protocol

- **Independent runs**: 30 per instance (configurable)
- **Genetic parameters**: $N = 100$, $G = 500$, $P_c = 0.7$, $P_m = 0.2$
- **Service time**: 20 minutes (HHC-standardized)
- **Evaluation metrics**: Hypervolume (Hv), Spacing (SP), Pareto front cardinality
- **Seeds**: Deterministic via `random_state` for full reproducibility

### Results

Extensive benchmark runs on the six Solomon instances indicate that the current implementation:

* **Meets or exceeds published hypervolume targets** for all distributions when evaluated over multiple seeds.
* Computes **Spacing (SP)** correctly using the Schott (1995) formula with per-objective normalization, resolving previously observed discrepancies.

Users may reproduce and inspect detailed numerical results by executing `main.py` with desired instances; output logs contain best/mean hypervolume, spacing, and Pareto front sizes.


---

## Repository Structure

```
KNSGA2_HHC_OPTIMIZER/
├── main.py                         Entry point (interactive CLI + CLI args)
├── LICENSE
│
├── src/
│   ├── __init__.py                 Package exports
│   ├── hybrid_knsga2.py            K-NSGA-II: 3-stage hybrid algorithm
│   ├── nsga2.py                    NSGA-II engine (sorting, crowding, selection)
│   ├── kmeans.py                   K-means++ clustering
│   ├── problem.py                  HHCInstance, Customer, Solution definitions
│   ├── data_parser.py              Solomon-format instance parser
│   └── experiment.py               Statistical experiment runner
│
└── datasets/
    ├── C_type/                     Clustered distribution instances
    ├── R_type/                     Random distribution instances
    └── RC_type/                    Mixed distribution instances
```

---

## Usage

### Interactive Mode

```bash
python main.py
```

Menu options:
1. Select a specific benchmark instance and parameter preset
2. Load a custom instance

### Command-Line Interface

```bash
# Single instance with defaults
python main.py C101.25

# Custom configuration: instance, population, generations, runs
python main.py C101.100 100 500 5
```

### Dependencies

Although the core algorithm relies solely on the Python standard library, a
`requirements.txt` file is provided to document the supported Python version and
track any future third‑party packages. Install with:

```bash
pip install -r requirements.txt
```


### Programmatic Interface

```python
from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII

instance = load_instance('C101.25')

optimizer = KNSGAII(
    instance=instance,
    population_size=100,
    max_generations=500,
    crossover_rate=0.7,
    mutation_rate=0.2,
    random_state=42
)

pareto_front = optimizer.run(verbose=True)
metrics = optimizer.get_performance_metrics()

print(f"Hypervolume: {metrics['hypervolume']:.4f}")
print(f"Spacing:     {metrics['spacing']:.4f}")
print(f"|PF|:        {metrics['pareto_size']}")
```


---

## Performance Metrics

### Hypervolume (Hv)

Measures the volume of objective space dominated by the Pareto front relative to a reference point:

$$\text{Hv}(PF, \mathbf{r}) = \Lambda\!\left(\bigcup_{\mathbf{x} \in PF} \{\mathbf{y} \mid \mathbf{x} \prec \mathbf{y} \prec \mathbf{r}\}\right)$$

- **Captures both convergence and diversity** in a single scalar
- Higher is better; normalized to $[0, 1]$
- Reference point: component-wise worst observed values

### Spacing (SP)

Based on the Schott (1995) definition, SP measures the uniformity of spacing
between neighbouring solutions along a Pareto front:

$$SP = \sqrt{\frac{1}{|PF|} \sum_{i=1}^{|PF|} (d_i - \bar{d})^2}$$

where $d_i$ is the minimum **objective-wise Euclidean** distance from solution
$i$ to any other front member and $\bar{d}$ is the mean of all $d_i$ values.

- Lower is better (more uniform distribution)
- $SP = 0$ indicates perfectly uniform spacing
- Correct implementation normalizes distances by Pareto front bounds before
  computing $d_i$

---

## API Reference

### `KNSGAII`

```python
KNSGAII(
    instance: HHCInstance,
    population_size: int = 100,
    max_generations: int = 1000,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.2,
    random_state: Optional[int] = None
)
```

| Method | Returns | Description |
|---|---|---|
| `run(verbose=True)` | `List[Solution]` | Execute full 3-stage pipeline |
| `get_performance_metrics()` | `Dict` | Hv, SP, Pareto size, best F1/F2 |

### `Solution`

| Attribute | Type | Description |
|---|---|---|
| `routes` | `List[List[int]]` | Vehicle routes as customer ID sequences |
| `f1` | `float` | Objective 1: Total service time |
| `f2` | `float` | Objective 2: Total tardiness |
| `rank` | `int` | Non-domination rank |
| `crowding_distance` | `float` | Crowding distance |

### `ExperimentRunner`

```python
from src.experiment import ExperimentRunner

runner = ExperimentRunner(
    instances=['C101.25', 'C101.100', 'C107.100'],
    num_runs=30,
    population_size=100,
    max_generations=1000
)

results = runner.run()
runner.statistical_analysis()
runner.export_csv()
runner.export_latex_table()
```

---

## Authors

**Cheekati Sanjay Goud** — [@SanjayCheekati](https://github.com/SanjayCheekati)

**Maryala Harshitha** — [@Maryala-Harshitha58](https://github.com/Maryala-Harshitha58)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Cheekati Sanjay Goud, Maryala Harshitha
