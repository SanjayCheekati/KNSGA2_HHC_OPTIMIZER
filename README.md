<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-K--NSGA--II-2563EB?style=for-the-badge" alt="Algorithm">
  <img src="https://img.shields.io/badge/Domain-Healthcare-DC2626?style=for-the-badge&logo=heart&logoColor=white" alt="Domain">
  <img src="https://img.shields.io/badge/License-MIT-9333EA?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Dependencies-Zero-16A34A?style=for-the-badge" alt="Deps">
</p>

<h1 align="center">K-NSGA-II for Home Health Care Optimization</h1>

<p align="center">
  <b>A hybrid decomposition-based multi-objective evolutionary algorithm</b><br>
  <i>for the Home Health Care Vehicle Routing Problem with Time Windows (HHC-MOVRPTW)</i>
</p>

<p align="center">
  <code>K-means Decomposition</code> &#8594; <code>NSGA-II Optimization</code> &#8594; <code>Pareto Front Combination</code>
</p>

---

<br>

## Highlights

| | Feature | Detail |
|---|---|---|
| 🧬 | **Hybrid Algorithm** | K-means clustering decomposes NP-hard routing into tractable subproblems |
| 📊 | **Multi-Objective** | Simultaneously minimizes service time *and* patient tardiness |
| ⚡ | **Pure Python** | Zero external dependencies — runs on any Python 3.8+ installation |
| 🔬 | **Validated** | Tested on Solomon VRPTW benchmarks (C, R, RC types, 25–100 customers) |
| 📈 | **Metrics** | Hypervolume, Schott Spacing, Pareto front cardinality — all built-in |
| 🔁 | **Reproducible** | Deterministic seeding for fully repeatable experiments |

<br>

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SanjayCheekati/KNSGA2_HHC_OPTIMIZER.git
cd KNSGA2_HHC_OPTIMIZER

# Run interactively (no dependencies needed)
python main.py

# Or run a specific instance directly
python main.py C101.25
```

<br>

---

## Problem Formulation

Home Health Care (HHC) organizations assign caregivers to visit geographically distributed patients within preferred time windows, giving rise to a **bi-objective optimization** problem:

<table>
<tr>
<td width="50%">

**Objective 1 — Minimize Total Service Time**

$$F_1 = \sum_{k=1}^{K} \sum_{i=0}^{n_k} \left( t_{i,i+1}^k + s_i^k \right)$$

</td>
<td width="50%">

**Objective 2 — Minimize Total Tardiness**

$$F_2 = \sum_{i=1}^{N} \max(0,\; a_i - d_i)$$

</td>
</tr>
</table>

> Where $t_{i,j}^k$ = travel time on arc $(i,j)$ for vehicle $k$, $s_i^k$ = service duration, $a_i$ = arrival time, $d_i$ = due date.

### Constraints

| Constraint | Formulation |
|:---|:---|
| Vehicle capacity | $\sum_{i \in R_k} q_i \leq Q_k \quad \forall k$ |
| Time windows | $e_i \leq a_i \leq l_i \quad \forall i$ |
| Depot origin/return | Each route starts and ends at the depot |

### Pareto Dominance

Solution $\mathbf{x}$ dominates $\mathbf{y}$ ($\mathbf{x} \prec \mathbf{y}$) iff:

$$\forall\, i \in \{1,2\}:\; f_i(\mathbf{x}) \leq f_i(\mathbf{y}) \;\;\land\;\; \exists\, j \in \{1,2\}:\; f_j(\mathbf{x}) < f_j(\mathbf{y})$$

<br>

---

## Algorithm Design

### Three-Stage Pipeline

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  STAGE 1: DECOMPOSITION                                        │
 │  K-means++ partitions N patients into K geographic clusters     │
 │  Search space: O(N!) → K independent O((N/K)!) subproblems     │
 ├─────────────────────────────────────────────────────────────────┤
 │  STAGE 2: OPTIMIZATION                                         │
 │  NSGA-II runs independently on each cluster                    │
 │  Operators: OX crossover · swap/insert mutation                │
 │  Selection: Binary tournament on (rank, crowding distance)     │
 ├─────────────────────────────────────────────────────────────────┤
 │  STAGE 3: COMBINATION                                          │
 │  Cluster-level Pareto fronts merged combinatorially            │
 │  Non-dominated sorting extracts the global Pareto front        │
 └─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Method |
|:---|:---|
| **Initialization** | Random permutation encoding |
| **Crossover** | Order Crossover (OX), rate $P_c$ |
| **Mutation** | Swap + Insert, rate $P_m$ |
| **Selection** | Binary tournament (rank, then crowding) |
| **Ranking** | Fast non-dominated sorting, $O(MN^2)$ |
| **Diversity** | Crowding distance in objective space |
| **Elitism** | Parent + offspring merge, truncate to $N$ |

### K-means Clustering

- **Initialization**: K-means++ for well-spread centroids
- **Distance metric**: Euclidean on $(x, y)$ patient coordinates
- **K selection**: Experimentally optimized per instance type and size

<br>

---

## Benchmark Results

Validated on **Solomon VRPTW** instances across three spatial distributions:

| Instance | Type | Customers | K | Time Windows | Result |
|:---|:---|:---:|:---:|:---|:---:|
| C101.25 | Clustered | 25 | 2 | Tight | ✅ PASS |
| C101.100 | Clustered | 100 | 4 | Tight | ✅ PASS |
| C107.100 | Clustered | 100 | 4 | Varied | ✅ PASS |
| C206.50 | Clustered | 50 | 3 | Wide | ✅ PASS |
| R109.25 | Random | 25 | 2 | Tight | ✅ PASS |
| RC106.50 | Mixed | 50 | 3 | Mixed | ✅ PASS |

> All instances **meet or exceed published hypervolume targets** when evaluated over multiple seeds with $N = 100$, $G = 500$, $P_c = 0.7$, $P_m = 0.2$.

<br>

---

## Repository Structure

```
KNSGA2_HHC_OPTIMIZER/
│
├── main.py                     Entry point (interactive CLI + CLI args)
├── requirements.txt            Python version & dependency tracking
├── LICENSE                     MIT License
│
├── src/
│   ├── __init__.py             Package exports
│   ├── hybrid_knsga2.py        K-NSGA-II: 3-stage hybrid algorithm
│   ├── nsga2.py                NSGA-II engine (sorting, crowding, selection)
│   ├── kmeans.py               K-means++ clustering
│   ├── problem.py              HHCInstance, Customer, Solution definitions
│   ├── data_parser.py          Solomon-format instance parser
│   └── experiment.py           Statistical experiment runner
│
└── datasets/
    ├── C_type/                 Clustered distribution instances
    ├── R_type/                 Random distribution instances
    └── RC_type/                Mixed distribution instances
```

<br>

---

## Usage

### Interactive Mode

```bash
python main.py
```

Select a benchmark instance and choose a parameter preset (fast / standard / research) or enter custom values.

### Command-Line Interface

```bash
# Single instance with default parameters
python main.py C101.25

# Custom: instance, population, generations, runs
python main.py C101.100 100 500 5
```

### Programmatic API

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

### Batch Experiments

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

<br>

---

## Performance Metrics

### Hypervolume (Hv)

$$\text{Hv}(PF, \mathbf{r}) = \Lambda\!\left(\bigcup_{\mathbf{x} \in PF} \{\mathbf{y} \mid \mathbf{x} \prec \mathbf{y} \prec \mathbf{r}\}\right)$$

Measures the volume of objective space dominated by the Pareto front. **Higher is better**; normalized to $[0, 1]$.

### Spacing — Schott (1995)

$$SP = \sqrt{\frac{1}{|PF|} \sum_{i=1}^{|PF|} (d_i - \bar{d})^2}$$

where $d_i$ is the minimum objective-wise Euclidean distance from solution $i$ to any other front member, normalized by Pareto front bounds. **Lower is better** ($SP = 0$ means perfectly uniform spacing).

<br>

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
|:---|:---|:---|
| `run(verbose=True)` | `List[Solution]` | Execute full 3-stage pipeline |
| `get_performance_metrics()` | `Dict` | Hv, SP, Pareto size, best F1/F2 |

### `Solution`

| Attribute | Type | Description |
|:---|:---|:---|
| `routes` | `List[List[int]]` | Vehicle routes as customer ID sequences |
| `f1` | `float` | Objective 1: Total service time |
| `f2` | `float` | Objective 2: Total tardiness |
| `rank` | `int` | Non-domination rank |
| `crowding_distance` | `float` | Crowding distance |

<br>

---

## Authors

<table>
<tr>
<td align="center">
  <a href="https://github.com/SanjayCheekati">
    <b>Cheekati Sanjay Goud</b>
  </a>
  <br>
  <sub>@SanjayCheekati</sub>
</td>
<td align="center">
  <a href="https://github.com/Maryala-Harshitha58">
    <b>Maryala Harshitha</b>
  </a>
  <br>
  <sub>@Maryala-Harshitha58</sub>
</td>
</tr>
</table>

---

<p align="center">
  <b>MIT License</b> &middot; Copyright &copy; 2026 Cheekati Sanjay Goud, Maryala Harshitha
</p>
