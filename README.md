# K-NSGA-II: Hybrid K-means + NSGA-II for Home Health Care Routing

Implementation of the K-NSGA-II algorithm for solving the Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows (HHC-MOVRPTW).

Based on the paper: "Multi-objective Evolutionary Approach based on K-means clustering for Home Health Care Routing and Scheduling Problem"

## Problem Description

**HHC-MOVRPTW** (Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows):
- A set of patients need to be visited by caregivers
- Each patient has time window preferences
- Caregivers start and end at a central depot

**Objectives (bi-objective optimization):**
1. **F1**: Minimize total service time (travel time + working time)
2. **F2**: Minimize total tardiness (deviation from patient time preferences)

## Algorithm: K-NSGA-II

The hybrid algorithm works in three stages:

### Stage 1: Decomposition (K-means Clustering)
- Divide patients into K clusters (K = number of caregivers)
- Uses geographical coordinates and time windows as features
- Balanced cluster sizes for fair workload distribution

### Stage 2: Optimization (NSGA-II per cluster)
- Run NSGA-II independently on each cluster
- Each cluster corresponds to one caregiver's route
- Produces K Pareto subsets

### Stage 3: Combination (Global Pareto Front)
- Combine K Pareto subsets
- Sum objective values across corresponding solutions
- Remove dominated solutions for final Pareto front

## Project Structure

```
Test 1/
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── datasets/              # Solomon VRPTW benchmark instances
│   ├── C_type/           # Clustered customers
│   ├── R_type/           # Random customers
│   └── RC_type/          # Mixed customers
└── src/
    ├── __init__.py        # Package initialization
    ├── data_parser.py     # Solomon file parser
    ├── problem.py         # HHC problem definition
    ├── kmeans.py          # K-means clustering
    ├── nsga2.py           # NSGA-II algorithm
    └── hybrid_knsga2.py   # K-NSGA-II hybrid model
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test
```bash
python main.py --test
```

### Single Instance
```bash
python main.py -i C101.25 -p 100 -g 500 -v
```

### Table 5 Replication (from paper)
```bash
python main.py --table5 -r 10
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i`, `--instance` | Instance name (e.g., C101.25) | None |
| `-p`, `--population` | Population size | 100 |
| `-g`, `--generations` | Maximum generations | 1000 |
| `-r`, `--runs` | Number of runs | 1 |
| `--table5` | Run Table 5 experiments | False |
| `--test` | Quick test mode | False |
| `-v`, `--verbose` | Verbose output | False |

## Available Instances

From Solomon VRPTW Benchmark:
- **C-type** (clustered): C101.25, C101.50, C101.100, C107.100, C206.50
- **R-type** (random): R109.25, R109.50, R109.100
- **RC-type** (mixed): RC106.25, RC106.50, RC106.100

## Example Output

```
K-NSGA-II RESULTS
============================================================

Global Pareto Front (6 solutions):
  1. F1=2613.41, F2=2475.22
  2. F1=2616.60, F2=1588.75
  3. F1=2617.78, F2=1501.04
  4. F1=2618.00, F2=1500.37
  5. F1=2619.32, F2=1499.16
  6. F1=2619.91, F2=1449.56

Timing:
  Decomposition: 0.03s
  Optimization:  0.81s
  Combination:   0.00s
  Total:         0.84s
```

## Performance Metrics

- **Pareto Size**: Number of non-dominated solutions
- **Spacing (SP)**: Distribution uniformity of Pareto front
- **Hypervolume (HV)**: Coverage of objective space

## References

1. Solomon, M.M. (1987). "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints"
2. Deb, K. et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
