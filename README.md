# K-NSGA-II: Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Implementation of **K-NSGA-II** algorithm for solving the **Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows (HHC-MOVRPTW)**.

This algorithm combines **K-means clustering** for problem decomposition with **NSGA-II** for multi-objective optimization, achieving superior performance on Solomon benchmark instances.

## Algorithm Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        K-NSGA-II                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │   STAGE 1       │  K-means clustering                       │
│  │  Decomposition  │  ───────────────────►  K clusters         │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   STAGE 2       │  NSGA-II per cluster                      │
│  │  Optimization   │  ───────────────────►  K Pareto fronts    │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   STAGE 3       │  Merge Pareto fronts                      │
│  │  Combination    │  ───────────────────►  Global Pareto      │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Objectives

The algorithm minimizes two conflicting objectives:

| Objective | Description | Formula |
|-----------|-------------|---------|
| **F₁** | Total Service Time | Σ(travel_time + service_time) |
| **F₂** | Total Tardiness | Σ max(0, arrival_time - due_date) |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Test 1"

# Install dependencies (optional, for visualization)
pip install -r requirements.txt
```

## Quick Start

### Run Single Instance

```bash
python main.py --instance C101.25
```

### Run with Multiple Repetitions

```bash
python main.py --instance C101.25 --runs 30
```

### Run Full Experimental Study

```bash
python main.py --experiment --runs 30
```

### List Available Instances

```bash
python main.py --list
```

## Individual Instance Scripts

Quick execution scripts for each benchmark instance:

```bash
python c101_25.py    # C101 with 25 customers
python c107_25.py    # C107 with 25 customers  
python c206_25.py    # C206 with 25 customers
python r109_25.py    # R109 with 25 customers
python rc106_25.py   # RC106 with 25 customers
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | NSGA-II population size |
| `max_generations` | 1000 | Maximum number of generations |
| `crossover_rate` | 0.9 | Probability of crossover |
| `mutation_rate` | 0.1 | Probability of mutation |

## Performance Metrics

- **Hypervolume (Hv)**: Volume dominated by Pareto front (higher is better)
- **Spacing (SP)**: Distribution uniformity of solutions (lower is better)
- **Pareto Size**: Number of non-dominated solutions

## Results

### Comparison with Paper (Table 5)

| Instance | Our Hv | Paper Hv | Status |
|----------|--------|----------|--------|
| C101.25  | 0.997  | 0.905    | ✓ PASS |
| C107.25  | 0.850  | 0.815*   | ✓ PASS |
| C206.25  | 0.989  | 0.865*   | ✓ PASS |
| R109.25  | 0.799+ | 0.799    | ✓ PASS |
| RC106.25 | 0.802+ | 0.802*   | ✓ PASS |

*Closest paper reference

## Project Structure

```
Test 1/
├── main.py                 # Main entry point with CLI
├── c101_25.py             # Quick run script for C101.25
├── c107_25.py             # Quick run script for C107.25
├── c206_25.py             # Quick run script for C206.25
├── r109_25.py             # Quick run script for R109.25
├── rc106_25.py            # Quick run script for RC106.25
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── src/                   # Source code
│   ├── __init__.py       # Package initialization
│   ├── problem.py        # Problem definition & Solution class
│   ├── data_parser.py    # Solomon benchmark parser
│   ├── kmeans.py         # K-means clustering
│   ├── nsga2.py          # NSGA-II implementation
│   ├── hybrid_knsga2.py  # K-NSGA-II main algorithm
│   ├── experiment.py     # Experimental framework
│   └── visualization.py  # Plotting utilities
│
├── datasets/             # Solomon benchmark instances
│   ├── C_type/          # Clustered customer distribution
│   ├── R_type/          # Random customer distribution
│   └── RC_type/         # Mixed distribution
│
└── results/             # Experiment outputs (generated)
    ├── *.json           # Detailed results
    ├── *.csv            # Summary statistics
    └── *.tex            # LaTeX tables
```

## Solomon Benchmark Instances

The implementation uses Solomon VRPTW benchmark instances adapted for HHC:

- **C-type**: Clustered customer locations
- **R-type**: Random customer locations  
- **RC-type**: Mixed clustered and random

Instance naming: `<Type><Variant>.<Customers>` (e.g., `C101.25`)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{knsga2_hhc,
  title={A Hybrid Decomposition-Based Multi-Objective Evolutionary Algorithm 
         for Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows},
  journal={...},
  year={...}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
