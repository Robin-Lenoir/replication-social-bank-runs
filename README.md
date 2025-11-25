# Replication Package: The Social Determinants of Bank Runs

**Author:** Robin Lenoir
**Institution:** New York University
**Email:** lenoir.robin@gmail.com

## Citation

Lenoir, R. (2025). The Social Determinants of Bank Runs. Working Paper. 

## Overview

This package contains the complete Julia implementation for all numerical results and figures in the paper "The Social Determinants of Bank Runs." The code implements a staged computational approach to solve equilibrium bank run models with social learning dynamics. See Appendix C for a description of the computational methods implemented here.  

## Quick Start

### 1. Set Up and Install Dependencies

```bash
cd replication_package/computation
julia --project=. -e 'using Pkg; Pkg.instantiate()'

This installs all required packages from `Project.toml`.

### 2. Run Complete Replication

Launch Julia and run the master script:

```bash
julia --project=.
```

Then in the Julia REPL:
```julia
include("MASTER.jl")
```

This single command will:
1. Run all baseline model computations (8 figures)
2. Run all extension computations (5 figures)
3. Generate all figure PDFs in `../output/figures/`

**Estimated runtime:** 5-15 minutes (depending on system)

### 3. (Optional) View Compiled Figures Document

You may compile a LaTeX document containing all figures:

```bash
cd ../output
pdflatex replication_figures.tex
```

This creates `replication_figures.pdf` with all figures organized by section.

## Package Structure

### Core Implementation
```
src/
├── baseline/                    # Core baseline model
│   ├── model.jl                # Parameter structures
│   ├── learning.jl             # Learning dynamics solver (ODE)
│   ├── solver.jl               # Equilibrium solver (bisection)
│   └── plotting.jl             # Visualization functions
│
└── extensions/                  # Model extensions
    ├── heterogeneity/          # Heterogeneous learning speeds
    ├── interest_rates/         # Positive interest rates (r > 0)
    └── social_learning/        # Endogenous learning from actions
```

### Replication Scripts
```
scripts/
├── 1_baseline.jl               # Baseline model (Figures 1-5)
├── 2_heterogeneity.jl         # Extension 1
├── 3_interest_rates.jl        # Extension 2
└── 4_social_learning.jl       # Extension 3
```

### Outputs
```
../output/                      # Parent-level output directory
├── figures/                    # All generated PDFs
│   ├── baseline/              # Main paper figures
│   ├── heterogeneity/
│   ├── interest_rates/
│   └── social_learning/
│
└── replication_figures.tex    # Auto-generated LaTeX document
```

**Note:** Figures are output to the parent `replication_package/output/` directory to integrate with the complete replication package structure.

## Generated Figures

### Baseline Model (8 figures)
- **Figure 1:** Learning dynamics for different communication speeds
- **Figure 2:** Hazard rate decomposition
- **Figure 3:** Equilibrium dynamics (main scenario)
  - 3bis: Fast communication variant
  - 3ter: Low deposit utility variant
- **Figure 4:** Comparative statics in deposit utility (2 panels)
- **Figure 5:** β-u interaction heatmap (peak withdrawals)

### Extensions (5 figures)
- **Extension 1 - Heterogeneity:** Aggregate withdrawals with heterogeneous groups
- **Extension 2 - Interest Rates:** Value function + hazard decomposition (2 figures)
- **Extension 3 - Social Learning:** Social learning equilibrium + baseline comparison (2 figures)

## Computational Methods

For detailed computational methods and mathematical derivations, see:
- **Paper Appendix C: Computational Methods**
- **Local copy:** `docs/appendix_c.pdf` (included in this package)


## Dependencies

The package requires the following Julia packages:
- `DifferentialEquations` - ODE solvers for learning dynamics
- `Interpolations` - Linear interpolation for continuous functions
- `Plots` - Plotting backend
- `PGFPlotsX` - Publication-quality LaTeX plots
- `LaTeXStrings` - LaTeX formatting in plots
- `Statistics` - Statistical utilities
- `LinearAlgebra` - Matrix operations (social learning extension)

All dependencies are automatically installed via `Project.toml`.


## License

This replication package is provided for academic use. Please cite the paper when using this code.

## Contact

For questions or issues with the replication package:
- **Email:** lenoir.robin@gmail.com
- **Issues:** Please report any bugs or difficulties

---

*Last updated: November 2025*
