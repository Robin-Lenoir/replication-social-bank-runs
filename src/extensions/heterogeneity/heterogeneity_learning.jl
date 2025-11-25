#=
heterogeneity_learning.jl

Learning dynamics solver for heterogeneous groups.
Implements SI network model with multiple learning rates βₖ.

Author: Robin Lenoir
=#

# Include required dependencies
using DifferentialEquations, Interpolations

# Include model definitions
include(joinpath(@__DIR__, "heterogeneity_model.jl"))

"""
    solve_SInetwork_hetero(params::LearningParametersHetero; tol=1e-6)

Solve SI model on network with heterogeneous groups.
Implements the ODE system: dG_k/dt = (1 - G_k) * β_k * ω(t)
where ω(t) = ∑_j dist_j * G_j is the average "infection" rate across all groups.

# Arguments
- `params::LearningParametersHetero`: Heterogeneous learning parameters

# Keyword Arguments  
- `tol::Float64 = eps()`: ODE solver tolerance - set to machine epsilon

# Returns
- `LearningResultsHetero`: Complete learning solution with CDFs/PDFs for each group

# Technical Implementation
- **ODE System**: Coupled differential equations for each group
- **Solver**: AutoTsit5(Rosenbrock23()) with adaptive timestepping
- **Grid**: Common adaptive grid inherited from ODE solver
- **PDF Computation**: Symbolic evaluation using ODE right-hand side

# Example
```julia
params = LearningParametersHetero(
    βs=[0.5, 1.0, 2.0], 
    dist=[0.3, 0.5, 0.2],
    tspan=(0.0, 30.0),
    x0=0.0001
)
lr_hetero = solve_SInetwork_hetero(params)
```
"""
function solve_SInetwork_hetero(params::LearningParametersHetero; tol=eps())
    solve_start = time()
    
    # Extract parameters
    (; βs, dist, tspan, x0) = params
    n_groups = length(βs)
    
    # Define the heterogeneous SI ODE system
    function ODE_SInetwork_hetero(du, I, p, t)
        βs, dist = p
        
        # Compute average infection rate across all groups
        ω = sum(dist[j] * I[j] for j in eachindex(I))
        
        # Update each group
        for k in eachindex(I)
            du[k] = (1 - I[k]) * βs[k] * ω
        end
    end
    
    # Set up initial conditions (same for all groups)
    I0 = [x0 for _ in 1:n_groups]
    
    # Define and solve the ODE problem
    prob = ODEProblem(ODE_SInetwork_hetero, I0, tspan, (βs, dist))
    sol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=tol, abstol=tol, verbose=false)
    
    # Extract common time grid from ODE solution
    grid = sol.t
    
    # Create learning CDFs for each group using LinearInterpolation
    learning_cdfs = []
    for k in 1:n_groups
        cdf_values = [sol.u[i][k] for i in 1:length(sol.u)]
        cdf_func = LinearInterpolation(grid, cdf_values)
        push!(learning_cdfs, cdf_func)
    end
    
    # Compute learning PDFs using symbolic evaluation of ODE right-hand side
    # Use helper function to avoid code duplication
    learning_pdfs = compute_pdf_hetero(βs, dist, learning_cdfs, grid)
    
    solve_time = time() - solve_start
    
    return LearningResultsHetero(params, learning_cdfs, learning_pdfs, grid, solve_time, sol)
end

"""
    compute_pdf_hetero(βs, learning_cdfs, t_values)

Compute PDFs for heterogeneous groups using symbolic evaluation.
Alternative implementation for cases where direct ODE evaluation is preferred.

# Arguments
- `βs::Vector{Float64}`: Learning rates for each group
- `learning_cdfs::Vector`: Learning CDF functions for each group
- `t_values::Vector{Float64}`: Time points for evaluation

# Returns
- `Vector`: Array of PDF functions for each group (LinearInterpolation objects)

# Technical Note
This is an alternative to the PDF computation in solve_SInetwork_hetero.
It can be useful for post-processing or when CDFs are available separately.
"""
function compute_pdf_hetero(βs, dist, learning_cdfs, t_values)
    n_groups = length(βs)
    learning_pdfs = []
    
    for k in 1:n_groups
        pdf_values = []
        for t in t_values
            # Evaluate all group states at time t
            I_t = [learning_cdfs[j](t) for j in 1:n_groups]
            # Compute average infection rate
            ω_t = sum(dist[j] * I_t[j] for j in 1:n_groups)
            # Evaluate PDF for group k
            pdf_k_t = (1 - I_t[k]) * βs[k] * ω_t
            push!(pdf_values, pdf_k_t)
        end
        pdf_func = LinearInterpolation(t_values, pdf_values)
        push!(learning_pdfs, pdf_func)
    end
    
    return learning_pdfs
end
