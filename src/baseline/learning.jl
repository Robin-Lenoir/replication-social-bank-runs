### Learning Dynamics Functions ###
# This file contains functions for solving learning ODEs
# Part of the Social Bank Runs replication package
# Author: Robin Lenoir

#########################################
# BASELINE LEARNING
#########################################
"""
    solve_SIhomogeneous(learning_params::LearningParameters; tol=nothing)
    solve_SIhomogeneous(model::ModelParameters; tol=nothing)

Solve SI model with homogeneous population: dx/dt = βx(1 - x)
This is the main learning dynamics function used throughout the Social Bank Runs model.

# Technical Implementation
- **ODE Solver**: AutoTsit5(Rosenbrock23()) - adaptive algorithm robust for stiff/non-stiff problems
- **Tolerance**: Default machine epsilon (eps()) for high accuracy
- **Grid**: Adaptive grid from ODE solver, optimized for solution curvature

# Arguments
- `learning_params`: LearningParameters struct containing learning dynamics parameters
- `model`: ModelParameters struct (convenience method, extracts learning parameters)
- `tol`: Optional tolerance override (default: machine epsilon)

# Returns
- `learning_cdf`: Interpolated CDF function G(t) representing fraction of informed agents
- `sol`: ODE solution object with full solution trajectory

# Example
```julia
# Primary usage
learning_params = LearningParameters(β=1.0, tspan=(0.0, 20.0), x0=[0.001])
learning_cdf, sol = solve_SIhomogeneous(learning_params)

# Convenience usage
params = ModelParameters(β=1.0, η_bar=15.0)
learning_cdf, sol = solve_SIhomogeneous(params)
```
"""
function solve_SIhomogeneous(learning_params::LearningParameters; tol=nothing)
    (; β, tspan, x0) = learning_params
    tol = tol === nothing ? eps() : tol

    function simpleODE_forSI!(dx, x, p, t)
        β = p
        dx[1] = β * x[1] * (1 - x[1])
    end

    prob = ODEProblem(simpleODE_forSI!, [x0], tspan, β)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=tol, abstol=tol, verbose=false)
    learning_cdf = LinearInterpolation(sol.t, [sol.u[i][1] for i in 1:length(sol.u)])
    return learning_cdf, sol
end

#########################################
# LEARNING RESULTS STRUCT
#########################################

"""
    LearningResults

Define structure of results from solving the learning dynamics stage of the Social Bank Runs model.
Contains all learning-related quantities computed from LearningParameters.

# Fields
- `params::LearningParameters`: Learning parameters used in computation
- `learning_cdf::LinearInterpolation`: Learning CDF function G(t)
- `learning_pdf::LinearInterpolation`: Learning PDF function g(t) = G'(t)
- `grid::Vector{Float64}`: Time grid from adaptive ODE solver
- `solve_time::Float64`: Computation time in seconds
- `ode_solution::ODESolution`: Full ODE solution object for advanced use
"""
struct LearningResults
    params::LearningParameters
    learning_cdf::Any  # Interpolation object (specific type varies)
    learning_pdf::Any  # Interpolation object  
    grid::Vector{Float64}
    solve_time::Float64
    ode_solution::ODESolution  # ODE solution from DifferentialEquations.jl
end

"""
    solve_learning(learning_params::LearningParameters; tol=nothing)
    solve_learning(m::ModelParameters; tol=nothing)

Solve the learning dynamics stage for given learning parameters.
This is the main constructor for LearningResults.

# Arguments
- `learning_params`: LearningParameters struct containing learning dynamics parameters
- `m`: ModelParameters struct (convenience method, extracts learning parameters)
- `tol`: Optional tolerance override (default: machine epsilon)

# Returns
- `LearningResults`: Complete learning stage solution

# Example
```julia
# Primary usage
learning_params = LearningParameters(β=2.0, tspan=(0.0, 30.0), x0=[0.001])
lr = solve_learning(learning_params)

# Convenience usage
m = ModelParameters(β=2.0, η_bar=30.0)  # η = 30.0/2.0 = 15.0
lr = solve_learning(m)
```
"""
function solve_learning(learning_params::LearningParameters; tol=nothing)
    solve_start = time()

    # Solve learning ODE using existing function
    learning_cdf, ode_sol = solve_SIhomogeneous(learning_params, tol=tol)

    # Compute PDF using existing function
    learning_pdf = compute_pdf_symbolic_baseline(learning_params.β, learning_cdf)

    # Extract time grid from ODE solution
    grid = ode_sol.t

    solve_time = time() - solve_start

    return LearningResults(learning_params, learning_cdf, learning_pdf, grid, solve_time, ode_sol)
end

"""
    show(io::IO, lr::LearningResults)

Display LearningResults in a readable format.
"""
function Base.show(io::IO, lr::LearningResults)
    print(io, "LearningResults(\n")
    print(io, "  Learning: β=$(lr.params.β), tspan=$(lr.params.tspan), x0=$(lr.params.x0)\n")
    print(io, "  Grid: $(length(lr.grid)) points from $(lr.grid[1]) to $(lr.grid[end])\n")
    print(io, "  Solve time: $(round(lr.solve_time*1000, digits=2)) ms\n")
    print(io, "  ODE status: $(lr.ode_solution.retcode)\n")
    print(io, ")")
end

#########################################
# HELPER FUNCTIONS
#########################################
"""
    compute_pdf_symbolic_baseline(β, learning_cdf, t_values=nothing)

Compute PDF symbolically using the ODE formula for homogeneous SI model.
Uses solved learning_cdf's underlying grid by default for consistency.

# Grid Strategy
- **Adaptive Grid**: Inherits time points from ODE solver's adaptive grid for consistency
- **Formula**: g(t) = β * G(t) * (1 - G(t)) from SI model dynamics, G being the learning_cdf

# Arguments
- `β`: Learning rate
- `learning_cdf`: CDF function (interpolation)
- `t_values`: Time points for evaluation (optional, uses learning_cdf's grid if not provided)

# Returns
- `pdf_func`: Interpolated PDF function
"""
function compute_pdf_symbolic_baseline(β, learning_cdf, t_values=nothing)
    # Use learning_cdf's underlying grid if t_values not provided
    if isnothing(t_values)
        t_values = learning_cdf.itp.knots[1]
    end

    # For homogeneous SI model: g(t) = β * G(t) * (1 - G(t))
    # Vectorized computation for speed
    G_vals = learning_cdf.(t_values)
    pdf_values = β .* G_vals .* (1 .- G_vals)

    return LinearInterpolation(t_values, pdf_values)
end
