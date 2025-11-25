#=
social_learning_dynamics.jl

Learning dynamics for the social learning extension.
Implements learning from aggregate withdrawals rather than word-of-mouth.

Author: Robin Lenoir
=#

# Include required dependencies
using DifferentialEquations, Interpolations

# Include baseline model definitions
include(joinpath(@__DIR__, "..", "..", "baseline", "model.jl"))

"""
    solve_ODE_social_learning(β, AW_cum, I0, tspan; tol=1e-12)

Solve homogeneous SI learning model where agents learn from observing aggregate withdrawals.

# Mathematical Framework
In the social learning extension, agents learn not from word-of-mouth communication
but from observing the aggregate withdrawal rate AW_cum(t). The learning dynamics become:

dG/dt = (1 - G) * β * AW_cum(t)

where:
- G(t): Fraction of agents that has learned by time t
- β: Learning rate (sensitivity to withdrawal observations)
- AW_cum(t): Aggregate withdrawal rate function (exogenous to this ODE)

# Arguments
- `β::Float64`: Learning rate
- `AW_cum`: Aggregate withdrawal function (LinearInterpolation object)
- `I0::Float64`: Initial learning state
- `tspan::Tuple{Float64, Float64}`: Time span for integration

# Keyword Arguments
- `tol::Float64 = eps()`: ODE solver tolerance (high precision for stability)

# Returns
- `learning_cdf`: Learning CDF function
- `sol`: ODE solution object

# Technical Implementation
- **Input Function**: AW_cum(t) treated as external forcing
- **Solver**: AutoTsit5(Rosenbrock23()) with high precision for numerical stability
- **Grid**: Adaptive grid from ODE solver, inherited by interpolation

# Example
```julia
# AW_cum from previous iteration or initial guess
β = 1.0
I0 = 0.0001
learning_cdf, sol = solve_ODE_social_learning(β, AW_cum, I0, (0.0, 30.0))
```
"""
function solve_ODE_social_learning(β, AW_cum, I0, tspan; tol=eps())

    # Define the social learning ODE system
    function ODE_social_learning!(du, I, p, t)
        β, AW_cum = p

        # Evaluate aggregate withdrawals at current time
        AW_t = AW_cum(t)
        du[1] = (1 - I[1]) * β * AW_t
    end

    # Set up and solve ODE problem
    prob = ODEProblem(ODE_social_learning!, [I0], tspan, (β, AW_cum))
    sol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=tol, abstol=tol, verbose=false)

    # Create learning CDF
    cdf_values = [sol.u[i][1] for i in 1:length(sol.u)]
    learning_cdf = LinearInterpolation(sol.t, cdf_values)

    return learning_cdf, sol
end

"""
    compute_pdf_social_learning(β, learning_cdf, AW_cum, t_values)

Compute PDF for homogeneous social learning model using symbolic evaluation.

# Mathematical Framework
For social learning, the PDF is given by:
g(t) = dG/dt = (1 - G(t)) * β * AW_cum(t)

# Arguments
- `β::Float64`: Learning rate
- `learning_cdf`: Learning CDF function
- `AW_cum`: Aggregate withdrawal function
- `t_values::Vector{Float64}`: Time points for evaluation

# Returns
- `learning_pdf`: PDF function (LinearInterpolation object)
"""
function compute_pdf_social_learning(β, learning_cdf, AW_cum, t_values)
    pdf_values = []

    for t in t_values
        # Evaluate learning state and withdrawal rate at time t
        G_t = learning_cdf(t)
        AW_t = AW_cum(t)

        # Compute PDF using ODE formula
        pdf_t = (1 - G_t) * β * AW_t
        push!(pdf_values, pdf_t)
    end

    learning_pdf = LinearInterpolation(t_values, pdf_values)

    return learning_pdf
end

"""
    LearningResultsSocial

Results structure for homogeneous social learning extension.
Similar to baseline LearningResults but tracks the withdrawal feedback function.

# Fields
- `params`: Learning parameters
- `learning_cdf`: Learning CDF function
- `learning_pdf`: Learning PDF function
- `grid::Vector{Float64}`: Time grid
- `AW_cum`: Aggregate withdrawal function that drove the learning
- `solve_time::Float64`: Total computation time
- `iterations::Int`: Number of fixed-point iterations
- `converged::Bool`: Whether fixed-point iteration converged
"""
struct LearningResultsSocial
    params                          # Learning parameters (flexible type)
    learning_cdf::Any               # Learning CDF function
    learning_pdf::Any               # Learning PDF function
    grid::Vector{Float64}           # Time grid
    AW_cum::Any                     # Aggregate withdrawal function
    solve_time::Float64             # Total computation time
    iterations::Int                 # Fixed-point iterations
    converged::Bool                 # Convergence status

    function LearningResultsSocial(params, learning_cdf, learning_pdf, grid, AW_cum,
                                  solve_time, iterations, converged)
        new(params, learning_cdf, learning_pdf, grid, AW_cum, solve_time, iterations, converged)
    end
end

# Display function for social learning results
function Base.show(io::IO, lr::LearningResultsSocial)
    println(io, "LearningResultsSocial(")
    println(io, "  Grid size: $(length(lr.grid))")
    println(io, "  Iterations: $(lr.iterations)")
    println(io, "  Converged: $(lr.converged)")
    println(io, "  Solve time: $(round(lr.solve_time*1000, digits=1))ms")
    print(io, ")")
end
