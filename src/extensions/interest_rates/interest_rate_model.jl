#=
interest_rate_model.jl

Model structure definitions for the interest rates extension.
Handles positive interest rates with HJB value functions and reentry dynamics.

Author: Robin Lenoir
=#

# Include baseline model definitions
include(joinpath(@__DIR__, "..", "..", "baseline", "model.jl"))

"""
    EconomicParametersInterest

Economic parameters for the interest rate extension, including positive interest rates
and recovery dynamics. Extends baseline economic parameters with r and δ.

# Fields
- All baseline parameters: `u`, `p`, `κ`, `λ`, `η_bar`, `η`
- Interest rate specific:
  - `r::Float64`: Interest rate (r ≥ 0, typically r < δ for convergence)
  - `δ::Float64`: Recovery/discount rate (δ > 0, typically δ > r)
"""
struct EconomicParametersInterest
    # Baseline parameters
    u::Float64      # Utility flow from deposits
    p::Float64      # Prior probability bank is fragile
    κ::Float64      # Solvency threshold
    λ::Float64      # Exponential rate for t₀
    η_bar::Float64  # Raw awareness window
    η::Float64      # Normalized awareness window
    
    # Interest rate specific  
    r::Float64      # Interest rate
    δ::Float64      # Bank maturity rate
    
    function EconomicParametersInterest(u, p, κ, λ, η_bar, η, r, δ)
        # Validation (baseline parameters)
        u ≥ 0 || throw(ArgumentError("Utility flow u must be non-negative, got u = $u"))
        0 ≤ p ≤ 1 || throw(ArgumentError("Prior probability p must be in [0,1], got p = $p"))
        0 < κ < 1 || throw(ArgumentError("Solvency threshold κ must be in (0,1), got κ = $κ"))
        λ > 0 || throw(ArgumentError("Exponential rate λ must be positive, got λ = $λ"))
        η_bar > 0 || throw(ArgumentError("Raw awareness window η_bar must be positive, got η_bar = $η_bar"))
        η > 0 || throw(ArgumentError("Normalized awareness window η must be positive, got η = $η"))
        
        # Validation (interest rate parameters)
        r ≥ 0 || throw(ArgumentError("Interest rate r must be non-negative, got r = $r"))
        δ > 0 || throw(ArgumentError("Recovery rate δ must be positive, got δ = $δ"))
        r < δ || throw(ArgumentError("Interest rate r must be less than recovery rate δ for convergence, got r = $r, δ = $δ"))
        
        new(u, p, κ, λ, η_bar, η, r, δ)
    end
end

# Keyword constructor for EconomicParametersInterest
function EconomicParametersInterest(; u::Float64, p::Float64, κ::Float64, λ::Float64, 
                                   η_bar::Float64, η::Float64, r::Float64, δ::Float64)
    return EconomicParametersInterest(u, p, κ, λ, η_bar, η, r, δ)
end

"""
    ModelParametersInterest

Master struct for the interest rate extension model.
Combines baseline learning parameters with interest rate economic parameters.

# Fields
- `learning::LearningParameters`: Baseline learning parameters (reused)
- `economic::EconomicParametersInterest`: Extended economic parameters with r, δ

# Usage
```julia
# Create interest rate model
m_interest = ModelParametersInterest(β=1.0, r=0.02, δ=0.1, u=0.1)

# Staged computation
lr = solve_learning(m_interest.learning)
result = solve_equilibrium_interest(lr, m_interest.economic, m_interest)
```
"""
struct ModelParametersInterest
    learning::LearningParameters
    economic::EconomicParametersInterest

    function ModelParametersInterest(learning::LearningParameters, economic::EconomicParametersInterest)
        new(learning, economic)
    end
end

"""
    ModelParametersInterest(; kwargs...)

Convenience constructor with keyword arguments and defaults.

# Interest Rate Specific Parameters
- `r = 0.0`: Interest rate (set to positive value to enable interest rate extension)
- `δ = 0.1`: Recovery/discount rate (must be > r)

# Baseline Parameters (same defaults as baseline model)
- `β = 1.0`: Communication speed
- `η = nothing`: Normalized awareness window (computed from η_bar if not provided)
- `η_bar = 15.0`: Raw awareness window
- `u = 0.1`: Utility flow
- `p = 0.5`: Prior fragility probability  
- `κ = 0.6`: Solvency threshold
- `λ = 0.01`: Exponential rate
- `tspan = (0.0, 2*η)`: Time span
- `x0 = 0.0001`: Initial learning condition

# Examples
```julia
# Interest rate model with r > 0
m = ModelParametersInterest(β=1.0, r=0.02, δ=0.1)

# Custom parameters
m = ModelParametersInterest(β=2.0, r=0.05, δ=0.15, u=0.05, κ=0.4)
```
"""
function ModelParametersInterest(; 
    β = 1.0,
    η = nothing,
    η_bar = 15.0,
    u = 0.1,
    p = 0.5,
    κ = 0.6,
    λ = 0.01,
    r = 0.0,
    δ = 0.1,
    tspan = nothing,
    x0 = 0.0001
)
    # Compute η from η_bar if not provided
    if η === nothing
        η = η_bar / β
    end

    # Set default tspan based on computed η if not provided
    if tspan === nothing
        tspan = (0.0, 2*η)
    end

    # Build substructs with resolved values
    learning = LearningParameters(β, tspan, x0)
    economic = EconomicParametersInterest(u, p, κ, λ, η_bar, η, r, δ)
    
    return ModelParametersInterest(learning, economic)
end

"""
    ModelParametersInterest(base::ModelParametersInterest; kwargs...)

Create new ModelParametersInterest by modifying an existing one.

# Example
```julia
baseline_interest = ModelParametersInterest(r=0.02, δ=0.1)
high_rate = ModelParametersInterest(baseline_interest; r=0.05, δ=0.15)
```
"""
function ModelParametersInterest(base::ModelParametersInterest; kwargs...)
    # Extract current values from substructs
    current = (
        β = base.learning.β,
        η = base.economic.η,
        η_bar = base.economic.η_bar,
        u = base.economic.u,
        p = base.economic.p,
        κ = base.economic.κ,
        λ = base.economic.λ,
        r = base.economic.r,
        δ = base.economic.δ,
        tspan = base.learning.tspan,
        x0 = base.learning.x0
    )
    
    # Merge with new values
    merged = merge(current, kwargs)
    
    # Use keyword constructor to handle interdependencies
    return ModelParametersInterest(
        β=merged.β, η=merged.η, η_bar=merged.η_bar, u=merged.u, p=merged.p, 
        κ=merged.κ, λ=merged.λ, r=merged.r, δ=merged.δ, tspan=merged.tspan, x0=merged.x0
    )
end

"""
    SolvedModelInterest

Complete solution of the interest rate extension model.
Extends baseline solution to include value function results.

# Fields (extending baseline SolvedModel)
- Core equilibrium outputs: `ξ`, `τ_bar_IN_UNC`, `τ_bar_OUT_UNC`, `HR`, `bankrun`
- Value function output: `V` (extrapolate object when r > 0, nothing when r = 0)
- Derived quantities: `τ_IN`, `τ_OUT`
- Original inputs: `model_params`, `learning_results`
- Solution metadata: `converged`, `solve_time`, `tolerance`
"""
struct SolvedModelInterest
    # Core equilibrium outputs
    ξ::Float64
    τ_bar_IN_UNC::Float64
    τ_bar_OUT_UNC::Float64
    HR::Any  # LinearInterpolation object
    bankrun::Bool

    # Value function output (when r > 0)
    V::Any  # Value function extrapolate object (or nothing when r = 0)

    # Derived quantities
    τ_IN::Float64
    τ_OUT::Float64

    # Original inputs
    model_params::ModelParametersInterest
    learning_results::LearningResults

    # Solution metadata
    converged::Bool
    solve_time::Float64
    tolerance::Float64

    # Computation cache for AW functions
    aw::Ref{Union{Nothing, NamedTuple}}

    function SolvedModelInterest(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun, V,
                                model_params::ModelParametersInterest, learning_results,
                                converged, solve_time, tolerance)
        # Compute derived quantities
        τ_IN = max(ξ - τ_bar_IN_UNC, 0)
        τ_OUT = max(ξ - τ_bar_OUT_UNC, 0)

        # Validation
        (ξ ≥ 0 || isnan(ξ)) || throw(ArgumentError("Crash time ξ must be non-negative or NaN, got ξ = $ξ"))
        τ_bar_IN_UNC ≥ 0 || throw(ArgumentError("τ_bar_IN_UNC must be non-negative, got $τ_bar_IN_UNC"))
        τ_bar_OUT_UNC ≥ 0 || throw(ArgumentError("τ_bar_OUT_UNC must be non-negative, got $τ_bar_OUT_UNC"))
        solve_time ≥ 0 || throw(ArgumentError("Solve time must be non-negative, got $solve_time"))
        tolerance ≥ 0 || throw(ArgumentError("Tolerance must be non-negative, got $tolerance"))

        new(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun, V, τ_IN, τ_OUT,
            model_params, learning_results, converged, solve_time, tolerance,
            Ref{Union{Nothing, NamedTuple}}(nothing))
    end
end

# Display functions
function Base.show(io::IO, params::EconomicParametersInterest)
    print(io, "EconomicParametersInterest(\n")
    print(io, "  Fundamentals: u=$(params.u), p=$(params.p), κ=$(params.κ)\n")
    print(io, "  Informational: λ=$(params.λ), η_bar=$(params.η_bar), η=$(params.η)\n")
    print(io, "  Interest Rates: r=$(params.r), δ=$(params.δ)\n")
    print(io, ")")
end

function Base.show(io::IO, model::ModelParametersInterest)
    print(io, "ModelParametersInterest(\n")
    print(io, "  Learning: β=$(model.learning.β), tspan=$(model.learning.tspan), x0=$(model.learning.x0)\n")
    print(io, "  Economic: u=$(model.economic.u), p=$(model.economic.p), κ=$(model.economic.κ), λ=$(model.economic.λ)\n") 
    print(io, "  Interest: r=$(model.economic.r), δ=$(model.economic.δ)\n")
    print(io, "  Awareness: η_bar=$(model.economic.η_bar), η=$(model.economic.η)\n")
    print(io, ")")
end

function Base.show(io::IO, result::SolvedModelInterest)
    print(io, "SolvedModelInterest(\n")
    print(io, "  Equilibrium: ξ=$(result.ξ), bankrun=$(result.bankrun)\n")
    print(io, "  Buffers: τ_bar_IN=$(result.τ_bar_IN_UNC), τ_bar_OUT=$(result.τ_bar_OUT_UNC)\n")
    print(io, "  Derived: τ_IN=$(result.τ_IN), τ_OUT=$(result.τ_OUT)\n")
    print(io, "  Interest: r=$(result.model_params.economic.r), δ=$(result.model_params.economic.δ)\n")
    print(io, "  Value function: $(result.V !== nothing ? "computed" : "not computed")\n")
    print(io, "  Solution: converged=$(result.converged), time=$(round(result.solve_time*1000, digits=1))ms\n")
    print(io, ")")
end
