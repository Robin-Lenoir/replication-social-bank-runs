#=
baseline_model.jl

Core baseline model structure definitions for the Social Bank Runs project.
Contains the original ModelParameters struct and related utilities.

Author: Robin Lenoir
=#

######################################################
###### Model declarations ############################
######################################################
"""
    LearningParameters

Pure learning dynamics parameters needed to solve the learning CDF.
These parameters directly affect the learning ODE.

# Fields
- `β::Float64`: Communication speed / learning rate (β > 0)
- `tspan::Tuple{Float64,Float64}`: Time span for learning dynamics simulation
- `x0::Float64`: Initial condition for learning ODE (typically 0.0001)
"""
struct LearningParameters
    β::Float64                      # Communication speed
    tspan::Tuple{Float64, Float64}  # Time span for learning dynamics
    x0::Float64                     # Initial condition for learning ODE

    function LearningParameters(β, tspan, x0)
        # Validation
        β > 0 || throw(ArgumentError("Communication speed β must be positive, got β = $β"))
        length(tspan) == 2 || throw(ArgumentError("Time span tspan must be a tuple of length 2"))
        tspan[1] ≥ 0 || throw(ArgumentError("Start time must be non-negative, got tspan[1] = $(tspan[1])"))
        tspan[2] > tspan[1] || throw(ArgumentError("End time must be greater than start time, got tspan = $tspan"))
        x0 ≥ 0 || throw(ArgumentError("Initial condition x0 must be non-negative, got x0 = $x0"))

        new(β, tspan, x0)
    end
end

# Keyword constructor for LearningParameters
function LearningParameters(; β::Float64, tspan::Tuple{Float64, Float64}, x0::Float64)
    return LearningParameters(β, tspan, x0)
end

"""
    EconomicParameters

Economic fundamentals and informational parameters used to interpret the learning CDF.
These parameters are used in hazard rate computation and equilibrium solving.
Note: we declate the normalized η here but it will typically be computed automatically.

# Fields
- `u::Float64`: Utility flow from deposits (u ≥ 0)
- `p::Float64`: Prior probability bank is fragile (p ∈ [0,1])
- `κ::Float64`: Solvency threshold - withdrawal fraction causing collapse (κ ∈ (0,1))
- `λ::Float64`: Exponential rate for t₀ arrival (λ > 0)
- `η_bar::Float64`: Raw awareness window before normalization (η_bar > 0)
- `η::Float64`: Normalized awareness window (η > 0, typically η = η_bar/β)
"""
struct EconomicParameters
    u::Float64      # Utility flow from deposits
    p::Float64      # Prior probability bank is fragile
    κ::Float64      # Solvency threshold
    λ::Float64      # Exponential rate for t₀
    η_bar::Float64  # Raw awareness window
    η::Float64      # Normalized awareness window
    
    function EconomicParameters(u, p, κ, λ, η_bar, η)
        # Validation
        u ≥ 0 || throw(ArgumentError("Utility flow u must be non-negative, got u = $u"))
        0 ≤ p ≤ 1 || throw(ArgumentError("Prior probability p must be in [0,1], got p = $p"))
        0 < κ < 1 || throw(ArgumentError("Solvency threshold κ must be in (0,1), got κ = $κ"))
        λ > 0 || throw(ArgumentError("Exponential rate λ must be positive, got λ = $λ"))
        η_bar > 0 || throw(ArgumentError("Raw awareness window η_bar must be positive, got η_bar = $η_bar"))
        η > 0 || throw(ArgumentError("Normalized awareness window η must be positive, got η = $η"))
        
        new(u, p, κ, λ, η_bar, η)
    end
end

# Keyword constructor for EconomicParameters
function EconomicParameters(; u::Float64, p::Float64, κ::Float64, λ::Float64, η_bar::Float64, η::Float64)
    return EconomicParameters(u, p, κ, λ, η_bar, η)
end

"""
    ModelParameters

Master struct containing all parameters for the baseline Social Bank Runs model.
Organized with clean separation between learning and economic parameters.

# Fields
- `learning::LearningParameters`: Parameters for learning dynamics stage
- `economic::EconomicParameters`: Economic fundamentals and informational parameters

# Usage
```julia
# Convenient construction with keyword arguments
m = ModelParameters(β=1.0, η_bar=15.0, u=0.1, p=0.5, κ=0.6, λ=0.01)
m.learning #prints learning parameters
m.economic #prints economic parameters

# Staged computation
lr = solve_learning(m.learning)          # Stage 1: Learning
results = solve_equilibrium_baseline(lr, m.economic)  # Stage 2: Equilibrium
```
"""
struct ModelParameters
    learning::LearningParameters
    economic::EconomicParameters

    function ModelParameters(learning::LearningParameters, economic::EconomicParameters)
        new(learning, economic)
    end
end


######################################################
###### Convenience utils #############################
######################################################
"""
    ModelParameters(; kwargs...)

Convenience constructor with keyword arguments and defaults.

# Default Parameters
- `β = 1.0`: Communication speed
- `η = nothing`: Normalized awareness window (computed from η_bar if not provided)
- `η_bar = 15.0`: Raw awareness window (η = η_bar/β if η not provided)
- `u = 0.1`: Utility flow
- `p = 0.5`: Prior fragility probability
- `κ = 0.6`: Solvency threshold
- `λ = 0.01`: Exponential rate
- `tspan = (0.0, 2*η)`: Simulation time span
- `x0 = 0.0001`: Initial learning condition

# Examples
```julia
# Use all defaults (η = η_bar/β = 15.0/1.0 = 15.0)
baseline = ModelParameters()

# Directly specify η
params = ModelParameters(β=2.0, η=10.0)

# Use η_bar (η will be computed as η_bar/β)
params = ModelParameters(β=2.0, η_bar=30.0)  # η = 30.0/2.0 = 15.0
```
"""
function ModelParameters(;
    β = 1.0,
    η = nothing,
    η_bar = 15.0,
    u = 0.1,
    p = 0.5,
    κ = 0.6,
    λ = 0.01,
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
    economic = EconomicParameters(u, p, κ, λ, η_bar, η)
    
    return ModelParameters(learning, economic)
end

"""
    ModelParameters(base::ModelParameters; kwargs...)

Create new ModelParameters by modifying an existing one.

# Example
```julia
baseline = ModelParameters()
high_fragility = ModelParameters(baseline; κ=0.3, p=0.8)
```
"""
function ModelParameters(base::ModelParameters; kwargs...)
    # Extract current values from substructs
    current = (
        β = base.learning.β,
        η = base.economic.η,
        η_bar = base.economic.η_bar,
        u = base.economic.u,
        p = base.economic.p,
        κ = base.economic.κ,
        λ = base.economic.λ,
        tspan = base.learning.tspan,
        x0 = base.learning.x0
    )
    
    # Merge with new values
    merged = merge(current, kwargs)
    
    # Use keyword constructor to handle interdependencies
    return ModelParameters(
        β=merged.β, η=merged.η, η_bar=merged.η_bar, u=merged.u, p=merged.p, 
        κ=merged.κ, λ=merged.λ, tspan=merged.tspan, x0=merged.x0
    )
end

"""
    show(io::IO, params::LearningParameters)

Display LearningParameters in a readable format.
"""
function Base.show(io::IO, params::LearningParameters)
    print(io, "LearningParameters(β=$(params.β), tspan=$(params.tspan), x0=$(params.x0))")
end

"""
    show(io::IO, params::EconomicParameters)

Display EconomicParameters in a readable format.
"""
function Base.show(io::IO, params::EconomicParameters)
    print(io, "EconomicParameters(\n")
    print(io, "  Fundamentals: u=$(params.u), p=$(params.p), κ=$(params.κ)\n")
    print(io, "  Informational: λ=$(params.λ), η_bar=$(params.η_bar), η=$(params.η)\n")
    print(io, ")")
end

"""
    show(io::IO, model::ModelParameters)

Display ModelParameters with nested substructs in a readable format.
"""
function Base.show(io::IO, model::ModelParameters)
    print(io, "ModelParameters(\n")
    print(io, "  Learning: β=$(model.learning.β), tspan=$(model.learning.tspan), x0=$(model.learning.x0)\n")
    print(io, "  Economic: u=$(model.economic.u), p=$(model.economic.p), κ=$(model.economic.κ), λ=$(model.economic.λ)\n") 
    print(io, "  Awareness: η_bar=$(model.economic.η_bar), η=$(model.economic.η)\n")
    print(io, ")")
end
