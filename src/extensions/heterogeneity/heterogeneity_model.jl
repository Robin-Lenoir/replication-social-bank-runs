#=
heterogeneity_model.jl

Model structure definitions for the heterogeneity extension.
Handles multiple groups with different learning rates βₖ.

Author: Robin Lenoir
=#

# Include baseline model definitions
include(joinpath(@__DIR__, "..", "..", "baseline", "model.jl"))

"""
    LearningParametersHetero

Learning dynamics parameters for heterogeneous groups with different learning rates.
Extends the baseline model to handle multiple agent groups.

# Fields
- `βs::Vector{Float64}`: Learning rates for each group (all > 0)
- `dist::Vector{Float64}`: Distribution of agents across groups (must sum to 1)
- `tspan::Tuple{Float64,Float64}`: Time span for learning dynamics simulation
- `x0::Float64`: Initial condition for learning ODE (replicated across groups)
"""
struct LearningParametersHetero
    βs::Vector{Float64}              # Learning rates for each group
    dist::Vector{Float64}            # Distribution across groups  
    tspan::Tuple{Float64, Float64}   # Time span for learning dynamics
    x0::Float64                      # Single initial condition replicated
    
    function LearningParametersHetero(βs, dist, tspan, x0)
        # Validation
        length(βs) ≥ 1 || throw(ArgumentError("Must have at least one group, got $(length(βs)) groups"))
        length(dist) == length(βs) || throw(ArgumentError("Distribution length $(length(dist)) must match βs length $(length(βs))"))
        all(β -> β > 0, βs) || throw(ArgumentError("All learning rates βs must be positive"))
        all(d -> d ≥ 0, dist) || throw(ArgumentError("All distribution weights must be non-negative"))
        abs(sum(dist) - 1.0) < 1e-10 || throw(ArgumentError("Distribution must sum to 1, got sum = $(sum(dist))"))
        length(tspan) == 2 || throw(ArgumentError("Time span must be a tuple of length 2"))
        tspan[1] ≥ 0 || throw(ArgumentError("Start time must be non-negative"))
        tspan[2] > tspan[1] || throw(ArgumentError("End time must be greater than start time"))
        x0 ≥ 0 || throw(ArgumentError("Initial condition must be non-negative"))
        
        new(βs, dist, tspan, x0)
    end
end

# Keyword constructor for LearningParametersHetero
function LearningParametersHetero(; βs::Vector{Float64}, dist::Vector{Float64}, 
                                 tspan::Tuple{Float64, Float64}, x0::Float64)
    return LearningParametersHetero(βs, dist, tspan, x0)
end

"""
    ModelParametersHetero

Master struct containing all parameters for the heterogeneous Social Bank Runs model.
Combines heterogeneous learning parameters with baseline economic parameters.

# Fields
- `learning::LearningParametersHetero`: Parameters for heterogeneous learning dynamics
- `economic::EconomicParameters`: Economic fundamentals (reused from baseline)

# Usage
```julia
# Create heterogeneous model with 3 groups
βs = [0.5, 1.0, 2.0]
dist = [0.3, 0.5, 0.2]
m_hetero = ModelParametersHetero(βs=βs, dist=dist, η_bar=15.0, u=0.1)

# Staged computation
lr_hetero = solve_SInetwork_hetero(m_hetero.learning)
result = solve_equilibrium_hetero(lr_hetero, m_hetero.economic, m_hetero)
```
"""
struct ModelParametersHetero
    learning::LearningParametersHetero
    economic::EconomicParameters

    function ModelParametersHetero(learning::LearningParametersHetero, economic::EconomicParameters)
        new(learning, economic)
    end
end

"""
    ModelParametersHetero(; kwargs...)

Convenience constructor with keyword arguments and defaults.

# Heterogeneity-Specific Parameters
- `βs::Vector{Float64}`: Learning rates for each group (required)
- `dist::Vector{Float64}`: Distribution across groups (required, must sum to 1)
- `η_bar = 15.0`: Raw awareness window (normalized by max(βs))
- `x0 = 0.0001`: Initial learning condition

# Baseline Parameters (same defaults as baseline model)
- `u = 0.1`: Utility flow
- `p = 0.5`: Prior fragility probability  
- `κ = 0.6`: Solvency threshold
- `λ = 0.01`: Exponential rate
- `tspan = nothing`: Time span (auto-computed if not provided)

# Examples
```julia
# Two-group model
m = ModelParametersHetero(βs=[0.5, 2.0], dist=[0.7, 0.3])

# Three-group model with custom parameters
m = ModelParametersHetero(
    βs=[0.2, 1.0, 5.0], 
    dist=[0.2, 0.6, 0.2],
    u=0.05, κ=0.4
)
```
"""
function ModelParametersHetero(; 
    βs::Vector{Float64},
    dist::Vector{Float64},
    η_bar = 15.0,
    u = 0.1,
    p = 0.5,
    κ = 0.6,
    λ = 0.01,
    tspan = nothing,
    x0 = 0.0001
)
    # Validate required parameters
    length(βs) ≥ 1 || throw(ArgumentError("βs cannot be empty"))
    length(dist) == length(βs) || throw(ArgumentError("dist must have same length as βs"))
    
    # Compute η from η_bar using average learning rate
    β_ave = sum(dist.*βs)
    η = η_bar / β_ave
    
    # Set default tspan based on computed η if not provided
    if tspan === nothing
        tspan = (0.0, 2*η)
    end
    
    # Build substructs
    learning = LearningParametersHetero(βs, dist, tspan, x0)
    economic = EconomicParameters(u, p, κ, λ, η_bar, η)
    
    return ModelParametersHetero(learning, economic)
end

"""
    ModelParametersHetero(base::ModelParametersHetero; kwargs...)

Create new ModelParametersHetero by modifying an existing one.

# Example
```julia
baseline_hetero = ModelParametersHetero(βs=[1.0, 2.0], dist=[0.5, 0.5])
high_fragility = ModelParametersHetero(baseline_hetero; κ=0.3, p=0.8)
```
"""
function ModelParametersHetero(base::ModelParametersHetero; kwargs...)
    # Extract current values from substructs
    current = (
        βs = base.learning.βs,
        dist = base.learning.dist,
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
    return ModelParametersHetero(
        βs=merged.βs, dist=merged.dist, η_bar=merged.η_bar, u=merged.u, p=merged.p, 
        κ=merged.κ, λ=merged.λ, tspan=merged.tspan, x0=merged.x0
    )
end

"""
    LearningResultsHetero

Results from solving heterogeneous learning dynamics.
Contains arrays of learning functions, one for each group.

# Fields
- `params::LearningParametersHetero`: Original learning parameters
- `learning_cdfs::Vector{Any}`: Learning CDFs for each group (LinearInterpolation objects)
- `learning_pdfs::Vector{Any}`: Learning PDFs for each group (LinearInterpolation objects)
- `grid::Vector{Float64}`: Common time grid used for all groups
- `solve_time::Float64`: Computation time for learning stage
- `ode_solution::Any`: Raw ODE solution object
"""
struct LearningResultsHetero
    params::LearningParametersHetero
    learning_cdfs::Vector{Any}       # Array of LinearInterpolation objects
    learning_pdfs::Vector{Any}       # Array of LinearInterpolation objects  
    grid::Vector{Float64}            # Common time grid
    solve_time::Float64
    ode_solution::Any                # ODESolution object
    
    function LearningResultsHetero(params, learning_cdfs, learning_pdfs, grid, solve_time, ode_solution)
        # Validation
        length(learning_cdfs) == length(params.βs) || throw(ArgumentError("Number of CDFs must match number of groups"))
        length(learning_pdfs) == length(params.βs) || throw(ArgumentError("Number of PDFs must match number of groups"))
        solve_time ≥ 0 || throw(ArgumentError("Solve time must be non-negative"))
        
        new(params, learning_cdfs, learning_pdfs, grid, solve_time, ode_solution)
    end
end

"""
    LearningResultsHetero display method
"""
function Base.show(io::IO, lr::LearningResultsHetero)
    println(io, "LearningResultsHetero(")
    println(io, "  Groups: $(length(lr.learning_cdfs))")
    println(io, "  βs = $(lr.params.βs)")
    println(io, "  Grid points: $(length(lr.grid))")
    println(io, "  Solve time: $(round(lr.solve_time*1000, digits=1))ms")
    print(io, ")")
end

"""
    SolvedModelHetero

Complete solution of the heterogeneous Social Bank Runs model.
Extends SolvedModel to handle multiple groups.

# Fields (extending SolvedModel)
- Core equilibrium outputs: `ξ`, `bankrun` (scalars)
- Group-specific outputs: `τ_bar_IN_UNCs`, `τ_bar_OUT_UNCs`, `HRs` (vectors)
- Derived quantities: `τ_INs`, `τ_OUTs` (vectors)
- Original inputs: `model_params`, `learning_results`
- Solution metadata: `converged`, `solve_time`, `tolerance`
"""
struct SolvedModelHetero
    # Core equilibrium outputs (scalars)
    ξ::Float64
    bankrun::Bool
    
    # Group-specific outputs (vectors)
    τ_bar_IN_UNCs::Vector{Float64}   # Unconstrained re-entry times per group
    τ_bar_OUT_UNCs::Vector{Float64}  # Unconstrained exit times per group
    HRs::Vector{Any}                 # Hazard rate functions per group
    
    # Derived quantities (vectors)  
    τ_INs::Vector{Float64}           # Constrained re-entry times per group
    τ_OUTs::Vector{Float64}          # Constrained exit times per group
    
    # Original inputs
    model_params::ModelParametersHetero
    learning_results::LearningResultsHetero
    
    # Solution metadata
    converged::Bool
    solve_time::Float64
    tolerance::Float64
    
    # Computation cache for AW functions
    aw::Ref{Union{Nothing, NamedTuple}}
    
    function SolvedModelHetero(ξ, τ_bar_IN_UNCs, τ_bar_OUT_UNCs, HRs, bankrun,
                              model_params::ModelParametersHetero, learning_results::LearningResultsHetero,
                              converged, solve_time, tolerance)
        # Compute derived quantities
        τ_INs = max.(ξ .- τ_bar_IN_UNCs, 0)
        τ_OUTs = max.(ξ .- τ_bar_OUT_UNCs, 0)
        
        # Validation
        (ξ ≥ 0 || isnan(ξ)) || throw(ArgumentError("Crash time ξ must be non-negative or NaN"))
        all(τ -> τ ≥ 0, τ_bar_IN_UNCs) || throw(ArgumentError("All τ_bar_IN_UNCs must be non-negative"))
        all(τ -> τ ≥ 0, τ_bar_OUT_UNCs) || throw(ArgumentError("All τ_bar_OUT_UNCs must be non-negative"))
        length(τ_bar_IN_UNCs) == length(model_params.learning.βs) || throw(ArgumentError("Length mismatch for τ_bar_IN_UNCs"))
        length(τ_bar_OUT_UNCs) == length(model_params.learning.βs) || throw(ArgumentError("Length mismatch for τ_bar_OUT_UNCs"))
        
        new(ξ, bankrun, τ_bar_IN_UNCs, τ_bar_OUT_UNCs, HRs, τ_INs, τ_OUTs,
            model_params, learning_results, converged, solve_time, tolerance,
            Ref{Union{Nothing, NamedTuple}}(nothing))
    end

    # Alternative constructor that reconstructs ModelParametersHetero from separate parts
    function SolvedModelHetero(ξ, τ_bar_IN_UNCs, τ_bar_OUT_UNCs, HRs, bankrun,
                              econ::EconomicParameters, learning_results::LearningResultsHetero,
                              converged, solve_time, tolerance)
        # Reconstruct full ModelParametersHetero from the parts we have
        model_params = ModelParametersHetero(learning_results.params, econ)

        # Call the main constructor
        return SolvedModelHetero(ξ, τ_bar_IN_UNCs, τ_bar_OUT_UNCs, HRs, bankrun,
                                model_params, learning_results, converged, solve_time, tolerance)
    end
end

# Display functions
function Base.show(io::IO, params::LearningParametersHetero)
    println(io, "LearningParametersHetero(")
    println(io, "  βs = $(params.βs)")
    println(io, "  dist = $(params.dist)")
    println(io, "  tspan = $(params.tspan)")
    println(io, "  x0 = $(params.x0)")
    print(io, ")")
end

function Base.show(io::IO, model::ModelParametersHetero)
    println(io, "ModelParametersHetero(")
    println(io, "  Groups: $(length(model.learning.βs))")
    println(io, "  βs = $(model.learning.βs)")
    println(io, "  dist = $(model.learning.dist)")
    println(io, "  Economic: u=$(model.economic.u), κ=$(model.economic.κ), p=$(model.economic.p)")
    println(io, "  Awareness: η_bar=$(model.economic.η_bar), η=$(model.economic.η)")
    print(io, ")")
end

function Base.show(io::IO, result::SolvedModelHetero)
    println(io, "SolvedModelHetero(")
    println(io, "  Equilibrium: ξ=$(result.ξ), bankrun=$(result.bankrun)")
    println(io, "  Groups: $(length(result.τ_bar_IN_UNCs))")
    println(io, "  βs = $(result.model_params.learning.βs)")
    println(io, "  τ_bar_INs = $(round.(result.τ_bar_IN_UNCs, digits=2))")
    println(io, "  τ_bar_OUTs = $(round.(result.τ_bar_OUT_UNCs, digits=2))")
    println(io, "  Solution: converged=$(result.converged), time=$(round(result.solve_time*1000, digits=1))ms")
    print(io, ")")
end
