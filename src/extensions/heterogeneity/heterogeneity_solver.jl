#=
heterogeneity_solver.jl

Equilibrium solver for heterogeneous groups extension.
Handles multiple groups with different learning rates and optimal decisions.

Author: Robin Lenoir
=#

# Include required dependencies
using Interpolations

# Include model definitions and learning solver
include(joinpath(@__DIR__, "heterogeneity_model.jl"))
include(joinpath(@__DIR__, "heterogeneity_learning.jl"))

# Include baseline equilibrium utilities (hazard rate, etc.)
include(joinpath(@__DIR__, "..", "..", "baseline", "solver.jl"))

"""
    compute_ξ_hetero(τ_bar_IN_UNCs, τ_bar_OUT_UNCs, dist, learning_cdfs, κ; kwargs...)

Solve for equilibrium crisis time ξ in heterogeneous model using Newton's method.
Extends the baseline ξ computation to handle multiple groups with different optimal times.

# Mathematical Framework
AW(ξ) = ∑_k dist_k * [G_k(min(ξ, τ̄_OUT_k)) - G_k(min(ξ, τ̄_IN_k))]
Find ξ such that AW(ξ) = κ

# Arguments
- `τ_bar_IN_UNCs::Vector{Float64}`: Unconstrained re-entry times for each group
- `τ_bar_OUT_UNCs::Vector{Float64}`: Unconstrained exit times for each group
- `dist::Vector{Float64}`: Distribution across groups
- `learning_cdfs::Vector`: Learning CDF functions for each group
- `κ::Float64`: Target aggregate withdrawal rate

# Keyword Arguments
- `ξ_guess::Float64`: Initial guess (default: mean of group optimal times)
- `max_iters::Int = 500`: Maximum iterations
- `tolerance::Float64 = 1e-12`: Convergence tolerance
- `verbose::Bool = false`: Print iteration details

# Returns
- `ξ::Float64`: Equilibrium crisis time (NaN if no solution)
- `tolerance_achieved::Float64`: Final tolerance achieved

"""
function compute_ξ_hetero(τ_bar_IN_UNCs, τ_bar_OUT_UNCs, dist, learning_cdfs, κ;
                         ξ_guess=nothing, max_iters=500, tolerance=1e-12, verbose=false)
    
    n_groups = length(τ_bar_IN_UNCs)
    
    # Set initial guess as weighted average of group optimal times
    if isnothing(ξ_guess)
        ξ_guess = sum(dist[k] * (τ_bar_IN_UNCs[k] + τ_bar_OUT_UNCs[k])/2 for k in 1:n_groups)
    end
    
    # Set search bounds
    ξmin = 0.0
    ξmax = maximum(τ_bar_OUT_UNCs) * 2  # Conservative upper bound
    ξ_new = ξ_guess

    for iter in 1:max_iters
        # Check convergence conditions
        if abs(ξmin - ξmax) < 2*eps(ξmin - ξmax)
            verbose && println("No solution: search interval collapsed to machine precision")
            return NaN, Inf
        end

        if iter == max_iters - 1
            verbose && println("Did not converge within $max_iters iterations")
            return NaN, Inf
        end

        ξ_old = ξ_new
        
        # Compute aggregate withdrawals for each group
        AW = 0.0
        AW_ϵ = 0.0
        
        for k in 1:n_groups
            # Constrain withdrawal times to [0, ξ]
            τ_bar_IN_CON = min(τ_bar_IN_UNCs[k], ξ_old)
            τ_bar_OUT_CON = min(τ_bar_OUT_UNCs[k], ξ_old)
            
            # Compute group contribution to aggregate withdrawals
            AW += dist[k] * (learning_cdfs[k](τ_bar_OUT_CON) - learning_cdfs[k](τ_bar_IN_CON))
            
            # Compute finite difference for direction checking
            ε = 1e-6
            AW_ϵ += dist[k] * (learning_cdfs[k](τ_bar_OUT_CON + ε) - learning_cdfs[k](τ_bar_IN_CON + ε))
        end

        # Update search bounds using bisection logic
        error = AW - κ
        
        if verbose && (iter % 50 == 0)
            println("Iteration $iter: ξ = $(round(ξ_new, digits=4)), AW = $(round(AW, digits=6)), error = $(round(error, digits=8))")
        end
        
        if error > tolerance
            # AW > κ: overshoot, reduce upper bound
            ξmax = ξ_old
            ξ_new = 0.5 * (ξ_old + ξmin)
        elseif error < -tolerance && AW_ϵ >= AW
            # AW < κ and increasing: undershoot, increase lower bound
            ξmin = ξ_old
            ξ_new = 0.5 * (ξ_old + ξmax)
        elseif AW_ϵ < AW
            # AW decreasing: potential overshoot, reduce upper bound
            ξmax = ξ_old
            ξ_new = 0.5 * (ξ_old + ξmin)
        else
            # Converged
            if verbose
                println("Converged in $iter iterations")
                println("ξ = $ξ_new, AW = $AW")
            end
            return ξ_new, abs(AW - κ)
        end
    end

    return NaN, Inf
end

"""
    solve_equilibrium_hetero(lr_hetero::LearningResultsHetero, econ::EconomicParameters; verbose=false)

Solve equilibrium for heterogeneous model using staged computation approach.

# Technical Implementation
1. **Hazard Rate Computation**: Compute separate hazard rates for each group
2. **Optimal Buffers**: Find group-specific optimal entry/exit times
3. **Crisis Time**: Solve for ξ using group-weighted aggregate withdrawals
4. **Result Assembly**: Create SolvedModelHetero with all group-specific results

# Arguments
- `lr_hetero::LearningResultsHetero`: Precomputed heterogeneous learning results
- `econ::EconomicParameters`: Economic parameters (shared across groups)

# Keyword Arguments
- `verbose::Bool = false`: Print convergence information

# Returns
- `SolvedModelHetero`: Complete heterogeneous equilibrium solution

# Example
```julia
# Staged computation
params = ModelParametersHetero(βs=[0.5, 2.0], dist=[0.7, 0.3])
lr_hetero = solve_SInetwork_hetero(params.learning)
result = solve_equilibrium_hetero(lr_hetero, params.economic)
```
"""
function solve_equilibrium_hetero(lr_hetero::LearningResultsHetero, econ::EconomicParameters; verbose=false)
    solve_start = time()

    # Extract parameters
    (; learning_cdfs, learning_pdfs) = lr_hetero
    (; η, u, p, κ, λ) = econ
    (; βs, dist) = lr_hetero.params
    n_groups = length(βs)
    
    # Use tspan from learning parameters
    tspan = lr_hetero.params.tspan

    # Compute hazard rates for each group
    # Loop over groups, calling baseline hazard_rate for each
    HRs = [hazard_rate(p, λ, learning_pdfs[k], η; grid=lr_hetero.grid) for k in 1:n_groups]
    
    # Compute optimal buffers for each group
    τ_bar_IN_UNCs = zeros(n_groups)
    τ_bar_OUT_UNCs = zeros(n_groups)
    
    for k in 1:n_groups
        τ_bar_IN_UNCs[k], τ_bar_OUT_UNCs[k] = optimal_buffer(u, HRs[k], tspan)
    end
    
    # Check if any group has valid optimal times
    if all(τ_bar_IN_UNCs .== τ_bar_OUT_UNCs)
        # No group can optimally exit - no bank run equilibrium
        ξ = NaN
        bankrun = false
        converged = true
        tolerance_achieved = 0.0
    else
        # Solve for equilibrium crisis time ξ
        ξ, tolerance_achieved = compute_ξ_hetero(
            τ_bar_IN_UNCs, τ_bar_OUT_UNCs, dist, learning_cdfs, κ;
            verbose=verbose
        )
        
        if isnan(ξ)
            bankrun = false
            converged = false
            tolerance_achieved = Inf
        else
            bankrun = true
            converged = true
        end
    end
    
    solve_time = time() - solve_start

    return SolvedModelHetero(ξ, τ_bar_IN_UNCs, τ_bar_OUT_UNCs, HRs, bankrun,
                            econ, lr_hetero, converged, solve_time, tolerance_achieved)
end

"""
    get_AW_hetero(result::SolvedModelHetero)

Compute aggregate withdrawal functions for heterogeneous equilibrium solution.
Extends the baseline AW computation to handle multiple groups.

# Arguments
- `result::SolvedModelHetero`: Solved heterogeneous model

# Returns
- `NamedTuple` with fields:
  - `AW_cum`: Total aggregate withdrawals function
  - `AW_OUT_groups`: Exit functions for each group
  - `AW_IN_groups`: Re-entry functions for each group
  - `AW_groups`: Net withdrawal functions for each group
  - `AW_max`: Maximum aggregate withdrawals

# Mathematical Framework
AW_total(t) = ∑_k dist_k * AW_k(t)
where AW_k(t) = G_k(max(0, t - ξ + τ̄_OUT_k)) - G_k(max(0, t - ξ + τ̄_IN_k))
"""
function get_AW_hetero(result::SolvedModelHetero)
    if !result.bankrun
        return nothing
    end
    
    (; ξ, τ_bar_IN_UNCs, τ_bar_OUT_UNCs, model_params, learning_results) = result
    (; learning_cdfs) = learning_results
    (; dist) = model_params.learning
    
    # Use common grid from learning results
    t_grid = learning_results.grid
    n_groups = length(learning_cdfs)
    
    # Initialize group-specific withdrawal arrays
    AW_OUT_groups = []
    AW_IN_groups = []
    AW_groups = []
    AW_cum = zeros(length(t_grid))
    
    for k in 1:n_groups
        # Constrain times to [0, ξ]
        τ_bar_IN_CON = min(τ_bar_IN_UNCs[k], ξ)
        τ_bar_OUT_CON = min(τ_bar_OUT_UNCs[k], ξ)
        
        # Re-entry dynamics for group k
        grid_IN_trunc = ifelse.(t_grid .- ξ .+ τ_bar_IN_CON .> 0,
                               t_grid .- ξ .+ τ_bar_IN_CON, 0)
        AW_IN_k = learning_cdfs[k].(grid_IN_trunc)
        AW_IN_k = ifelse.(t_grid .- ξ .+ τ_bar_IN_CON .>= 0, AW_IN_k, 0)
        
        # Exit dynamics for group k
        grid_OUT_trunc = ifelse.(t_grid .- ξ .+ τ_bar_OUT_CON .> 0,
                                t_grid .- ξ .+ τ_bar_OUT_CON, 0)
        AW_OUT_k = learning_cdfs[k].(grid_OUT_trunc)
        AW_OUT_k = ifelse.(t_grid .- ξ .+ τ_bar_OUT_CON .>= 0, AW_OUT_k, 0)
        
        # Net withdrawals for group k
        AW_k = AW_OUT_k .- AW_IN_k
        
        # Add to aggregate (weighted by group size)
        AW_cum .+= dist[k] .* AW_k
        
        # Store group-specific results
        push!(AW_OUT_groups, LinearInterpolation(t_grid, AW_OUT_k))
        push!(AW_IN_groups, LinearInterpolation(t_grid, AW_IN_k))
        push!(AW_groups, LinearInterpolation(t_grid, AW_k))
    end
    
    # Create aggregate function
    AW_cum_func = LinearInterpolation(t_grid, AW_cum)
    AW_max = maximum(AW_cum)
    
    return (
        AW_cum = AW_cum_func,
        AW_OUT_groups = AW_OUT_groups,
        AW_IN_groups = AW_IN_groups,
        AW_groups = AW_groups,
        AW_max = AW_max
    )
end

"""
    get_AW_functions_hetero!(result::SolvedModelHetero)

Lazy evaluation of heterogeneous aggregate withdrawal functions.
Computes and caches AW functions on first call.

# Returns
- `NamedTuple`: Cached or newly computed AW functions for heterogeneous model
"""
function get_AW_functions_hetero!(result::SolvedModelHetero)
    # Return cached result if already computed
    if result.aw[] !== nothing
        return result.aw[]
    end

    # Compute AW functions (expensive operation)
    if result.bankrun
        aw_result = get_AW_hetero(result)
        result.aw[] = aw_result
        return result.aw[]
    else
        # No bank run - return nothing
        result.aw[] = nothing
        return nothing
    end
end
