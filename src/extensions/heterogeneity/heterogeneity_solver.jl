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
        
        # Compute finite difference epsilon using grid spacing (not fixed value)
        # All groups share compatible grids from coupled ODE solver
        grid_points = learning_cdfs[1].itp.knots[1]
        current_idx = searchsortedlast(grid_points, ξ_old)
        ε = grid_points[min(current_idx + 1, length(grid_points))] - grid_points[current_idx]

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
            AW_ϵ += dist[k] * (learning_cdfs[k](τ_bar_OUT_CON + ε) - learning_cdfs[k](τ_bar_IN_CON + ε))
        end

        # Bisection algorithm with 5 distinct cases (matching baseline solver)
        error = AW - κ
        is_increasing = AW_ϵ >= AW

        if verbose && (iter % 50 == 0)
            println("Iteration $iter: ξ = $(round(ξ_new, digits=4)), AW = $(round(AW, digits=6)), error = $(round(error, digits=8))")
        end

        if abs(error) <= tolerance
            if is_increasing
                # Valid equilibrium - root on increasing branch
                # For heterogeneous groups, need additional max check (see is_valid_equilibrium_hetero)
                if !is_valid_equilibrium_hetero(ξ_old, τ_bar_IN_UNCs, learning_cdfs, κ, dist; verbose=verbose)
                    if verbose
                        println("Max check failed: earlier peak exceeded threshold")
                        println("No valid run equilibrium exists")
                    end
                    return NaN, Inf
                end
                if verbose
                    println("Converged in $iter iterations")
                    println("ξ = $ξ_old, AW = $AW")
                end
                return ξ_old, abs(error)
            else
                # False equilibrium - root on decreasing branch
                if verbose
                    println("False equilibrium detected at iteration $iter")
                    println("ξ = $ξ_old, AW = $AW, but slope negative")
                    println("No valid run equilibrium exists")
                end
                return NaN, Inf
            end
        elseif error > 0
            # Overshoot - AW > κ, reduce upper bound
            ξmax = ξ_old
            ξ_new = 0.5 * (ξ_old + ξmin)
        else
            # Undershoot
            ξmin = ξ_old
            ξ_new = 0.5 * (ξ_old + ξmax)
        end
    end

    return NaN, Inf
end

"""
    is_valid_equilibrium_hetero(ξ_star, τ_bar_IN_UNCs, learning_cdfs, κ, dist; tol=1e-10, verbose=false)

Validate that ξ_star is the FIRST crossing of threshold κ, not a later crossing after peak.

For heterogeneous groups, AW(t; ξ) may be multimodal (multiple humps due to different
group dynamics), so the slope check is insufficient. This function computes
max_{t < ξ*} AW(t; ξ*) to ensure no earlier crossing occurred.

# Mathematical Condition
Valid equilibrium requires: max_{t ∈ [0, ξ*)} AW(t; ξ*) < κ + tol

If this fails, the aggregate withdrawals exceeded κ at some earlier time t < ξ*,
meaning the bank would have crashed then, not at ξ*. This indicates a "false
equilibrium" where the bisection found a root after the peak.

# Arguments
- `ξ_star`: Candidate equilibrium crash time
- `τ_bar_IN_UNCs`: Vector of unconstrained reentry times for each group
- `learning_cdfs`: Vector of learning CDF functions for each group
- `κ`: Fragility threshold
- `dist`: Group distribution weights
- `tol`: Numerical tolerance for comparison (default: 1e-10)
- `verbose`: Print diagnostic information (default: false)

# Returns
- `true` if valid equilibrium (no earlier crossing)
- `false` if false equilibrium (earlier peak exceeded κ)

# Implementation Note
Uses the grid from learning_cdfs[1] since all groups share compatible grids
from the coupled ODE solver. Only checks times t strictly before ξ*.
"""
function is_valid_equilibrium_hetero(ξ_star, τ_bar_IN_UNCs, learning_cdfs, κ, dist; tol=1e-10, verbose=false)
    # Get grid points strictly before ξ*
    # Use first group's grid (all groups share compatible grids from coupled ODE solver)
    grid = learning_cdfs[1].itp.knots[1]
    grid  = grid[grid .<= ξ_star]

    if isempty(grid)
        # No time points before ξ*, trivially valid
        return true
    end

    # Compute AW(t; ξ*) for all t < ξ*
    n_groups = length(learning_cdfs)
    AW_path = zeros(length(grid))

    for k in 1:n_groups
        τ_I_k = max(0, ξ_star - τ_bar_IN_UNCs[k])
        for (i, t) in enumerate(grid)
            # AW contribution from group k at time t
            # Note: τ_OUT constraint not needed since we're checking t < ξ*
            AW_path[i] += dist[k] * (learning_cdfs[k](t) - learning_cdfs[k](max(0, t - τ_I_k)))
        end
    end

    # Check if it crosses back bellow \kappa at some point (would happen only if we are at a crossing >1)
    above_κ=AW_path.>κ
    for i in (length(grid)-1):-1:1
        if above_κ[i] && !above_κ[i+1]
            if verbose
                println("False equilibrium detected - not the first crossing")
            end
            return false
        end
    end

    return true
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
