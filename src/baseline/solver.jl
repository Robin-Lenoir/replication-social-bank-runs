### Equilibrium Solver Functions ###
# This file contains the main equilibrium solving functions
# Part of the Social Bank Runs replication package
# Author: Robin Lenoir

#########################################
# SOLVED MODEL STRUCT
#########################################

"""
    SolvedModel

Define structure for complete solution of the Social Bank Runs model containing equilibrium outputs,
original parameters, and learning results. This struct encapsulates everything
needed for analysis and plotting.

# Core Equilibrium Outputs
- `ξ::Float64`: Crash time when bank collapses
- `τ_bar_IN_UNC::Float64`: Unconstrained re-entry buffer time, in reversed time
- `τ_bar_OUT_UNC::Float64`: Unconstrained exit buffer time, in reversed time
- `HR::Any`: Hazard rate function (LinearInterpolation object)
- `bankrun::Bool`: Whether a bank run occurs in equilibrium

# Derived Quantities
- `τ_IN::Float64`: Constrained re-entry time, normal time = max(ξ - τ_bar_IN_UNC, 0)
- `τ_OUT::Float64`: Constrained exit time, normal time = max(ξ - τ_bar_OUT_UNC,0)

# Original Inputs (for traceability and plotting)
- `model_params::ModelParameters`: Complete original model parameters
- `learning_results::LearningResults`: Learning stage solution

# Solution Metadata
- `converged::Bool`: Whether equilibrium solver converged
- `solve_time::Float64`: Computation time for equilibrium stage (seconds)
- `tolerance::Float64`: Convergence tolerance achieved

# Note on Lazy Evaluation
Aggregate withdrawal functions (AW) can computed on-demand and are added to the struct
- `get_AW_functions!(result)`: Returns AW functions, computed and cached on first call
- `get_max_AW(result)`: Convenience function for maximum withdrawals
- `has_AW_cache(result)`: Check if AW functions are cached

# Usage
```julia
# Staged approach (pedagogical)
m = ModelParameters(β=1.0, u=0.1, κ=0.6)
lr = solve_learning(m.learning)
result = solve_equilibrium(lr, m.economic, m)

# Direct access for plotting
plot_equilibrium(result)  # Can access all parameters and results
```

"""
struct SolvedModel
    # Core equilibrium outputs
    ξ::Float64
    τ_bar_IN_UNC::Float64
    τ_bar_OUT_UNC::Float64
    HR::Any  # LinearInterpolation object
    bankrun::Bool

    # Derived quantities
    τ_IN::Float64
    τ_OUT::Float64

    # Original inputs
    model_params::ModelParameters
    learning_results::LearningResults

    # Solution metadata
    converged::Bool
    solve_time::Float64
    tolerance::Float64

    # Computation cache for AW functions
    aw::Ref{Union{Nothing, NamedTuple}}

    function SolvedModel(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun,
                        model_params::ModelParameters, learning_results, converged, solve_time, tolerance)
        # Compute derived quantities
        τ_IN = max(ξ - τ_bar_IN_UNC,0)
        τ_OUT = max(ξ - τ_bar_OUT_UNC,0)

        # Validation (allow NaN for ξ when no bank run occurs)
        (ξ ≥ 0 || isnan(ξ)) || throw(ArgumentError("Crash time ξ must be non-negative or NaN, got ξ = $ξ"))
        τ_bar_IN_UNC ≥ 0 || throw(ArgumentError("τ_bar_IN_UNC must be non-negative, got $τ_bar_IN_UNC"))
        τ_bar_OUT_UNC ≥ 0 || throw(ArgumentError("τ_bar_OUT_UNC must be non-negative, got $τ_bar_OUT_UNC"))
        solve_time ≥ 0 || throw(ArgumentError("Solve time must be non-negative, got $solve_time"))
        tolerance ≥ 0 || throw(ArgumentError("Tolerance must be non-negative, got $tolerance"))

        new(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun, τ_IN, τ_OUT,
            model_params, learning_results, converged, solve_time, tolerance,
            Ref{Union{Nothing, NamedTuple}}(nothing))
    end

    # Alternative constructor that reconstructs ModelParameters from separate parts
    # Useful in certain paramters sweep
    function SolvedModel(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun,
                        econ::EconomicParameters, learning_results::LearningResults,
                        converged, solve_time, tolerance)
        # Reconstruct full ModelParameters from the parts we have
        model_params = ModelParameters(learning_results.params, econ)

        # Call the main constructor
        return SolvedModel(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun,
                          model_params, learning_results, converged, solve_time, tolerance)
    end
end

"""
    show(io::IO, result::SolvedModel)

Display SolvedModel in a readable format showing key equilibrium results.
"""
function Base.show(io::IO, result::SolvedModel)
    print(io, "SolvedModel(\n")
    print(io, "  Equilibrium: ξ=$(result.ξ), bankrun=$(result.bankrun)\n")
    print(io, "  Buffers: τ_bar_IN=$(result.τ_bar_IN_UNC), τ_bar_OUT=$(result.τ_bar_OUT_UNC)\n")
    print(io, "  Derived: τ_IN=$(result.τ_IN), τ_OUT=$(result.τ_OUT)\n")
    print(io, "  Solution: converged=$(result.converged), time=$(round(result.solve_time*1000, digits=1))ms\n")
    print(io, "  Model:
                        β=$(result.model_params.learning.β),
                        u=$(result.model_params.economic.u),
                        κ=$(result.model_params.economic.κ),
                        p=$(result.model_params.economic.p),
                        λ=$(result.model_params.economic.λ)\n")
    print(io, ")")
end


#########################################
# HAZARD RATE SOLVER
#########################################

"""
    hazard_rate(p, a, learning_pdf, η; grid=nothing)

Compute hazard rate for a single group.
Uses the symbolic formula from the paper, taken parameters and learning pdf (g) as given.


# Arguments
- `p`: Prior probability bank is fragile
- `a`: Exponential rate
- `learning_pdf`: PDF function for a single group
- `η`: Awareness window
- `grid`: Optional time grid

# Returns
- Interpolated hazard rate function
"""
function hazard_rate(p, a, learning_pdf, η; grid=nothing)
    # Use the learning_pdf's underlying grid if no custom grid provided
    if isnothing(grid)
        # Get the underlying grid from the LinearInterpolation
        # Cut at η because HR don't go beyond
        τ_bar = learning_pdf.itp.knots[1][learning_pdf.itp.knots[1] .<= η]
        if length(τ_bar) == 0 || τ_bar[end] != η
            push!(τ_bar, η)
        end
    else
        τ_bar = grid[grid .<= η]
        push!(τ_bar, η)
    end

    ## Define integrand function
    eg(t) = exp(a * t) * learning_pdf(t)

    ## Compute integral terms from trapezoidal rule
    ## The loop is important: we compute next integral point from the cumulative sum
    int_0_τ_bar = zeros(length(τ_bar))
    for i in 2:length(τ_bar)
        int_0_τ_bar[i] = int_0_τ_bar[i-1] + 0.5 * (eg(τ_bar[i-1]) + eg(τ_bar[i])) * (τ_bar[i] - τ_bar[i-1])
    end
    int_0_η = int_0_τ_bar[end]

    ## Generate HR as linear interpolation
    ## Underlying is a vec operation to compute all points
    HR = LinearInterpolation(τ_bar,
        (p .* exp.(a .* τ_bar) .* learning_pdf.(τ_bar)) ./
        (p .* int_0_τ_bar .+ (1 - p) .* int_0_η))

    return HR
end

#########################################
# OPTIMAL BUFFER COMPUTATION
#########################################

"""
    optimal_buffer(u, HR, tspan)

Compute optimal buffer times by finding where hazard rate crosses utility threshold.
Buffer refers to the fact that we are in reversed time:
agents chose buffer before estimated collapse time.

# Technical Implementation
- **Method**: Linear interpolation between consecutive grid points where HR crosses u
- **Logic**: Find exact crossings HR(τ) = u by interpolating between adjacent grid points
- **Grid**: Uses HR's underlying interpolation grid for efficiency

# Arguments
- `u`: Utility threshold for optimal switching
- `HR`: Hazard rate function (LinearInterpolation)
- `tspan`: Time span for fallback if no crossing found

# Returns
- `(τ_bar_IN_UNC, τ_bar_OUT_UNC)`: Unconstrained optimal entry/exit times (interpolated)
"""
function optimal_buffer(u, HR, tspan)
    # Use HR's underlying grid directly
    τ_grid = HR.itp.knots[1]
    hr_values = HR.(τ_grid) #store as vector

    # Find crossings where HR transitions from below to above u (for IN) and above to below u (for OUT)
    # this declares as 1 if above and 0 if below
    above_threshold = hr_values .> u

    # Handle boundary cases
    if all(.!above_threshold)
        # No point above threshold - both times at boundary
        return tspan[2], tspan[2]
    elseif all(above_threshold)
        # All points above threshold - both times at start
        return τ_grid[1], τ_grid[end]
    end

    # Find τ_bar_IN_UNC: first crossing from below to above threshold
    τ_bar_IN_UNC = tspan[2]  # default to boundary
    for i in 1:(length(τ_grid)-1)
        if !above_threshold[i] && above_threshold[i+1]
            # Linear interpolation to find exact crossing point
            τ1, τ2 = τ_grid[i], τ_grid[i+1]
            hr1, hr2 = hr_values[i], hr_values[i+1]
            # Solve: hr1 + (hr2-hr1)/(τ2-τ1) * (τ - τ1) = u
            τ_bar_IN_UNC = τ1 + (u - hr1) * (τ2 - τ1) / (hr2 - hr1)
            break
        end
    end

    # Find τ_bar_OUT_UNC: last crossing from above to below threshold
    τ_bar_OUT_UNC = tspan[2]  # default to boundary
    for i in (length(τ_grid)-1):-1:1
        if above_threshold[i] && !above_threshold[i+1]
            # Linear interpolation to find exact crossing point
            τ1, τ2 = τ_grid[i], τ_grid[i+1]
            hr1, hr2 = hr_values[i], hr_values[i+1]
            # Solve: hr1 + (hr2-hr1)/(τ2-τ1) * (τ - τ1) = u
            τ_bar_OUT_UNC = τ1 + (u - hr1) * (τ2 - τ1) / (hr2 - hr1)
            break
        end
    end

    # If no crossing found but we have points above threshold, use grid boundaries
    if τ_bar_IN_UNC == tspan[2] && any(above_threshold)
        τ_bar_IN_UNC = τ_grid[findfirst(above_threshold)]
    end
    if τ_bar_OUT_UNC == tspan[2] && any(above_threshold)
        τ_bar_OUT_UNC = τ_grid[findlast(above_threshold)]
    end

    return τ_bar_IN_UNC, τ_bar_OUT_UNC
end

##########################################
# Solve for \xi given \bar{\tau}
##########################################
"""
    compute_ξ(τ_bar_IN_UNC, τ_bar_OUT_UNC, learning_cdf, κ; kwargs...)

Solve for equilibrium crisis time ξ using adaptive bisection method.
See Appendix C.4 for complete mathematical exposition.

Finds ξ such that AW(ξ) = κ, where AW(ξ) = G(min(ξ, τ̄_OUT)) - G(min(ξ, τ̄_IN))
represents the aggregate withdrawal rate at the crisis time.

# Algorithm Details
The bisection implements 5 distinct cases to find the FIRST crossing of κ:
- **Case 1**: AW > κ (overshoot) → reduce upper bound
- **Case 2**: AW < κ and increasing → increase lower bound (approaching first root)
- **Case 3**: AW < κ and decreasing → reduce upper bound (passed peak, no earlier crossing found yet)
- **Case 4**: |AW - κ| ≤ tol and increasing → valid equilibrium found (first crossing)
- **Case 5**: |AW - κ| ≤ tol and decreasing → false equilibrium (peak occurred earlier, no valid run exists)

The slope check (via finite difference with grid-based epsilon) validates that we find
the FIRST crossing of κ, not a later crossing after the peak of AW(t; ξ). This prevents
"false equilibria" where the bisection converges to a root on the decreasing branch.

See Appendix C.4.2 for mathematical justification and discussion of false equilibria.

# Arguments
- `τ_bar_IN_UNC`: Unconstrained optimal re-entry time
- `τ_bar_OUT_UNC`: Unconstrained optimal exit time
- `learning_cdf`: Learning CDF function G(t)
- `κ`: Target aggregate withdrawal rate

# Keyword Arguments
- `ξ_guess=(τ_bar_IN_UNC+τ_bar_OUT_UNC)/2`: Initial guess for ξ
- `max_iters=500`: Maximum iterations
- `ξmin=τ_bar_IN_UNC`: Lower bound for search
- `ξmax=τ_bar_OUT_UNC`: Upper bound for search
- `tolerance=eps()`: Convergence tolerance
- `verbose=false`: Print iteration details

# Returns
- `ξ`: Equilibrium crisis time (NaN if no solution found)
"""
function compute_ξ(τ_bar_IN_UNC, τ_bar_OUT_UNC, learning_cdf, κ;
                   ξ_guess=(τ_bar_IN_UNC+τ_bar_OUT_UNC)/2, max_iters=100,
                   ξmin=τ_bar_IN_UNC, ξmax=τ_bar_OUT_UNC, tolerance=10*eps(κ), verbose=false)

    ξ_new = ξ_guess

    for iter in 1:max_iters
        # Check convergence conditions
        if abs(ξmin - ξmax) < 2*eps(ξmin - ξmax) #2 times machine epsilon for safety
            verbose && println("No solution: search interval collapsed to machine precision")
            return NaN, Inf
        end

        if iter == max_iters - 1
            verbose && println("Did not converge within $max_iters iterations")
            return NaN, Inf
        end

        ξ_old = ξ_new

        # Constrain withdrawal times to [0, ξ] (0 is automatic from construction)
        τ_bar_IN_CON = min(τ_bar_IN_UNC, ξ_old)
        τ_bar_OUT_CON = min(τ_bar_OUT_UNC, ξ_old)

        # Compute aggregate withdrawal rate at current ξ
        AW = learning_cdf(τ_bar_OUT_CON) - learning_cdf(τ_bar_IN_CON)

        # Compute AW at next grid point to determine direction
        grid_points = learning_cdf.itp.knots[1]
        current_idx = searchsortedlast(grid_points, ξ_old)
        epsilon = grid_points[current_idx + 1]-grid_points[current_idx]
        AW_ϵ = learning_cdf(τ_bar_OUT_CON+epsilon) - learning_cdf(τ_bar_IN_CON+epsilon)

        # Bisection algorithm with 5 distinct cases (see Appendix C.4.2.3)
        error = AW - κ
        is_increasing = AW_ϵ >= AW

        if abs(error) <= tolerance
            if is_increasing
                # Case 4: Valid equilibrium - root on increasing branch
                if verbose
                    println("Converged in $iter iterations")
                    println("ξ = $ξ_old, AW = $AW")
                end
                return ξ_old, abs(error)
            else
               # Case 3b: False equilibrium - root on decreasing branch
                # This means the peak of AW(t; ξ) occurred before ξ, so the bank
                # would have crashed earlier. No valid run equilibrium exists.
                if verbose
                    println("False equilibrium detected at iteration $iter")
                    println("ξ = $ξ_old, AW = $AW, but slope negative")
                    println("No valid run equilibrium exists (peak exceeded threshold earlier)")
                end
                return NaN, Inf
            end
        elseif error > 0
            # Case 1: Overshoot - AW > κ, reduce upper bound
            ξmax = ξ_old
            ξ_new = 0.5 * (ξ_old + ξmin)
        else
            # Case 2: Undershoot
            ξmin = ξ_old
            ξ_new = 0.5 * (ξ_old + ξmax)
        end
    end

    return NaN, Inf
end

#########################################
# MAIN EQUILIBRIUM SOLVER
#########################################

"""
    solve_equilibrium(lr::LearningResults, econ::EconomicParameters; verbose=false)
    solve_equilibrium_baseline(lr::LearningResults, econ::EconomicParameters; verbose=false)

Solve equilibrium using precomputed learning results and economic parameters.
This is the staged approach that separates learning dynamics from economic fundamentals.

# Technical Implementation
- **Convergence**: Bisection method with tolerance = 10*eps(κ) for numerical stability
- **Grid Strategy**: Inherits adaptive grid from learning stage for consistency
- **Max Iterations**: 100 iterations with early termination on convergence

# Arguments
- `lr`: LearningResults containing precomputed learning dynamics
- `econ`: EconomicParameters containing economic fundamentals (u, κ, p, λ, η)
- `verbose`: Print convergence information (default: false)

# Returns
- `SolvedModel`: Complete solution struct with all equilibrium outputs

# Example
```julia
# Primary usage
lr = solve_learning(learning_params)
result = solve_equilibrium(lr, econ_params)

# Convenience usage
lr = solve_learning(model)
result = solve_equilibrium(lr, model)
```
"""
function solve_equilibrium_baseline(lr::LearningResults, econ::EconomicParameters; ξ_guess=nothing, verbose=false)
    solve_start = time()

    ## Extract model parameters
    (; learning_cdf, learning_pdf) = lr
    (; η, u, p, κ, λ) = econ

    # Use tspan from learning parameters for optimal buffer computation
    tspan = lr.params.tspan

    # Compute hazard rate (will use learning_pdf's grid automatically)
    HR = hazard_rate(p, λ, learning_pdf, η)

    # Optimal buffer computation
    τ_bar_IN_UNC, τ_bar_OUT_UNC = optimal_buffer(u, HR, tspan)

    if τ_bar_IN_UNC == τ_bar_OUT_UNC  # happens if u is above the max of HR
        ξ = NaN
        bankrun = false
        converged = true  # Trivial case, no iteration needed
        tolerance_achieved = 0.0
    else

        #### Compute ξ using bisection method
        # tolerance 10*eps(κ)
        target_tolerance = 10*eps(κ)

        # Use provided guess or default (midpoint)
        guess = isnothing(ξ_guess) ? (τ_bar_IN_UNC+τ_bar_OUT_UNC)/2 : ξ_guess

        # Call bisection routine
        ξ, tolerance_achieved = compute_ξ(τ_bar_IN_UNC, τ_bar_OUT_UNC, learning_cdf, κ;
                      ξ_guess=guess,
                      verbose=verbose)
        if isnan(ξ)
            bankrun = false
            converged = false
            tolerance_achieved = Inf
        else
            bankrun = true
            converged = true
            tolerance_achieved = tolerance_achieved
        end
    end

    solve_time = time() - solve_start

    return SolvedModel(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun,
                      econ, lr, converged, solve_time, tolerance_achieved)
end

# Convenience overload for ModelParameters
# function solve_equilibrium(lr::LearningResults, model::ModelParameters; kwargs...)
#     return solve_equilibrium_baseline(lr, model.economic; kwargs...)
# end

# function solve_equilibrium_baseline(lr::LearningResults, model::ModelParameters; kwargs...)
#     return solve_equilibrium_baseline(lr, model.economic; kwargs...)
# end


#########################################
# Additional helpers
#########################################

"""
    get_AW(lr::LearningResults, ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR)

Compute aggregate withdrawal functions for given equilibrium solution.

# Arguments
- `lr`: LearningResults struct containing learning functions
- `ξ`: Crisis time
- `τ_bar_IN_UNC`: Unconstrained re-entry time
- `τ_bar_OUT_UNC`: Unconstrained exit time
- `HR`: Hazard rate function

# Returns
- `AW_cum_func`: Cumulative aggregate withdrawals function
- `AW_OUT_func`: Aggregate exit function
- `AW_IN_func`: Aggregate re-entry function
"""
function get_AW(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR,learning_cdf)

    # Extract grid
    t_grid = HR.itp.knots[1]
    # Constrain times to [0, ξ]
    τ_bar_IN_CON = τ_bar_IN_UNC >= ξ ? ξ : τ_bar_IN_UNC
    τ_bar_OUT_CON = τ_bar_OUT_UNC > ξ ? ξ : τ_bar_OUT_UNC

    # Intitalize aggregate withdrawals curves
    AW_OUT = [] # People out
    AW_IN = [] # People reentered
    AW = [] # Total
    AW_cum = zeros(length(t_grid))


    # Re-Entry dynamics (G(t-(\xi-\bar\tau)))
    grid_IN_trunc = ifelse.(t_grid .- ξ .+ τ_bar_IN_CON .> 0,
                            t_grid .- ξ .+ τ_bar_IN_CON, 0)
    AW_IN=learning_cdf(grid_IN_trunc)
    AW_IN = ifelse.(t_grid .- ξ .+ τ_bar_IN_CON .>= 0, AW_IN, 0)

    # Exit dynamics
    grid_OUT_trunc = ifelse.(t_grid .- ξ .+ τ_bar_OUT_CON .> 0,
                            t_grid .- ξ .+ τ_bar_OUT_CON, 0) #In equilibrium will be simply G(t_grid) because τ_bar_OUT=0, but we allow it nonetheless
    AW_OUT=learning_cdf(grid_OUT_trunc)
    AW_OUT = ifelse.(t_grid .- ξ .+ τ_bar_OUT_CON .>= 0, AW_OUT, 0)

    # Net withdrawals
    AW_cum=AW_OUT .- AW_IN
    AW_cum .+= learning_cdf(0)  # Add initial withdrawals

    # Create LinearInterpolation functions
    AW_IN_func = LinearInterpolation(t_grid, AW_IN)
    AW_OUT_func = LinearInterpolation(t_grid, AW_OUT)
    AW_cum_func = LinearInterpolation(t_grid, AW_cum)

    return AW_cum_func, AW_OUT_func, AW_IN_func
end

"""
    get_AW_functions!(result::SolvedModel)

Lazy evaluation of aggregate withdrawal functions. Computes and caches AW functions in struct
on first call, returns cached results on subsequent calls.

# Returns
- `NamedTuple` with fields:
  - `AW_cum`: Cumulative aggregate withdrawals function
  - `AW_OUT`: Outflow function
  - `AW_IN`: Inflow (re-entry) function

# Example
```julia
result = solve_equilibrium(lr, econ, model)
aw = get_AW_functions!(result)
println("Max withdrawals: \$(maximum(aw.AW_cum.itp.coefs))")
```
"""
function get_AW_functions!(result::SolvedModel)
    # Return cached result if already computed
    if result.aw[] !== nothing
        return result.aw[]
    end

    # Compute AW functions (expensive operation)
    if result.bankrun
        AW_cum_func, AW_OUT_func, AW_IN_func = get_AW(
            result.ξ, result.τ_bar_IN_UNC, result.τ_bar_OUT_UNC,
            result.HR, result.learning_results.learning_cdf
        )
        AW_max=maximum(AW_cum_func.itp.coefs)

        # Cache the result
        aw_result = (AW_cum=AW_cum_func, AW_OUT=AW_OUT_func, AW_IN=AW_IN_func, AW_max=AW_max)
        result.aw[] = aw_result
        return result.aw[]
    else
        # No bank run - return NaN functions or empty results
        # This handles the case where ξ = NaN
        return result.aw[]
    end
end
