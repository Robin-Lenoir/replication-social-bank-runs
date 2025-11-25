#=
interest_rate_solver.jl

Equilibrium solver for interest rates extension.
Integrates value function computation with equilibrium solving when r > 0.

Author: Robin Lenoir
=#

# Include required dependencies
using Interpolations

# Include model definitions and value function solver
include(joinpath(@__DIR__, "interest_rate_model.jl"))
include(joinpath(@__DIR__, "value_function_solver.jl"))

# Include baseline equilibrium utilities
include(joinpath(@__DIR__, "..", "..", "baseline", "solver.jl"))

"""
    solve_equilibrium_interest(lr::LearningResults, econ::EconomicParametersInterest, 
                              model::ModelParametersInterest; verbose=false)

Solve equilibrium for interest rate extension using value function approach.
See Appendix C for details on solving method.
We first compute the hazard rate, then the value function from HR (see dedicated function),
then we assign HR-rV to the hazard_rate field in baseline.
This allows to use the exact same routine as baseline but where HR-->HR-rV: get optimal buffer, etc.


# Arguments
- `lr::LearningResults`: Precomputed baseline learning results
- `econ::EconomicParametersInterest`: Interest rate economic parameters
- `model::ModelParametersInterest`: Complete interest rate model parameters

# Keyword Arguments
- `verbose::Bool = false`: Print convergence and computation information
- `ξ_guess`: Initial guess for crisis time (optional)

# Returns
- `SolvedModelInterest`: Complete interest rate equilibrium solution

# Example
```julia
# Staged computation with interest rates
m_interest = ModelParametersInterest(β=1.0, r=0.02, δ=0.1)
lr = solve_learning(m_interest.learning)
result = solve_equilibrium_interest(lr, m_interest.economic, m_interest)
```
"""
function solve_equilibrium_interest(lr::LearningResults, econ::EconomicParametersInterest, 
                                   model::ModelParametersInterest; ξ_guess=nothing, verbose=false)
    solve_start = time()
    
    # Extract parameters
    (; learning_cdf, learning_pdf) = lr
    (; η, u, p, κ, λ, r, δ) = econ
    
    # Use tspan from learning parameters
    tspan = lr.params.tspan
    
    if verbose
        println("Solving interest rate extension equilibrium...")
        println("  Parameters: r=$r, δ=$δ, u=$u")
    end

    # Compute hazard rate (same as baseline)
    HR = hazard_rate(p, λ, learning_pdf, η)
    
    # Branch based on whether interest rates are positive
    if r > 0
        if verbose
            println("  Computing value function (r > 0)...")
        end
        
        # Solve value function and compute optimal buffers
        V = solve_value_function(HR, δ, r, u)

        # Create extrapolate object for h - rV ("effective" hazard for reentry)
        grid = V.itp.knots[1]
        h_rV_values = [HR(t) for t in grid] .-  r .* [V(t) for t in grid]
        h_rV = extrapolate(interpolate((grid,), h_rV_values, Gridded(Linear())), Throw())

        if verbose
            println("  Value function computed, grid size: $(length(grid))")
        end

        τ_bar_IN_UNC, τ_bar_OUT_UNC = optimal_buffer(u, h_rV, tspan)
    else
        if verbose
            println("  Using baseline approach (r = 0)...")
        end

        # Use baseline approach when r = 0
        τ_bar_IN_UNC, τ_bar_OUT_UNC = optimal_buffer(u, HR, tspan)
        V = nothing

        if verbose
            println("  Baseline buffers: τ̄_IN=$(round(τ_bar_IN_UNC, digits=3)), τ̄_OUT=$(round(τ_bar_OUT_UNC, digits=3))")
        end
    end

    # Check for bank run equilibrium
    if τ_bar_IN_UNC == τ_bar_OUT_UNC
        # u is above the maximum of effective hazard rate - no bank run
        ξ = NaN
        bankrun = false
        converged = true
        tolerance_achieved = 0.0
        
        if verbose
            println("  No bank run equilibrium (optimal times coincide)")
        end
    else
        # Solve for crisis time ξ using standard approach
        if verbose
            println("  Solving for crisis time ξ...")
        end
        
        # Use provided guess or default
        # Note we use the same function as baseline
        ξ, tolerance_achieved = compute_ξ(τ_bar_IN_UNC, τ_bar_OUT_UNC, learning_cdf, κ;verbose=verbose)
        
        if isnan(ξ)
            bankrun = false
            converged = false
            tolerance_achieved = Inf
            
            if verbose
                println("  Crisis time solver did not converge")
            end
        else
            bankrun = true
            converged = true
            
            if verbose
                println("  Crisis time: ξ=$(round(ξ, digits=3))")
            end
        end
    end

    solve_time = time() - solve_start

    if verbose
        println("  Equilibrium solved in $(round(solve_time*1000, digits=1))ms")
    end

    return SolvedModelInterest(ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, bankrun, V,
                              model, lr, converged, solve_time, tolerance_achieved)
end

"""
    get_AW_functions_interest!(result::SolvedModelInterest)

Lazy evaluation of interest rate aggregate withdrawal functions.
Computes and caches AW functions on first call.

# Returns
- `NamedTuple`: Cached or newly computed AW functions for interest rate model
"""
function get_AW_functions_interest!(result::SolvedModelInterest)
    # Return cached result if already computed
    if result.aw[] !== nothing
        return result.aw[]
    end

    # Compute AW functions (expensive operation)
    # The value function only affects optimal buffer times, not AW calculation itself
    # So we can use baseline get_AW() directly
    if result.bankrun
        AW_cum_func, AW_OUT_func, AW_IN_func = get_AW(result.ξ, result.τ_bar_IN_UNC, result.τ_bar_OUT_UNC,
                                                       result.HR, result.learning_results.learning_cdf)
        AW_max = maximum(AW_cum_func.itp.coefs)

        # Cache the result
        aw_result = (AW_cum=AW_cum_func, AW_OUT=AW_OUT_func, AW_IN=AW_IN_func, AW_max=AW_max)
        result.aw[] = aw_result
        return result.aw[]
    else
        # No bank run - return nothing
        result.aw[] = nothing
        return nothing
    end
end

