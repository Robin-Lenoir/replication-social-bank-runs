#=
social_learning_solver.jl

Fixed-point equilibrium solver for the social learning extension.
Handles the feedback between learning and aggregate withdrawals.

Author: Robin Lenoir
=#

# Include required dependencies
using Interpolations, LinearAlgebra, Statistics

# Include model definitions and dynamics
include(joinpath(@__DIR__, "social_learning_dynamics.jl"))
include(joinpath(@__DIR__, "..", "..", "baseline", "learning.jl")) #Needed for intial guess

# Include baseline equilibrium utilities
include(joinpath(@__DIR__, "..", "..", "baseline", "solver.jl"))

"""
    solve_equilibrium_social_learning(model::ModelParameters;
                                     tol=1e-4, max_iter=250, verbose=false,
                                     init_out=0.0, learning_tol=1e-12)

Solve social learning equilibrium using fixed-point iteration.
Returns a complete SolvedModel object like the baseline solver.

# Algorithm
```
1. Initialize: AW_cum^(0) from baseline SI learning (word-of-mouth)
2. For iteration n = 1, 2, ...:
   a) Solve learning: I_k^(n) from AW_cum^(n-1)
   b) Solve equilibrium: ξ^(n) from G_k^(n)
   c) Compute withdrawals: AW_cum^(n) from ξ^(n), G_k^(n)
   d) Check convergence: ||AW_cum^(n) - AW_cum^(n-1)||_∞ < tol
3. Extract final equilibrium and return SolvedModel
```
See discussion in Appendix C.


# Arguments
- `model::ModelParameters`: Complete model parameters (β, u, p, κ, etc.)

# Keyword Arguments
- `tol::Float64 = 1e-4`: Fixed-point convergence tolerance
- `max_iter::Int = 250`: Maximum iterations
- `verbose::Bool = false`: Print iteration details
- `init_out::Float64 = 0.0`: Exogenous initial outflow

# Returns
- `SolvedModel`: Complete equilibrium solution with all results and metadata
  The solve_time includes the full fixed-point iteration time
  Convergence status reflects both fixed-point and equilibrium convergence

# Example
```julia
model = ModelParameters(β=1.0, u=0.1, κ=0.6)
result = solve_equilibrium_social_learning(model; verbose=true)
# Access equilibrium: result.ξ, result.τ_bar_IN_UNC, etc.
# Plot directly: plot_equilibrium(result)
```
"""
function solve_equilibrium_social_learning(model::ModelParameters;
                                          tol=1e-4, max_iter=250, verbose=false,
                                          init_out=0.0, learning_tol=1e-12)
    solve_start = time()

    # Extract parameters - homogeneous model
    β = model.learning.β
    x0 = model.learning.x0[1]  # Extract scalar

    # Economic parameters
    (; η, u, p, κ, λ) = model.economic

    # Override tspan to [0, η] - all dynamics only relevant within awareness window
    tspan = (0.0, η)

    if verbose
        println("Solving social learning equilibrium...")
        println("  Parameters: β=$β, u=$u, κ=$κ, max_iter=$max_iter, tol=$tol")
        println("  Time span: [0, η=$(round(η, digits=2))]")
    end

    # Step 1: Initialize with baseline SI learning (word-of-mouth)
    if verbose
        println("  Step 1: Computing initial guess from baseline SI learning...")
    end

    # Guess: word-of-mouth learning as initial AW
    learning_params_guess = LearningParameters(β, tspan, x0)
    lr_guess = solve_learning(learning_params_guess)

    # Initial AW is just the learning CDF (everyone exits immediately)
    AW_cum_guess = LinearInterpolation(lr_guess.grid, lr_guess.learning_cdf.(lr_guess.grid))

    if verbose
        println("    Initial grid size: $(length(lr_guess.grid)) points")
        println("    Max initial AW: $(round(maximum(lr_guess.learning_cdf.(lr_guess.grid)), digits=3))")
    end

    # Fixed comparison grid for convergence checking
    # Use fixed grid to avoid spurious error from grid mismatch between iterations
    # Each iteration uses adaptive ODE grids (optimal for accuracy), but we compare
    # the interpolated functions at the same points to get clean convergence measure
    grid_comparison = range(0.0, η, length=1000)

    # Step 2: Fixed-point iteration
    if verbose
        println("  Step 2: Starting fixed-point iteration...")
    end

    AW_cum_new = AW_cum_guess
    ξ_new = 0.0
    learning_cdf_new = lr_guess.learning_cdf
    converged = false

    # Initialize result_temp (will be updated each iteration)
    result_temp = nothing

    for iter in 1:max_iter
        ξ_old = ξ_new
        AW_cum_old = AW_cum_new

        if verbose && (iter % 10 == 1 || iter <= 5)
            println("    Iteration $iter:")
        end

        # (a) Solve learning from withdrawals
        t_learning_start = time()
        learning_cdf_new, sol = solve_ODE_social_learning(β, AW_cum_old, x0, tspan)
        CDF_grid = learning_cdf_new.itp.knots[1]
        t_learning = time() - t_learning_start

        # Create LearningResults structure for baseline equilibrium solver
        learning_pdf_new = compute_pdf_social_learning(β, learning_cdf_new, AW_cum_old, CDF_grid)
        temp_learning_params = LearningParameters(β, tspan, x0)
        temp_lr = LearningResults(temp_learning_params, learning_cdf_new, learning_pdf_new, CDF_grid, 0.0, sol)

        # (b) Solve equilibrium from new learning
        t_equilibrium_start = time()
        result_temp = solve_equilibrium_baseline(temp_lr, model.economic; verbose=false)
        t_equilibrium = time() - t_equilibrium_start

        if verbose && (iter % 10 == 1 || iter <= 5)
            println("      Timing: learning=$(round(t_learning*1000, digits=1))ms, " *
                    "equilibrium=$(round(t_equilibrium*1000, digits=1))ms")
        end

        if !result_temp.bankrun ##Case where no equilibrium with that FIXED learning curve
            if verbose
                println("      No equilibrium found in iteration $iter")
            end

            # Use small increment and average with old AW
            ξ_new = ξ_old + η/500
            if ξ_new > η
                if verbose
                    println("  Search exceeded η, stopping iteration")
                end
                break
            end

            # Compute new AW with incremented ξ
            AW_cum_new_func, _, _ = get_AW(ξ_new, result_temp.τ_bar_IN_UNC, result_temp.τ_bar_OUT_UNC,
                                          result_temp.HR, learning_cdf_new)

            # Check convergence BEFORE damping (on undamped version)
            error_norm_undamped = norm([AW_cum_new_func(t) for t in grid_comparison] -
                                       [AW_cum_old(t) for t in grid_comparison], Inf)

            if error_norm_undamped < tol
                # Converged! Use undamped version
                AW_cum_new = AW_cum_new_func
                if verbose
                    println("      No equilibrium case: Converged with undamped AW")
                    println("      Error = $(round(error_norm_undamped, digits=8))")
                end
                converged = true
                break
            end

            # Not converged - apply uniform damping
            α = 0.5
            values_old = [AW_cum_old(t) for t in CDF_grid]
            values_new = [AW_cum_new_func(t) for t in CDF_grid]
            damped_values = (1.0 - α) .* values_old .+ α .* values_new
            AW_cum_new = LinearInterpolation(CDF_grid, damped_values)

            if verbose && (iter % 10 == 1 || iter <= 5)
                println("      No equilibrium: damped with α = $(α)")
            end

        else ## Case where an equilibrium was found with that fixed learning
            # Successful equilibrium
            ξ_new = result_temp.ξ

            # Compute new aggregate withdrawals
            AW_cum_new_func, _, _ = get_AW(ξ_new, result_temp.τ_bar_IN_UNC, result_temp.τ_bar_OUT_UNC,
                                          result_temp.HR, learning_cdf_new)

            # Check convergence BEFORE damping (on undamped version)
            error_norm_undamped = norm([AW_cum_new_func(t) for t in grid_comparison] -
                                       [AW_cum_old(t) for t in grid_comparison], Inf)

            if verbose && (iter % 10 == 1 || iter <= 5)
                println("      ξ = $(ξ_new), AW error = $(error_norm_undamped)")
            end

            if error_norm_undamped < tol
                # Converged! Use undamped version
                AW_cum_new = AW_cum_new_func
                if verbose
                    println("  Convergence reached after $iter iterations")
                    println("    Final ξ = $(round(ξ_new, digits=3))")
                    println("    Final AW error = $(round(error_norm_undamped, digits=8))")
                    println("    Max AW = $(round(maximum([AW_cum_new_func(t) for t in grid_comparison]), digits=3))")
                end
                converged = true
                break
            end

            # Not converged - apply uniform damping
            α = 0.5
            values_old = [AW_cum_old(t) for t in CDF_grid]
            values_new = [AW_cum_new_func(t) for t in CDF_grid]
            damped_values = (1.0 - α) .* values_old .+ α .* values_new
            AW_cum_new = LinearInterpolation(CDF_grid, damped_values)

            if verbose && (iter % 10 == 1 || iter <= 5)
                println("      Damped with α = $(α)")
            end
        end

        # Check for max iterations
        if iter == max_iter
            if verbose
                println("  Did not converge after $max_iter iterations")
                error_norm_final = norm([AW_cum_new(t) for t in grid_comparison] -
                                       [AW_cum_old(t) for t in grid_comparison], Inf)
                println("    Final AW error = $(round(error_norm_final, digits=6))")
            end
            break
        end
    end

    solve_time = time() - solve_start

    if verbose
        println("  Social learning completed in $(round(solve_time*1000, digits=1))ms")
    end

    if result_temp === nothing
        # Edge case: no iterations completed (shouldn't happen with max_iter > 0)
        error("Social learning solver failed: no iterations completed")
    end

    if verbose
        println("  Final result: ξ = $(result_temp.bankrun ? round(result_temp.ξ, digits=3) : "No run")")
        println("  Convergence status: $converged")
    end

    return result_temp
end


