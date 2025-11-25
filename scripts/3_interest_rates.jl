### INTEREST RATES EXTENSION ###
# This script demonstrates the interest rates extension of "The Social Determinants of Bank Runs"
# Author: Robin Lenoir
# Date: 2025

# Load required packages
using Plots, PGFPlotsX, LaTeXStrings
using DifferentialEquations, Interpolations, Statistics

# Include baseline modules for shared utilities
include(joinpath(@__DIR__, "..", "src", "baseline", "model.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline", "learning.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline", "solver.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline", "plotting.jl"))

# Include interest rates extension modules
include(joinpath(@__DIR__, "..", "src", "extensions", "interest_rates", "interest_rate_model.jl"))
include(joinpath(@__DIR__, "..", "src", "extensions", "interest_rates", "value_function_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "extensions", "interest_rates", "interest_rate_solver.jl"))

# Set plotting backend
gr()
default(fontfamily="Computer Modern")

# Set paths
plot_path = joinpath(@__DIR__, "..", "output", "figures", "interest_rates/") * "/"
mkpath(plot_path)  # Create directory if doesn't exist

println("Starting interest rates extension demonstration")
println("="^60)

#########################################
# INTEREST RATE MODEL PARAMETERS
#########################################

# Define interest rate model with positive r and δ
m_interest = ModelParametersInterest(
    β = 1.0,         # Communication speed
    η_bar = 15.0,    # Raw awareness window
    u = 0.0,         # utility flow from deposits
    p = 0.5,         # Prior probability bank is fragile
    κ = 0.6,         # Solvency threshold
    λ = 0.01,        # Exponential rate for t0
    r = 0.06,        # Interest rate (positive!)
    δ = 0.1,        # Recovery/discount rate
)

println("Interest rate model parameters:")
println(m_interest)

#########################################
# SOLVE INTEREST RATE LEARNING
#########################################

println("\nSolving learning dynamics (same as baseline)...")
lr_interest = solve_learning(m_interest.learning)
println("Learning solved in $(round(lr_interest.solve_time*1000, digits=1))ms")

#########################################
# SOLVE INTEREST RATE EQUILIBRIUM
#########################################

println("\nSolving interest rate equilibrium...")
result_interest = solve_equilibrium_interest(lr_interest, m_interest.economic, m_interest; verbose=true)
println(result_interest)

#########################################
# COMPUTE AGGREGATE WITHDRAWALS
#########################################

println("\nComputing aggregate withdrawal functions...")
get_AW_functions_interest!(result_interest)

#########################################
# PLOTTING
#########################################

println("\nGenerating demonstration plots...")

# Plot 1: Value function (when r > 0)
if result_interest.V !== nothing
    println("  Plot 1: Value function...")

    p1 = plot(title="Value Function",
             xlabel="Time", ylabel="Value V(t)",
             legend=:topleft, grid=true)

    # Transform to normal time: t = ξ - τ̄
    ξ = result_interest.ξ
    τ_range = range(0, min(m_interest.economic.η, maximum(result_interest.V.itp.knots[1])), length=500)
    t_values = ξ .- τ_range  # Normal time
    V_values = result_interest.V.(τ_range)

    # Filter to start at t=0
    valid_idx = t_values .>= 0
    t_plot = t_values[valid_idx]
    V_plot = V_values[valid_idx]

    # Reverse to plot in chronological order
    plot!(p1, reverse(t_plot), reverse(V_plot),
          color=:royalblue, linewidth=2, label="V(t)")

    # Add terminal value
    V_terminal = m_interest.economic.δ / (m_interest.economic.δ - m_interest.economic.r)
    hline!(p1, [V_terminal], color=:darkgray, linestyle=:dash, linewidth=1,
           label="Terminal value = $(round(V_terminal, digits=2))")

    # Set x-axis to start at 0
    xlims!(p1, (0, maximum(t_plot)))

    savefig(p1, plot_path * "value_function.pdf")
    println("    Saved: value_function.pdf")
end

# Plot 2: Hazard rate decomposition (following baseline style)
println("  Plot 2: Hazard rate decomposition...")

ξ = result_interest.ξ
(; p, λ, η) = result_interest.model_params.economic
learning_pdf = result_interest.learning_results.learning_pdf

# Compute hazard rate decomposition like in baseline
# h̄_f(τ) = hazard rate when bank is definitely fragile (p=1)
HR_fragile = hazard_rate(1.0, λ, learning_pdf, η)

# h(τ) = total hazard rate with uncertainty about fragility
HR_total = hazard_rate(p, λ, learning_pdf, η)

# Evaluate at grid points in reversed time
τ_grid = range(0, min(η, ξ), length=1000)
h_bar_f_vals = HR_fragile.(τ_grid)
h_vals = HR_total.(τ_grid)

# Derive π(τ) = h(τ) / h̄_f(τ) with safe division
π_vals = h_vals ./ h_bar_f_vals
π_vals = [isnan(v) ? 0.0 : clamp(v, 0.0, 1.0) for v in π_vals]

# Transform to normal time
t_values = clamp.(ξ .- τ_grid, 0.0, 1.3*ξ)

# Compute rV + u threshold if value function exists
if result_interest.V !== nothing
    rV_vals = m_interest.economic.r .* result_interest.V.(τ_grid)
    threshold_vals = rV_vals .+ m_interest.economic.u
end

mid_h_bar = h_bar_f_vals[div(length(h_bar_f_vals), 2)]
# Create plot
p2 = plot(title=L"h(\tau) = \pi(\tau) \times h_f(\tau)",
         xlabel="Time", ylabel="Hazard Rate",
         legend=:topleft, grid=true,
         ylims=(0,mid_h_bar *1.2),
         xlims=(0, 1.2*ξ))

# Plot components (reversed for chronological order)
plot!(p2, reverse(t_values), reverse(h_vals),
      label=L"h(\tau) \textrm{~-~Total~hazard}",
      lw=1.5, color=:mediumvioletred)

plot!(p2, reverse(t_values), reverse(π_vals),
      label=L"\pi(\tau) \textrm{~-~Belief~fragile}",
      lw=1, color=:royalblue)

plot!(p2, reverse(t_values), reverse(h_bar_f_vals),
      label=L"h_f(\tau) \textrm{~-~Conditional~hazard}",
      lw=1, color=:tomato)

# Add rV + u threshold instead of just u
if result_interest.V !== nothing
    plot!(p2, reverse(t_values), reverse(threshold_vals),
          color=:darkgray, lw=1, linestyle=:solid, label="")
    mid_thresh = threshold_vals[div(length(threshold_vals), 2)]
    annotate!(p2, 0.7*ξ, 1.15*mid_thresh,
             text(L"rV(\tau)", 10, :darkgray, :left)) #write rV because u=0 in example
end

# Add collapse time vertical line
vline!(p2, [ξ], color=:darkgoldenrod, lw=1.5, linestyle=:dashdot, label="")
mid_h_bar = h_bar_f_vals[div(length(h_bar_f_vals), 2)]
annotate!(p2, 1.08*ξ, mid_h_bar, text(L"\xi=%$(round(ξ, digits=1))", 10, :darkgoldenrod, :center))

savefig(p2, plot_path * "hazard_decomposition.pdf")
println("    Saved: hazard_decomposition.pdf")

#########################################
# SUMMARY
#########################################

println("\n" * "="^60)
println("INTEREST RATES EXTENSION COMPLETE")
println("Figures saved to: $(abspath(plot_path))")
println("="^60)
