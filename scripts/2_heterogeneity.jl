### HETEROGENEITY EXTENSION ###
# This script demonstrates the heterogeneity extension of "The Social Determinants of Bank Runs"
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

# Include heterogeneity extension modules
include(joinpath(@__DIR__, "..", "src", "extensions", "heterogeneity", "heterogeneity_model.jl"))
include(joinpath(@__DIR__, "..", "src", "extensions", "heterogeneity", "heterogeneity_learning.jl"))
include(joinpath(@__DIR__, "..", "src", "extensions", "heterogeneity", "heterogeneity_solver.jl"))

# Set plotting backend
gr()
default(fontfamily="Computer Modern")

# Set paths
plot_path = joinpath(@__DIR__, "..", "output", "figures", "heterogeneity/") * "/"
mkpath(plot_path)  # Create directory if doesn't exist

println("Starting heterogeneity extension demonstration")
println("="^60)


#########################################
# HETEROGENEOUS MODEL PARAMETERS
#########################################

# Define a simple two-group heterogeneous model
βs = [0.125, 12.5]   # Slow and fast learners
dist = [0.9, 0.1]    # Most slow, few fast

m_hetero = ModelParametersHetero(
    βs = βs,
    dist = dist,
    η_bar = 30.0,       # Raw awareness window
    u = 0.1,            # Utility flow from deposits
    p = 0.9,            # Prior probability bank is fragile
    κ = 0.3,            # Solvency threshold
    λ = 0.1,           # Exponential rate for t0
)

println("Heterogeneous model parameters:")
println(m_hetero)

#########################################
# SOLVE HETEROGENEOUS LEARNING
#########################################

println("\nSolving heterogeneous learning dynamics...")
lr_hetero = solve_SInetwork_hetero(m_hetero.learning)
println("Learning solved in $(round(lr_hetero.solve_time*1000, digits=1))ms")

#########################################
# SOLVE HETEROGENEOUS EQUILIBRIUM
#########################################

println("\nSolving heterogeneous equilibrium...")
result_hetero = solve_equilibrium_hetero(lr_hetero, m_hetero.economic; verbose=true)
println("Equilibrium solved in $(round(result_hetero.solve_time*1000, digits=1))ms")
println(result_hetero)

#########################################
# COMPUTE AGGREGATE WITHDRAWALS
#########################################

println("\nComputing aggregate withdrawal functions...")
aw_hetero = get_AW_functions_hetero!(result_hetero)

if aw_hetero !== nothing
    println("Max heterogeneous AW: $(round(aw_hetero.AW_max, digits=3))")
else
    println("No bank run in heterogeneous model")
end

#########################################
# PLOTTING
#########################################

println("\nGenerating demonstration plots...")

# Define colors for groups (consistent across plots) - elegant darkish tones
colors = [:royalblue, :darkgreen, :mediumvioletred, :darkorange]
t_range = range(0, 2*result_hetero.ξ, length=1000)

# Plot: Aggregate withdrawal dynamics (if bank run occurs)
println("  Plot: Aggregate withdrawal dynamics...")

p1 = plot(legend=:topleft, grid=true)

# Plot total AW (matching baseline style)
plot!(p1, t_range, aw_hetero.AW_cum.(t_range),
        color=:darkred, linewidth=2, label="Total AW")

# Plot group-specific withdrawals (unscaled - showing per-group rates)
for k in 1:length(βs)
    plot!(p1, t_range, aw_hetero.AW_groups[k].(t_range),label="Group $k (β=$(βs[k]))",
            line=:dash, color=colors[k])
end

# Add threshold line (matching baseline style)
hline!(p1, [m_hetero.economic.κ], color=:grey, linestyle=:solid, linewidth=1, label=nothing)
annotate!(p1, [(result_hetero.ξ/2, m_hetero.economic.κ+0.015,
                text(L"κ = %$(round(m_hetero.economic.κ, digits=2))", 7, :left, :grey))])

# Add crisis time vertical line (matching baseline style)
vline!(p1, [result_hetero.ξ], color=:darkgoldenrod, linewidth=2, label=nothing)
# Position ξ annotation lower to stay within plot bounds
annotate!(p1, [(result_hetero.ξ+0.4, m_hetero.economic.κ * 0.85,
                text(L"ξ = %$(round(result_hetero.ξ, digits=1))", 7, :left, :darkgoldenrod))])

xlabel!("Time")
ylabel!("AW(t)")
title!("Aggregate Withdrawals - Heterogeneous Groups")

savefig(p1, plot_path * "aggregate_withdrawals_hetero.pdf")
println("    Saved: aggregate_withdrawals_hetero.pdf")

#########################################
# SUMMARY
#########################################

println("\n" * "="^60)
println("HETEROGENEITY EXTENSION COMPLETE")
println("Figures saved to: $plot_path")
println("="^60)
