### SOCIAL LEARNING EXTENSION ###
# This script demonstrates the social learning extension of "The Social Determinants of Bank Runs"
# Author: Robin Lenoir
# Date: 2025

# Load required packages
using Plots, PGFPlotsX, LaTeXStrings
using DifferentialEquations, Interpolations, Statistics, LinearAlgebra

# Include baseline modules for shared utilities
include(joinpath(@__DIR__, "..", "src", "baseline", "model.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline", "learning.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline", "solver.jl"))
include(joinpath(@__DIR__, "..", "src", "baseline", "plotting.jl"))

# Include social learning extension modules
include(joinpath(@__DIR__, "..", "src", "extensions", "social_learning", "social_learning_dynamics.jl"))
include(joinpath(@__DIR__, "..", "src", "extensions", "social_learning", "social_learning_solver.jl"))

# Set plotting backend
gr()
default(fontfamily="Computer Modern")

# Set paths
plot_path = joinpath(@__DIR__, "..", "output", "figures", "social_learning/") * "/"
mkpath(plot_path)  # Create directory if doesn't exist

println("Starting social learning extension demonstration")
println("="^60)

#########################################
# SOCIAL LEARNING MODEL PARAMETERS
#########################################

# Define model parameters for social learning
m_social = ModelParameters(
    β = .9,         # Communication speed (now learning from withdrawals)
    η_bar = 30.0,    # Raw awareness window
    u = 0.5,         # Utility flow from deposits
    p = 0.99,        # Prior probability bank is fragile
    κ = 0.25,         # Solvency threshold
    λ = 0.25,        # Exponential rate for t0
)

println("Social learning model parameters:")
println(m_social)

#########################################
# SOLVE SOCIAL LEARNING EQUILIBRIUM
#########################################

println("\nSolving social learning equilibrium...")
println("This involves fixed-point iteration between learning and withdrawals...")

result_social = solve_equilibrium_social_learning(m_social;
                                                  tol=1e-4, max_iter=500, verbose=true)

println("\nFinal social learning equilibrium:")
println(result_social)

#########################################
# COMPARISON WITH BASELINE (WORD-OF-MOUTH)
#########################################

println("\nComparing with baseline model (word-of-mouth learning)...")

# Solve baseline model for comparison
lr_baseline = solve_learning(m_social.learning)
result_baseline = solve_equilibrium_baseline(lr_baseline, m_social.economic)

println("Comparison results:")
social_xi = result_social.bankrun ? round(result_social.ξ, digits=2) : "No run"
baseline_xi = result_baseline.bankrun ? round(result_baseline.ξ, digits=2) : "No run"
println("  Social learning: ξ* = $social_xi, bankrun = $(result_social.bankrun)")
println("  Baseline (WOM): ξ* = $baseline_xi, bankrun = $(result_baseline.bankrun)")

if result_social.bankrun && result_baseline.bankrun
    Δξ = result_social.ξ - result_baseline.ξ
    timing = Δξ > 0 ? "later" : "earlier"
    println("  Crisis time difference: Δξ* = $(round(Δξ, digits=3)) ($timing with social learning)")
end

#########################################
# COMPUTE AGGREGATE WITHDRAWALS
#########################################

println("\nComputing aggregate withdrawal functions...")
aw_social = get_AW_functions!(result_social)
get_AW_functions!(result_baseline)

if aw_social !== nothing
    println("Max social learning AW: $(round(aw_social.AW_max, digits=3))")
else
    println("No bank run in social learning model")
end

#########################################
# PLOT EQUILIBRIUM DYNAMICS
#########################################

println("\nGenerating equilibrium plots...")

# Plot social learning equilibrium
if result_social.bankrun
    p_social = plot_equilibrium(result_social)
    savefig(p_social, plot_path * "social_learning_equilibrium.pdf")
    println("  ✓ Social learning equilibrium plot saved")
else
    println("  ! No social learning equilibrium to plot (no bank run)")
end

# Plot baseline equilibrium
if result_baseline.bankrun
    p_baseline = plot_equilibrium(result_baseline)
    savefig(p_baseline, plot_path * "baseline_equilibrium.pdf")
    println("  ✓ Baseline equilibrium plot saved")
else
    println("  ! No baseline equilibrium to plot (no bank run)")
end

#########################################
# SUMMARY
#########################################

println("\n" * "="^60)
println("SOCIAL LEARNING EXTENSION COMPLETE")
println("Figures saved to: $plot_path")
println("="^60)
