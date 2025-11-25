### MASTER REPLICATION SCRIPT ###
# This script runs all replication scripts and generates the complete figure package
# for "The Social Determinants of Bank Runs"
#
# Author: Robin Lenoir
# Date: 2025

println("="^80)
println(" " * "MASTER REPLICATION SCRIPT")
println(" " * "The Social Determinants of Bank Runs")
println(" " * "Robin Lenoir (2025)")
println("="^80)
println()

# Track execution time
master_start = time()

# Track which figures are generated
generated_figures = String[]

#########################################
# 1. BASELINE REPLICATION
#########################################
println("\n" * "="^80)
println("STEP 1/4: Running Baseline Replication")
println("="^80)

include(joinpath(@__DIR__, "scripts", "1_baseline.jl"))

# Record generated baseline figures
baseline_figs = [
    "baseline/learning_dynamics.pdf",
    "baseline/hazard_rate.pdf",
    "baseline/equilibrium_dynamics_main.pdf",
    "baseline/equilibrium_dynamics_fast.pdf",
    "baseline/equilibrium_dynamics_low_u.pdf",
    "baseline/comp_stat_u_panel_a.pdf",
    "baseline/comp_stat_u_panel_b.pdf",
    "baseline/comp_stat_cross_heatmap_AW.pdf"
]
append!(generated_figures, baseline_figs)

#########################################
# 2. HETEROGENEITY EXTENSION
#########################################
println("\n" * "="^80)
println("STEP 2/4: Running Heterogeneity Extension")
println("="^80)

include(joinpath(@__DIR__, "scripts", "2_heterogeneity.jl"))

# Record generated heterogeneity figures
hetero_figs = [
    "heterogeneity/aggregate_withdrawals_hetero.pdf"
]
append!(generated_figures, hetero_figs)

#########################################
# 3. INTEREST RATES EXTENSION
#########################################
println("\n" * "="^80)
println("STEP 3/4: Running Interest Rates Extension")
println("="^80)

include(joinpath(@__DIR__, "scripts", "3_interest_rates.jl"))

# Record generated interest rate figures
interest_figs = [
    "interest_rates/value_function.pdf",
    "interest_rates/hazard_decomposition.pdf"
]
append!(generated_figures, interest_figs)

#########################################
# 4. SOCIAL LEARNING EXTENSION
#########################################
println("\n" * "="^80)
println("STEP 4/4: Running Social Learning Extension")
println("="^80)

include(joinpath(@__DIR__, "scripts", "4_social_learning.jl"))

# Record generated social learning figures
social_figs = [
    "social_learning/social_learning_equilibrium.pdf",
    "social_learning/baseline_equilibrium.pdf"
]
append!(generated_figures, social_figs)

#########################################
# FINAL SUMMARY
#########################################
master_time = time() - master_start

println("\n" * "="^80)
println("REPLICATION COMPLETE!")
println("="^80)
println()
println("Total execution time: $(round(master_time, digits=1)) seconds")
println()
println("Generated $(length(generated_figures)) figures:")
for fig in generated_figures
    println("  ✓ output/figures/$fig")
end
println()
println("Output files:")
println("  ✓ output/replication_figures.tex (LaTeX document with all figures)")
println()
println("To view figures: cd output && pdflatex replication_figures.tex")
println("="^80)
