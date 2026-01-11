### BASELINE REPLICATION SCRIPT ###
# This script reproduces all baseline figures in "The Social Determinants of Bank Runs"

# Author: Robin Lenoir
# Date: 2025

# Load required packages
using Plots, PGFPlotsX, LaTeXStrings
using DifferentialEquations, Interpolations, Statistics

# Include baseline modules
include(joinpath(@__DIR__, "..", "src", "baseline", "model.jl"))       # Baseline model structures
include(joinpath(@__DIR__, "..", "src", "baseline", "learning.jl"))    # Learning dynamics solver
include(joinpath(@__DIR__, "..", "src", "baseline", "solver.jl"))      # Equilibrium solver
include(joinpath(@__DIR__, "..", "src", "baseline", "plotting.jl"))    # Plotting functions

# Set plotting backend
gr()
default(fontfamily="Computer Modern")

# Set paths
plot_path = joinpath(@__DIR__, "..", "output", "figures", "baseline/") * "/"
mkpath(plot_path)  # Create directory if doesn't exist

println("Starting baseline replication for 'The Social Determinants of Bank Runs'")
println("="^60)


#########################################
# BASELINE MODEL PARAMETERS
#########################################

# Define main model that will be reused throughout
m_base = ModelParameters(
    β = 1.0,         # Communication speed
    η_bar = 15.0,    # Raw awareness window (η = η_bar/β = 15.0)
    u = 0.1,         # Utility flow from deposits
    p = 0.5,         # Prior probability bank is fragile
    κ = 0.6,         # Solvency threshold
    λ = 0.01,        # Exponential rate for t0
)

# We precompute the learning dynamics because they will be used many times
lr_base = solve_learning(m_base.learning)

# Display model parameters for verification
println("Main model parameters:")
println(m_base)

#########################################
# FIGURE 1: Learning Dynamics
#########################################
println("\nGenerating Figure 1: Learning Dynamics...")

# Generate learning dynamics for different β values using staged approach
β_values = [0.5, 1.0, 2.0]
learning_cdfs = []
learning_results = []  # Store LearningResults for potential reuse

for β in β_values
    # Create a learning model with specific β value
    m_fig1 = LearningParameters(β=β,tspan=(0.0,20.0),x0=0.0001)
    # Solve learning ODE and store
    lr = solve_learning(m_fig1)
    push!(learning_results, lr)
    push!(learning_cdfs, lr.learning_cdf)

    println("    β=$β: solved in $(round(lr.solve_time*1000, digits=1)) ms")
end

# Create plot
p_learning = plot_learning_distribution(learning_cdfs, (0.0, 20.0), β_values)
savefig(p_learning, plot_path * "learning_dynamics.pdf")
println("  ✓ Figure 1 saved")

#########################################
# FIGURES 2 & 3: Main Equilibrium + Hazard Rate Decomposition
#########################################
# We do figure 3 first so that figure 2 parameters correspond to a real equilibrium
println("\nGenerating Figures 2 & 3: Main Equilibrium and Hazard Rate...")

result = solve_equilibrium_baseline(lr_base, m_base.economic)
println("  Main equilibrium: ξ* = $(round(result.ξ, digits=2)), bankrun = $(result.bankrun)")

# Access AW properties
get_AW_functions!(result) #this adds the the full AW curve to the result struct, which doesn't computes them automatically
println("  Max withdrawals: $(round(result.aw[].AW_max, digits=3))")

# Verify participation constraint
println("\n")
check_participation_verbose(result)
println("\n")

# Figure 3: Equilibrium dynamics plot
p_eq = plot_equilibrium(result; x_range=(0, 15))
savefig(p_eq, plot_path * "equilibrium_dynamics_main.pdf")
println("  ✓ Figure 3 saved")

# Figure 2: Hazard rate decomposition (reuse computed ξ)
p_hazard = plot_hazard_rate_decomposition(result)
savefig(p_hazard, plot_path * "hazard_rate.pdf")
println("  ✓ Figure 2 saved")


#########################################
# FIGURES 3bis & 3ter: Alternative scenarios
#########################################
println("\nGenerating Figures 3bis and 3ter...")

#### 3bis: equilibrium dynamics with fast communication
m_fast = ModelParameters(m_base; β=3.0)
lr_fast = solve_learning(m_fast.learning)
result = solve_equilibrium_baseline(lr_fast, m_fast.economic)
println("  Fast communication equilibrium: ξ* = $(round(result.ξ, digits=2)), bankrun = $(result.bankrun)")

get_AW_functions!(result)
p_eq = plot_equilibrium(result; x_range=(0, 15))
savefig(p_eq, plot_path * "equilibrium_dynamics_fast.pdf")
println("  ✓ Figure 3bis saved")


#### 3ter: equilibrium dynamics with low u
m_low_u = ModelParameters(m_base; u=0.01)
lr_low_u = solve_learning(m_low_u.learning)
result = solve_equilibrium_baseline(lr_low_u, m_low_u.economic)
println("  Low utility equilibrium: ξ* = $(round(result.ξ, digits=2)), bankrun = $(result.bankrun)")

get_AW_functions!(result)
p_eq = plot_equilibrium(result; x_range=(0, 15))
savefig(p_eq, plot_path * "equilibrium_dynamics_low_u.pdf")
println("  ✓ Figure 3ter saved")

#########################################
# FIGURE 4: Comparative Statics in u
#########################################
println("\nGenerating Figure 4: Effect of Deposit Utility...")

# u values to sweep over
# I keep the paper resultion of 5000 points here
# It takes ~1 min to run
# You may want to set lower resolution for faster sweep
u_values = range(0.001, 0.2, length=5000)

# Initialize storage vectors
max_withdrawals = Float64[]
collapse_times = Float64[]
return_times = Float64[]
h_un_values = Float64[]  # Track uninformed hazard rate for participation boundary

# Loop over u values
# Early termination optimization: if no bank run for consecutive u values,
# stop computing (higher u means higher incentive to stay, less likely bank runs)
nan_threshold = 5  # Number of consecutive NaNs before early termination
consecutive_nan_count = 0
skipped_iterations = 0

for (i, u) in enumerate(u_values)

    # Stop after 5 NAs: non-run region
    global consecutive_nan_count, skipped_iterations
    if consecutive_nan_count >= nan_threshold
        # Fill remaining values with NaN and break
        remaining_count = length(u_values) - i + 1
        append!(max_withdrawals, fill(NaN, remaining_count))
        append!(collapse_times, fill(NaN, remaining_count))
        append!(return_times, fill(NaN, remaining_count))
        append!(h_un_values, fill(NaN, remaining_count))
        skipped_iterations = remaining_count
        println("    Early termination: skipped $skipped_iterations iterations (no-run region)")
        break
    end

    # Create model with specific u value
    m_u = ModelParameters(m_base; u=u)

    result_u = solve_equilibrium_baseline(lr_base, m_u.economic)

    get_AW_functions!(result_u)

    if result_u.bankrun
        # Compute max withdrawals
        push!(max_withdrawals, result_u.aw[].AW_max)
        push!(collapse_times, result_u.ξ)
        push!(return_times, result_u.ξ - result_u.τ_bar_IN_UNC)  # Return time

        # Compute uninformed hazard rate for participation check
        h_un = compute_h_un(result_u)
        push!(h_un_values, h_un(result_u.ξ))

        global consecutive_nan_count
        consecutive_nan_count = 0  # Reset counter on successful bank run
    else
        global consecutive_nan_count
        push!(max_withdrawals, NaN)
        push!(collapse_times, NaN)
        push!(return_times, NaN)
        push!(h_un_values, NaN)
        consecutive_nan_count += 1  # Increment counter for early termination
    end

    # Progress indicator
    if i % 100 == 0
        println("    Progress: $(round(100*i/length(u_values), digits=1))%")
    end
end

# Create comparative statics plot (original version)
p1, p2 = plot_comp_stat_withdrawals_and_collapse(u_values, max_withdrawals, collapse_times, m_base.economic.κ;
                                                    return_times=return_times)

savefig(p1, plot_path * "comp_stat_u_panel_a.pdf")
savefig(p2, plot_path * "comp_stat_u_panel_b.pdf")
println("  ✓ Figure 4 (original) saved")

# Create comparative statics plot with participation boundary shading
p1_part, p2_part = plot_comp_stat_with_participation(u_values, max_withdrawals, collapse_times,
                                                      m_base.economic.κ, h_un_values;
                                                      return_times=return_times)

savefig(p1_part, plot_path * "comp_stat_u_panel_a_with_participation.pdf")
savefig(p2_part, plot_path * "comp_stat_u_panel_b_with_participation.pdf")
println("  ✓ Figure 4 (with participation constraint) saved")

#########################################
# FIGURE 5: Heatmap - β vs u interaction
#########################################
println("\nGenerating Figure 5: β-u Interaction Heatmap (Peak Withdrawals)... \n Progress is for the total grid - it will stop around 10%, \n when we found the no-run equilibrium boundary")

# Create grid - use average meeting time
# The version in the paper use 5000*5000 grid for esthetics but it takes a couple hours to run
# For replication 500*500 is enough
ave_meeting_time = range(0.0001, stop=1, length=500)
β_vals = 1 ./ ave_meeting_time  # β = 1/average_meeting_time
u_vals = range(0.001, 1, length=500)
max_AW_matrix = zeros(length(u_vals), length(β_vals))
participation_matrix = fill(NaN, length(u_vals), length(β_vals))  # Track participation: 1=participate, 0=no participation

# Progress tracking
total_iterations = length(β_vals) * length(u_vals)
current_iter = 0
total_skipped_iterations = 0

# Early termination optimization parameters
nan_threshold = 5  # Number of consecutive NaNs before early termination per β


for (i, β) in enumerate(β_vals)
    # Create model with specific β (η will be computed as η_bar/β automatically)
    m_heatmap = ModelParameters(m_base; β=β)
    lr_heatmap = solve_learning(m_heatmap.learning)

    # Reset early termination counter for each β
    global consecutive_nan_count = 0

    for (j, u) in enumerate(u_vals)
        global current_iter += 1

        # Stop after 5 NAs: non-run region
        if consecutive_nan_count >= nan_threshold
            # Fill remaining u values for this β with NaN
            remaining_j = length(u_vals) - j + 1
            max_AW_matrix[j:end, i] .= NaN
            participation_matrix[j:end, i] .= NaN
            skipped_for_this_beta = remaining_j
            global total_skipped_iterations += skipped_for_this_beta
            global current_iter += skipped_for_this_beta - 1  # Adjust counter (we already incremented once)
            break
        end

        m_heatmap = ModelParameters(m_heatmap; u=u)

        result_heatmap = solve_equilibrium_baseline(lr_heatmap, m_heatmap.economic)

        get_AW_functions!(result_heatmap)

        if result_heatmap.bankrun
            # Compute max withdrawals using lazy evaluation
            max_AW_matrix[j, i] = result_heatmap.aw[].AW_max

            # Check participation constraint
            h_un = compute_h_un(result_heatmap)
            participates = u > h_un(result_heatmap.ξ)
            participation_matrix[j, i] = participates ? 1.0 : 0.0

            global consecutive_nan_count = 0  # Reset counter on successful bank run
        else
            max_AW_matrix[j, i] = NaN  # Set to NaN for missing equilibria
            participation_matrix[j, i] = NaN  # No participation data for non-run equilibria
            global consecutive_nan_count += 1  # Increment counter for early termination
        end

        # Progress indicator
        if current_iter % 1000 == 0
            progress_pct = round(100*current_iter/total_iterations, digits=1)
            println("    Progress: $progress_pct% (skipped $total_skipped_iterations total)")
        end
    end
end

# Report optimization results
if total_skipped_iterations > 0
    println("  Early termination: skipped $total_skipped_iterations total iterations (no-run region)")
end

# Convert missing values to NaN for heatmap
heatmap_data = map(x -> isnan(x) ? NaN : x, max_AW_matrix)

# Create heatmap using average meeting time (original version)
p_heatmap_AW = heatmap(ave_meeting_time, u_vals, heatmap_data,
                       xlabel="Average meeting time",
                       ylabel="Deposit Utility",
                       title="Peak Withdrawals",
                       color=:viridis, alpha=0.8,
                       xtickfont=font(10), ytickfont=font(10))
savefig(p_heatmap_AW, plot_path * "comp_stat_cross_heatmap_AW.pdf")
println("  ✓ Figure 5 (original) saved")

# Create heatmap with viridis colormap and participation boundary
p_heatmap_participation = heatmap(ave_meeting_time, u_vals, heatmap_data,
                                   xlabel="Average meeting time",
                                   ylabel="Deposit Utility",
                                   title="Peak Withdrawals (with Participation)",
                                   color=:viridis,
                                   alpha=0.8,
                                   xtickfont=font(10), ytickfont=font(10))

# Identify no-participation region
no_part_mask = participation_matrix .== 0.0

# Find participation boundary for contour line
# Extract boundary points where participation transitions
boundary_u = Float64[]
boundary_beta = Float64[]
for i in 1:length(ave_meeting_time)
    # Find transition point in u for this beta value
    for j in 2:length(u_vals)
        if !isnan(no_part_mask[j-1, i]) && !isnan(no_part_mask[j, i])
            if no_part_mask[j-1, i] && !no_part_mask[j, i]
                push!(boundary_beta, ave_meeting_time[i])
                push!(boundary_u, u_vals[j])
                break
            end
        end
    end
end

# Add thin boundary line
if !isempty(boundary_beta)
    plot!(p_heatmap_participation, boundary_beta, boundary_u,
          color=:black, linewidth=1, linestyle=:solid, label="", alpha=0.7)
end

# Add diagonal hatching to no-participation region
# Create diagonal lines with appropriate spacing
line_spacing = (maximum(ave_meeting_time) - minimum(ave_meeting_time)) / 25  # Adjust for density

for i in 1:length(ave_meeting_time)
    for j in 1:length(u_vals)
        if !isnan(no_part_mask[j, i]) && no_part_mask[j, i]
            # Draw short diagonal line segments in hatching pattern
            # Only draw every Nth point to avoid overcrowding
            if (i + j) % 40 == 0  # Spacing control
                x1 = ave_meeting_time[i]
                y1 = u_vals[j]
                # Diagonal line at ~45 degrees
                dx = line_spacing * 0.4
                dy = (maximum(u_vals) - minimum(u_vals)) / length(u_vals) * 0.6
                plot!(p_heatmap_participation, [x1-dx, x1+dx], [y1-dy, y1+dy],
                      color=:black, linewidth=0.5, alpha=0.4, label="")
            end
        end
    end
end

savefig(p_heatmap_participation, plot_path * "comp_stat_cross_heatmap_AW_with_participation.pdf")
println("  ✓ Figure 5 (with participation boundary) saved")

# Create categorical heatmap showing just the three regions as solid blocks
println("\nGenerating Figure 5bis: Participation Regions...")

# Create region classification matrix
# 1 = No participation, 2 = Participation + run, 3 = No run
region_matrix = fill(NaN, length(u_vals), length(β_vals))

for i in 1:length(ave_meeting_time)
    for j in 1:length(u_vals)
        if isnan(heatmap_data[j, i])
            # No run equilibrium
            region_matrix[j, i] = 3.0
        elseif !isnan(no_part_mask[j, i]) && no_part_mask[j, i]
            # No participation
            region_matrix[j, i] = 1.0
        else
            # Participation + run
            region_matrix[j, i] = 2.0
        end
    end
end

# Create custom colormap matching Figure 4 colors
using Colors
red_orange_green = [RGB(1.0, 0.7, 0.7),    # Light red - no participation
                    RGB(1.0, 0.9, 0.7),    # Light orange - participation + run
                    RGB(0.8, 1.0, 0.8)]    # Light green - no run

p_regions = heatmap(ave_meeting_time, u_vals, region_matrix,
                    xlabel="Average meeting time",
                    ylabel="Deposit Utility",
                    title="Participation Regions",
                    color=cgrad(red_orange_green, 3, categorical=true),
                    colorbar=false,
                    clims=(0.5, 3.5),
                    xtickfont=font(10), ytickfont=font(10))

savefig(p_regions, plot_path * "comp_stat_cross_regions.pdf")
println("  ✓ Figure 5bis (participation regions) saved")

# Create gradient version: solid colors for regions 1 & 3, gradient for region 2
println("\nGenerating Figure 5ter: Participation Regions with Gradient...")

# Create modified data matrix for gradient visualization
gradient_matrix = fill(NaN, length(u_vals), length(β_vals))

# Get min/max of peak withdrawals in region 2 for normalization
region2_values = Float64[]
for i in 1:length(ave_meeting_time)
    for j in 1:length(u_vals)
        if !isnan(heatmap_data[j, i]) && (!isnan(no_part_mask[j, i]) && !no_part_mask[j, i])
            # This is region 2: participation + run
            push!(region2_values, heatmap_data[j, i])
        end
    end
end

min_region2 = minimum(region2_values)
max_region2 = maximum(region2_values)

# Populate gradient matrix
for i in 1:length(ave_meeting_time)
    for j in 1:length(u_vals)
        if isnan(heatmap_data[j, i])
            # Region 3: No run - map to 1.0 (light green end)
            gradient_matrix[j, i] = 1.0
        elseif !isnan(no_part_mask[j, i]) && no_part_mask[j, i]
            # Region 1: No participation - map to 0.0 (light red end)
            gradient_matrix[j, i] = 0.0
        else
            # Region 2: Participation + run - normalize to [0, 1] based on peak withdrawals
            normalized = (heatmap_data[j, i] - min_region2) / (max_region2 - min_region2)
            gradient_matrix[j, i] = normalized
        end
    end
end

# Create red-to-green gradient colormap
red_to_green_gradient = cgrad([RGB(1.0, 0.7, 0.7),    # Light red
                                RGB(1.0, 0.9, 0.7),    # Light orange (middle)
                                RGB(0.8, 1.0, 0.8)])   # Light green

p_gradient = heatmap(ave_meeting_time, u_vals, gradient_matrix,
                     xlabel="Average meeting time",
                     ylabel="Deposit Utility",
                     title="Participation Regions (with Gradient)",
                     color=red_to_green_gradient,
                     colorbar=true,
                     clims=(0.0, 1.0),
                     xtickfont=font(10), ytickfont=font(10))

savefig(p_gradient, plot_path * "comp_stat_cross_regions_gradient.pdf")
println("  ✓ Figure 5ter (participation regions with gradient) saved")

# Extract participation boundary and plot max AW along it
println("\nGenerating Figure 5quater: Peak Withdrawals Along Participation Boundary...")

# Extract boundary points with their corresponding max AW values
boundary_meeting_times = Float64[]
boundary_u_vals = Float64[]
boundary_max_AW = Float64[]

for i in 1:length(ave_meeting_time)
    # Find transition point in u for this beta value
    for j in 2:length(u_vals)
        if !isnan(no_part_mask[j-1, i]) && !isnan(no_part_mask[j, i])
            if no_part_mask[j-1, i] && !no_part_mask[j, i]
                # Found boundary point
                push!(boundary_meeting_times, ave_meeting_time[i])
                push!(boundary_u_vals, u_vals[j])
                # Get the max AW at this boundary point
                push!(boundary_max_AW, heatmap_data[j, i])
                break
            end
        end
    end
end

# Sort by average meeting time (1/β)
sort_indices = sortperm(boundary_meeting_times)
sorted_meeting_times = boundary_meeting_times[sort_indices]
sorted_u = boundary_u_vals[sort_indices]
sorted_max_AW = boundary_max_AW[sort_indices]

# Create two-panel figure
p_boundary_1 = plot(sorted_meeting_times, sorted_max_AW,
                    xlabel="Average meeting time (1/β)",
                    ylabel="Peak Withdrawals at Boundary",
                    title="(a) Peak Withdrawals Along Participation Boundary",
                    label="", color=:darkred,
                    marker=:circle, markersize=3,
                    legend=false)

p_boundary_2 = plot(sorted_u, sorted_max_AW,
                    xlabel="Deposit Utility (u) at Boundary",
                    ylabel="Peak Withdrawals at Boundary",
                    title="(b) Peak Withdrawals vs u at Boundary",
                    label="", color=:darkblue,
                    marker=:circle, markersize=3,
                    legend=false)

p_boundary_combined = plot(p_boundary_1, p_boundary_2, layout=(1,2), size=(1200, 400))

savefig(p_boundary_combined, plot_path * "comp_stat_boundary_analysis.pdf")
println("  ✓ Figure 5quater (boundary analysis) saved")

#########################################
# SUMMARY
#########################################

println("\n" * "="^60)
println("BASELINE REPLICATION COMPLETE")
println("All baseline figures saved to: $plot_path")
println("="^60)
println("\nBaseline Figure List:")
println("  1. learning_dynamics.pdf")
println("  2. hazard_rate.pdf")
println("  3. equilibrium_dynamics_main.pdf")
println("  3bis. equilibrium_dynamics_fast.pdf")
println("  3ter. equilibrium_dynamics_low_u.pdf")
println("  4a. comp_stat_u_panel_a.pdf")
println("  4b. comp_stat_u_panel_b.pdf")
println("  4a_part. comp_stat_u_panel_a_with_participation.pdf")
println("  4b_part. comp_stat_u_panel_b_with_participation.pdf")
println("  5. comp_stat_cross_heatmap_AW.pdf")
println("  5_part. comp_stat_cross_heatmap_AW_with_participation.pdf")
println("  5bis. comp_stat_cross_regions.pdf")
println("  5ter. comp_stat_cross_regions_gradient.pdf")
println("  5quater. comp_stat_boundary_analysis.pdf")
