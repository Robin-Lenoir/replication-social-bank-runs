### Plotting Functions ###
# This file contains all plotting and visualization functions
# Part of the Social Bank Runs replication package
# Author: Robin Lenoir

#########################################
# LEARNING DISTRIBUTION PLOTS
#########################################

"""
    plot_learning_distribution(learning_cdf, tspan, β_values; labels=nothing)

Plot learning CDFs for different parameter values.

# Arguments
- `learning_cdf`: Array of CDF functions
- `tspan`: Time span
- `β_values`: Learning rate values (for labels)
- `labels`: Optional custom labels

# Returns
- Plot object
"""
function plot_learning_distribution(learning_cdf, tspan, β_values; labels=nothing)
    p = plot(legend=:bottomright, grid=true,
             xlabel="Time", ylabel="Fraction Informed",
             title="Learning Dynamics")

    t_values = range(tspan[1], tspan[2], length=1000)
    colors = [:blue, :red, :green, :purple, :orange]

    for (i, cdf) in enumerate(learning_cdf)
        cdf_values = [cdf(t) for t in t_values]
        label = isnothing(labels) ? L"\beta = %$(β_values[i])" : labels[i]
        color = colors[(i-1) % length(colors) + 1]
        plot!(t_values, cdf_values, label=label, linewidth=1.5, color=color)
    end

    return p
end

#########################################
# HAZARD RATE PLOTS
#########################################

"""
    plot_hazard_rate_decomposition(result::SolvedModel)

Plot the hazard rate decomposition showing h(τ) = π(τ) × h_f(τ)

# Arguments
- `result`: SolvedModel containing all equilibrium results and parameters

# Returns
- Plot showing h(τ), π(τ), and h_f(τ)
# Example
```julia
result = solve_equilibrium(lr, econ, model)
p = plot_hazard_rate_decomposition(result)
```
"""
function plot_hazard_rate_decomposition(result::SolvedModel)
    # Extract fields from SolvedModel
    (; κ, u, p, λ, η) = result.model_params.economic
    ξ = result.ξ
    learning_pdf = result.learning_results.learning_pdf

    ### Set up \bar_\tau grid
    # Compute h̄_f(τ) = hazard rate when bank is definitely fragile (p=1)
    HR_fragile = hazard_rate(1.0, λ, learning_pdf, η)

    # Compute h(τ) = total hazard rate with uncertainty about fragility
    HR_total = hazard_rate(p, λ, learning_pdf, η)

    # Evaluate at all grid points
    τ_grid = range(0, η, length=10000)
    h_bar_f_vals = HR_fragile.(τ_grid)
    h_vals = HR_total.(τ_grid)

    # Derive π(τ) = h(τ) / h̄_f(τ) with safe division
    π_vals = h_vals ./ h_bar_f_vals
    π_vals = [isnan(v) ? 0.0 : clamp(v, 0.0, 1.0) for v in π_vals]

    # Create interpolation functions for plotting
    π_func = LinearInterpolation(τ_grid, π_vals)
    h_bar_f_func = HR_fragile
    h_func = HR_total

    # For each time 't' on the x-axis, calculate τ̄ = ξ - t
    # The hazard functions are evaluated at τ̄.
    # We clamp the values to the valid domain of τ (0 to η).
    t_plot_values = range(0, ξ, length=1000)
    eval_points = clamp.(ξ .- t_plot_values, 0.0,1.3*ξ)

    # Evaluate functions at the transformed points
    h_values = reverse(h_func.(eval_points))
    π_values = reverse(π_func.(eval_points))
    h_bar_f_values = reverse(h_bar_f_func.(eval_points))

    # Find max of h for y-axis scaling
    max_h = maximum(h_values)
    mid_h_bar=h_bar_f_func.((eval_points[1]+eval_points[end])/2)

    # Create plot with LaTeX strings
    p = plot(legend=:topleft, grid=true,
             xlabel=L"\textrm{Time~since~learning~} (\tau)",
             ylabel=L"\textrm{Hazard~Rate}",
             title=L"h(\tau) = \pi(\tau) \times h_f(\tau)",
             ylims=(0,mid_h_bar * 1.2), xlims=(0, 1.2*ξ))

    # Plot components
    plot!(eval_points, h_values,
          label=L"h(\tau) \textrm{~-~Total~hazard}",
          lw=1.5, color=:mediumvioletred)

    plot!(eval_points, π_values,
          label=L"\pi(\tau) \textrm{~-~Belief~fragile}",
          lw=1, color=:royalblue)

    plot!(eval_points, h_bar_f_values,
          label=L"h_f(\tau) \textrm{~-~Conditional~hazard}",
          lw=1, color=:tomato)

    # Add horizontal line for utility 'u'
    hline!([u], color=:darkgray, lw=1, label="", linestyle=:solid)
    annotate!(0.7*ξ, 1.3*u, text(L"u = %$u", 10, :darkgray, :left))

    # Add vertical line for collapse time 'ξ'
    vline!([ξ], color=:darkgoldenrod, lw=1.5, label="", linestyle=:dashdot)
    annotate!(1.08*ξ, mid_h_bar, text(L"\xi=%$(round(ξ, digits=1))", 10, :darkgoldenrod, :center)) # Adjusted position
    return p
end

#########################################
# PLOT EQUILIBRIUM DYNAMICS
#########################################
"""
    plot_equilibrium(result::SolvedModel; x_range=nothing, y_range=nothing)

Plot equilibrium dynamics showing aggregate withdrawals over time.

# Arguments
- `result`: SolvedModel containing all equilibrium results and parameters
- `x_range`: Optional x-axis range
- `y_range`: Optional y-axis range

# Returns
- Plot object showing aggregate withdrawal dynamics

# Example
```julia
result = solve_equilibrium(lr, econ, model)
p = plot_equilibrium(result)
```
"""
function plot_equilibrium(result::SolvedModel; x_range=nothing, y_range=nothing)
    # Extract parameters from SolvedModel
    (; κ, η) = result.model_params.economic
    (; ξ, τ_bar_IN_UNC, τ_bar_OUT_UNC, HR, τ_IN) = result

    # Set grid
    t_grid = collect(0:0.1:min(2*ξ, η))

    # Get the AW values using lazy evaluation
    get_AW_functions!(result)
    AW_cum, AW_OUT, AW_IN = result.aw[].AW_cum, result.aw[].AW_OUT, result.aw[].AW_IN
    ## Start plot
    # Create the plot
    p1 = plot(legend=:topleft, grid=true)
    plot!(p1, t_grid, AW_cum(t_grid), label="AW", color=:darkred, linewidth=2)
    plot!(p1, t_grid,AW_OUT(t_grid), label="Informed", line=:dash, color=:darkred)
    plot!(p1, t_grid,AW_IN(t_grid), label="Reentered", line=:dash, color=:royalblue)
    # Calculate positions for the arrow

    # Add critical lines
    vline!(p1, [ξ], color=:darkgoldenrod, linewidth=2, label=nothing)
    annotate!(p1, [(ξ+0.4, 0.9, text(L"\xi = %$(round(ξ, digits=1))", 7, :left, :darkgoldenrod))])

    if !isnothing(κ)
        hline!(p1, [κ], color=:grey, linewidth=1, label=nothing)
        annotate!(p1, [((ξ)/2, κ+0.015,
                       text(L"\kappa = %$(round(κ, digits=2))", 7, :left, :grey))])
    end

    xlabel!("Time")
    ylabel!("AW(t)")
    title!("Aggregate Withdrawals")

    if !isnothing(y_range)
        ylims!(p1, y_range)
    else
        ylims!(p1, (0, 1))
    end

    if !isnothing(x_range)
        xlims!(p1, x_range)
    end

    arrow_start = (0.8*ξ, AW_OUT(0.8*ξ))
    arrow_end = (arrow_start[1] + τ_IN, arrow_start[2])

    # Add the two-sided arrow
    plot!(p1, [arrow_start[1], arrow_end[1]], [arrow_start[2], arrow_end[2]], linecolor=:darkgreen, linewidth=2, arrow=:both, label=nothing)

    # Annotate the plot
    period = round(τ_IN, digits=2)
    annotate!(p1, mean([arrow_start[1], arrow_end[1]]), mean([arrow_start[2], arrow_end[2]]) - 0.04, text("Return after $period", 6, :center, :darkgreen))

    return p1 # Return the two plot objects as a tuple
end

#########################################
# COMPARATIVE STATICS PLOTS
#########################################

"""
    plot_comp_stat_withdrawals_and_collapse(u_values, max_withdrawals, collapse_times, κ;
                                           return_times=nothing, plot_path="./")

Plot comparative statics for deposit utility showing peak withdrawals and collapse times.

# Arguments
- `u_values`: Array of utility parameter values
- `max_withdrawals`: Array of maximum withdrawal values (may contain NaN)
- `collapse_times`: Array of collapse times (may contain NaN)
- `κ`: Solvency threshold for horizontal line
- `return_times`: Optional array of return times to overlay on collapse time plot
- `plot_path`: Path to save plots (optional)

# Returns
- Combined plot with two panels
"""
function plot_comp_stat_withdrawals_and_collapse(u_values, max_withdrawals, collapse_times, κ;
                                                return_times=nothing)
    gr()
    default(fontfamily="Computer Modern")

    # Find valid (non-NaN) values for collapse times
    valid_collapse = .!isnan.(collapse_times)

    # Panel 1: Peak Withdrawals
    p1 = plot(u_values, max_withdrawals,
              xlabel="Deposit Utility (u)", ylabel="Peak Withdrawals",
              title="(a) Effect on Peak Withdrawals",
              legend=false, color=:darkred,
              ylims=(0, 1))

    # Add threshold line (changed to grey)
    hline!(p1, [κ], color=:grey, linewidth=1, linestyle=:dash, label="")
    # Move kappa annotation slightly left and up
    annotate!(p1, u_values[1] + 0.03, κ + 0.025,
             text("κ = $κ", 8, :left, :grey))

    # Add shaded region for invalid values if any
    invalid_indices = findall(isnan, max_withdrawals)
    if !isempty(invalid_indices)
        if length(invalid_indices) > 1
            u_start = u_values[invalid_indices[1]]
            u_end = u_values[invalid_indices[end]]
            vspan!(p1, [u_start, u_end], color=:gray, alpha=0.2, label="")

            # Add annotation
            mid_u = (u_start + u_end) / 2
            mid_y = (0 + 1) / 2
            annotate!(p1, mid_u, mid_y,
                     text("No Bank Run", 8, :black, rotation=90))
        end
    end

    # Panel 2: Collapse Time
    p2 = plot(u_values[valid_collapse], collapse_times[valid_collapse],
              xlabel="Deposit Utility (u)", ylabel="Time",
              title="(b) Collapse Time and Return Time",
              label="Collapse Time", color=:darkgoldenrod,
              linestyle=:dash, legend=:topright)

    # Add return times if provided
    if !isnothing(return_times)
        valid_return = .!isnan.(return_times)
        plot!(p2, u_values[valid_return], return_times[valid_return],
              label="Return Time")
    end

    # Add shaded region for no-run cases
    invalid_collapse_indices = findall(.!valid_collapse)
    if !isempty(invalid_collapse_indices)
        if length(invalid_collapse_indices) > 1
            u_start = u_values[invalid_collapse_indices[1]]
            u_end = u_values[invalid_collapse_indices[end]]
            vspan!(p2, [u_start, u_end], color=:gray, alpha=0.2, label="")

            # Add annotation
            mid_u = (u_start + u_end) / 2
            y_range = ylims(p2)
            mid_y = (y_range[1] + y_range[2]) / 2
            annotate!(p2, mid_u, mid_y,
                     text("No Bank Run", 8, :black, rotation=90))
        end
    end

    return p1, p2
end

"""
    plot_comp_stat_with_participation(u_values, max_withdrawals, collapse_times, κ, h_un_values; return_times=nothing)

Create comparative statics plots with participation boundary shading.

This function extends plot_comp_stat_withdrawals_and_collapse by adding a shaded region
where agents do not participate (u < h_un(ξ*)). The "no participation" region is shown
in addition to the existing "no bank run" region.

# Arguments
- `u_values`: Vector of utility values
- `max_withdrawals`: Vector of peak withdrawals
- `collapse_times`: Vector of collapse times
- `κ`: Solvency threshold
- `h_un_values`: Vector of uninformed hazard rates h_un(ξ*) for each u
- `return_times`: Optional vector of return times

# Returns
- Tuple of two plots: (peak_withdrawals_plot, collapse_time_plot)
"""
function plot_comp_stat_with_participation(u_values, max_withdrawals, collapse_times, κ, h_un_values;
                                          return_times=nothing)
    gr()
    default(fontfamily="Computer Modern")

    # Find valid (non-NaN) values for collapse times
    valid_collapse = .!isnan.(collapse_times)

    # Identify participation boundary: where u ≤ h_un(ξ*)
    no_participation = u_values .<= h_un_values

    # Panel 1: Peak Withdrawals
    p1 = plot(u_values, max_withdrawals,
              xlabel="Deposit Utility (u)", ylabel="Peak Withdrawals",
              legend=false, color=:darkred,
              ylims=(0, 1))

    # Add threshold line
    hline!(p1, [κ], color=:grey, linewidth=1, linestyle=:dash, label="")
    annotate!(p1, u_values[1] + 0.03, κ + 0.025,
             text("κ = $κ", 8, :left, :grey))

    # Three-region color scheme with tame colors and high transparency
    alpha_val = 0.15

    # Region 1: No participation (u ≤ h_un) - Light red
    no_part_indices = findall(i -> !isnan(h_un_values[i]) && u_values[i] <= h_un_values[i],
                               eachindex(u_values, h_un_values))
    if !isempty(no_part_indices) && length(no_part_indices) > 1
        u_start = u_values[no_part_indices[1]]
        u_end = u_values[no_part_indices[end]]
        vspan!(p1, [u_start, u_end],
               color=RGB(1.0, 0.7, 0.7),  # Light red
               alpha=alpha_val, label="")
    end

    # Region 2: Participation + run (h_un < u and has bankrun) - Light orange
    part_indices = findall(i -> !isnan(h_un_values[i]) && u_values[i] > h_un_values[i] && !isnan(max_withdrawals[i]),
                           eachindex(u_values, h_un_values, max_withdrawals))
    if !isempty(part_indices) && length(part_indices) > 1
        u_start = u_values[part_indices[1]]
        u_end = u_values[part_indices[end]]
        vspan!(p1, [u_start, u_end],
               color=RGB(1.0, 0.9, 0.7),  # Light orange
               alpha=alpha_val, label="")
    end

    # Region 3: No run (u above threshold) - Light green
    no_run_indices = findall(isnan.(max_withdrawals))
    if !isempty(no_run_indices) && length(no_run_indices) > 1
        u_start = u_values[no_run_indices[1]]
        u_end = u_values[no_run_indices[end]]
        vspan!(p1, [u_start, u_end],
               color=RGB(0.8, 1.0, 0.8),  # Light green
               alpha=alpha_val, label="")
    end

    # Add sideways text labels to Panel 1
    # Region 1 label: "No participation"
    if !isempty(no_part_indices) && length(no_part_indices) > 1
        u_mid = (u_values[no_part_indices[1]] + u_values[no_part_indices[end]]) / 2
        annotate!(p1, u_mid, 0.5,
                 text("No participation", 8, :grey, rotation=90))
    end

    # Region 2 label: "Run"
    if !isempty(part_indices) && length(part_indices) > 1
        u_mid = (u_values[part_indices[1]] + u_values[part_indices[end]]) / 2
        annotate!(p1, u_mid, 0.5,
                 text("Run", 8, :grey, rotation=90))
    end

    # Region 3 label: "No Run"
    if !isempty(no_run_indices) && length(no_run_indices) > 1
        u_mid = (u_values[no_run_indices[1]] + u_values[no_run_indices[end]]) / 2
        annotate!(p1, u_mid, 0.5,
                 text("No Run", 8, :grey, rotation=90))
    end

    # Panel 2: Collapse Time
    p2 = plot(u_values[valid_collapse], collapse_times[valid_collapse],
              xlabel="Deposit Utility (u)", ylabel="Time",
              label="Collapse Time", color=:darkgoldenrod,
              linestyle=:dash, legend=:topright)

    # Add return times if provided
    if !isnothing(return_times)
        valid_return = .!isnan.(return_times)
        plot!(p2, u_values[valid_return], return_times[valid_return],
              label="Return Time")
    end

    # Apply same three-region color scheme to Panel 2
    # Region 1: No participation (u ≤ h_un) - Light red
    if !isempty(no_part_indices) && length(no_part_indices) > 1
        u_start = u_values[no_part_indices[1]]
        u_end = u_values[no_part_indices[end]]
        vspan!(p2, [u_start, u_end],
               color=RGB(1.0, 0.7, 0.7),  # Light red
               alpha=alpha_val, label="")
    end

    # Region 2: Participation + run (h_un < u and has bankrun) - Light orange
    if !isempty(part_indices) && length(part_indices) > 1
        u_start = u_values[part_indices[1]]
        u_end = u_values[part_indices[end]]
        vspan!(p2, [u_start, u_end],
               color=RGB(1.0, 0.9, 0.7),  # Light orange
               alpha=alpha_val, label="")
    end

    # Region 3: No run (u above threshold) - Light green
    if !isempty(no_run_indices) && length(no_run_indices) > 1
        u_start = u_values[no_run_indices[1]]
        u_end = u_values[no_run_indices[end]]
        vspan!(p2, [u_start, u_end],
               color=RGB(0.8, 1.0, 0.8),  # Light green
               alpha=alpha_val, label="")
    end

    # Add sideways text labels to Panel 2
    # Calculate middle y-position based on data range
    valid_times = collapse_times[valid_collapse]
    if !isnothing(return_times)
        valid_ret = return_times[.!isnan.(return_times)]
        if !isempty(valid_ret)
            valid_times = vcat(valid_times, valid_ret)
        end
    end
    y_mid = (minimum(valid_times) + maximum(valid_times)) / 2

    # Region 1 label: "No participation"
    if !isempty(no_part_indices) && length(no_part_indices) > 1
        u_mid = (u_values[no_part_indices[1]] + u_values[no_part_indices[end]]) / 2
        annotate!(p2, u_mid, y_mid,
                 text("No participation", 8, :grey, rotation=90))
    end

    # Region 2 label: "Run"
    if !isempty(part_indices) && length(part_indices) > 1
        u_mid = (u_values[part_indices[1]] + u_values[part_indices[end]]) / 2
        annotate!(p2, u_mid, y_mid,
                 text("Run", 8, :grey, rotation=90))
    end

    # Region 3 label: "No Run"
    if !isempty(no_run_indices) && length(no_run_indices) > 1
        u_mid = (u_values[no_run_indices[1]] + u_values[no_run_indices[end]]) / 2
        annotate!(p2, u_mid, y_mid,
                 text("No Run", 8, :grey, rotation=90))
    end

    return p1, p2
end

"""
    plot_heatmap_with_participation_boundary(ave_meeting_time, u_vals, heatmap_data, participation_matrix)

Create a heatmap of peak withdrawals with participation boundary visualization.

This function creates a viridis-colored heatmap showing peak withdrawals across the (β, u) parameter space,
with an overlay indicating the participation constraint boundary. The no-participation region (where u < h_un(ξ*))
is shaded in grey and labeled.

# Arguments
- `ave_meeting_time`: Vector of average meeting times (1/β values) for x-axis
- `u_vals`: Vector of deposit utility values for y-axis
- `heatmap_data`: Matrix of peak withdrawal values (dimensions: length(u_vals) × length(ave_meeting_time))
- `participation_matrix`: Boolean matrix indicating participation (1.0 = participates, 0.0 = no participation)

# Returns
- Plot object with viridis heatmap, boundary line, shading, and text label

# Implementation Details
The function uses a three-pass boundary detection algorithm:
1. **Vertical scan**: For each β column, find where u transitions from no-participation to participation
2. **Horizontal scan**: For each u row, find leftmost β transition (captures nearly-vertical edge)
3. **Top edge scan**: Add points where no-participation extends to max(u)

# Example
```julia
p = plot_heatmap_with_participation_boundary(ave_meeting_time, u_vals, heatmap_data, participation_matrix)
savefig(p, "heatmap_with_boundary.pdf")
```
"""
function plot_heatmap_with_participation_boundary(ave_meeting_time, u_vals, heatmap_data, participation_matrix)
    gr()
    default(fontfamily="Computer Modern")

    # Create base viridis heatmap
    p_heatmap = heatmap(ave_meeting_time, u_vals, heatmap_data,
                        xlabel="Average meeting time",
                        ylabel="Deposit Utility",
                        colorbar=true,
                        color=:viridis,
                        alpha=0.8,
                        size=(700, 400),
                        right_margin=5Plots.mm,
                        xtickfont=font(10), ytickfont=font(10))

    # Identify no-participation region
    no_part_mask = participation_matrix .== 0.0

    # Three-pass boundary detection

    # Pass 1: Vertical scan (for each β, find transition in u)
    boundary_u = Float64[]
    boundary_beta = Float64[]
    for i in 1:length(ave_meeting_time)
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

    # Pass 2: Horizontal scan (for each u, find leftmost transition in β)
    # This captures the nearly-vertical part at the left edge
    vertical_u = Float64[]
    vertical_beta = Float64[]
    for j in 1:length(u_vals)
        for i in 2:length(ave_meeting_time)
            if !isnan(no_part_mask[j, i-1]) && !isnan(no_part_mask[j, i])
                if no_part_mask[j, i-1] && !no_part_mask[j, i]
                  # Only add if it's in the leftmost region
                    if i <= 3
                        push!(vertical_beta, ave_meeting_time[i])
                        push!(vertical_u, u_vals[j])
                        break
                    end
                end
            end
        end
    end

    # Pass 3: Top edge scan (where no-participation extends to top of grid)
    edge_u = Float64[]
    edge_beta = Float64[]
    for i in 1:length(ave_meeting_time)
        if !isnan(no_part_mask[end, i]) && no_part_mask[end, i]
            push!(edge_beta, ave_meeting_time[i])
            push!(edge_u, u_vals[end])
        end
    end

    # Combine all boundary segments
    all_boundary_beta = vcat(boundary_beta, vertical_beta, edge_beta)
    all_boundary_u = vcat(boundary_u, vertical_u, edge_u)

    # Sort by beta for smooth plotting
    if !isempty(all_boundary_beta)
        sort_idx = sortperm(all_boundary_beta)
        all_boundary_beta = all_boundary_beta[sort_idx]
        all_boundary_u = all_boundary_u[sort_idx]
    end

    # Add thin boundary line
    if !isempty(all_boundary_beta)
        plot!(p_heatmap, all_boundary_beta, all_boundary_u,
              color=:black, linewidth=1, linestyle=:solid, label="", alpha=0.7)
    end

    # Add grey shading to no-participation region using filled polygon
    # (Using heatmap! overlay would suppress the colorbar in GR backend)
    if !isempty(all_boundary_beta)
        # Build polygon: boundary curve + corners of the no-participation region
        beta_min = minimum(ave_meeting_time)
        beta_max = maximum(ave_meeting_time)
        u_min = minimum(u_vals)
        # Polygon: bottom-right → bottom-left → up the left edge → along boundary → close
        poly_x = vcat([beta_max], reverse(all_boundary_beta), [beta_min, beta_min, beta_max])
        poly_y = vcat([u_min], reverse(all_boundary_u), [all_boundary_u[1], u_min, u_min])
        plot!(p_heatmap, Shape(poly_x, poly_y),
              fillcolor=:black, fillalpha=0.35, linewidth=0, label="")
    end

    # Add "No participation" text label in white at bottom-left
    label_beta_idx = findfirst(i -> any(no_part_mask[:, i]), 1:length(ave_meeting_time))
    if !isnothing(label_beta_idx)
        label_u_idx = findfirst(j -> no_part_mask[j, label_beta_idx], 1:length(u_vals))
        if !isnothing(label_u_idx)
            # Position in the no-participation region
            label_beta = 0.15
            label_u = 0.02
            annotate!(p_heatmap, label_beta, label_u,
                     text("No participation", 7, :white, :left))
        end
    end

    return p_heatmap
end
