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
