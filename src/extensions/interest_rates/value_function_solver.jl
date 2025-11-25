#=
value_function_solver.jl

Value function solver for the interest rates extension.
Solves HJB equations when r > 0 to determine optimal reentry decisions.

Author: Robin Lenoir
=#

# Include required dependencies
using DifferentialEquations, Interpolations

# Include model definitions
include(joinpath(@__DIR__, "interest_rate_model.jl"))

"""
    solve_value_function(hr, δ, r, u; stop_at=100, tol=1e-12)

Solve Hamilton-Jacobi-Bellman equation for value function with positive interest rates.

# Mathematical Framework (see Appendix C, Section "Interest Rates and Value Functions")
The value function V(τ̄) satisfies the HJB equation in reversed time (Appendix C, eq. after line 188):
dV/dτ̄ = (h(τ̄) + δ)(1 - V(τ̄)) + max(u + rV(τ̄) - h(τ̄), 0)

This implements the modified HJB from Section 5.2 of the main text, expressed in reversed time
coordinates τ̄ = ξ* - τ where ξ* is the equilibrium crash time and τ is time since learning.

Variables:
- τ̄ is the "buffer time" before potential collapse (reversed time)
- h(τ̄) is the hazard rate function
- δ is the deposit maturity rate (δ > r to ensure finite value)
- r is the interest rate on deposits
- u is the utility flow from deposits
- V(τ̄) represents continuation value of 1 dollar of deposits

# Boundary Condition (Appendix C, line 192)
V(0) = δ/(δ-r) (terminal value at crash time, τ̄ = 0 corresponds to τ = ξ*)

# Arguments
- `hr`: Hazard rate function (LinearInterpolation object)
- `δ::Float64`: Recovery/discount rate (δ > 0)
- `r::Float64`: Interest rate (0 ≤ r < δ)
- `u::Float64`: Utility flow from deposits

# Keyword Arguments
- `stop_at::Float64 = 100`: Maximum integration time
- `tol::Float64 = 1e-12`: ODE solver tolerance

# Returns
- `V_grid::Vector{Float64}`: Time grid for value function
- `V_values::Vector{Float64}`: Value function values V(τ̄)

# Technical Implementation
- **Time Direction**: Integrates forward in τ̄ (reversed time from collapse)
- **ODE Solver**: AutoTsit5(Rosenbrock23()) with high precision
- **Boundary Treatment**: Starts from V(0) = δ/(δ-r)
- **Reentry Threshold**: Optimal reentry when rV(τ̄) > h(τ̄)

# Example
```julia
# Solve value function given hazard rate
V_grid, V_values = solve_value_function(HR, 0.1, 0.02, 0.05)
V_func = LinearInterpolation(V_grid, V_values)
```
"""
function solve_value_function(hr, δ, r, u; tol=eps())
    # Validation
    r < δ || throw(ArgumentError("Interest rate r must be less than recovery rate δ, got r=$r, δ=$δ"))
    δ > 0 || throw(ArgumentError("Recovery rate δ must be positive, got δ=$δ"))
    r ≥ 0 || throw(ArgumentError("Interest rate r must be non-negative, got r=$r"))

    # Extract grid from hazard rate interpolation object
    hr_grid = hr.itp.knots[1]

    # Terminal condition (boundary value at τ̄ = 0)
    V_terminal = δ / (δ - r)

    # Time span for integration (forward in reversed time τ̄)
    tspan = (0.0, hr_grid[end])

    # Define the HJB equation (Appendix C, Section "Interest Rates and Value Functions")
    # This implements the modified HJB with deposit maturity:
    # V'(τ̄) = (h(τ̄) + δ)(1 - V(τ̄)) + max{u + rV(τ̄) - h(τ̄), 0}
    #
    # The max term captures the reentry option: agents optimally hold deposits when
    # the adjusted threshold u + rV(τ̄) exceeds the hazard rate h(τ̄).
    # See Appendix C line 196-206 for the modified optimal buffer computation.
    function hjb_equation!(dV, V, p, τ_bar)
        h_τ = hr(τ_bar)  # Evaluate hazard rate at current reversed time

        # Reentry option value: max{u + rV - h, 0}
        # Positive when u + rV(τ̄) > h(τ̄), zero otherwise
        reentry_value = max(u + r * V[1] - h_τ, 0.0)

        # Complete HJB equation: V'(τ̄) = (h + δ)(1 - V) + reentry_value
        dV[1] = (h_τ + δ) * (1 - V[1]) + reentry_value
    end

    # Set up ODE problem with initial condition
    # Since we are reversed time the intial condition is the terminal condition V(\xi)
    V0 = [V_terminal]
    prob = ODEProblem(hjb_equation!, V0, tspan)

    # Solve HJB equation with high precision, saving at hr_grid points
    sol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=tol, abstol=tol, saveat=hr_grid, verbose=false)

    # Extract scalar values from solution (sol.u contains vectors)
    V_values = [sol.u[i][1] for i in 1:length(sol.u)]
    V = LinearInterpolation(sol.t, V_values)

    return V
end

