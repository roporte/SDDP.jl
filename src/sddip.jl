#  Copyright 2017-19, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This file implements the SDDiP method of Zou, J., Ahmed, S. & Sun, X.A. Math.
# Program. (2018). https://doi.org/10.1007/s10107-018-1249-5

"""
    SDDP.BinaryState

Variable type to add binary `{0, 1}` state variables to the subproblem.

### Example

    @variable(subproblem, x, SDDP.BinaryState, initial_value = 0)
"""
struct BinaryState end

struct BinaryStateInfo
    in::JuMP.VariableInfo
    out::JuMP.VariableInfo
    initial_value::Float64
end
BinaryStateInfo(x::StateInfo) = BinaryStateInfo(x.in, x.out, x.initial_value)
StateInfo(x::BinaryStateInfo) = StateInfo(x.in, x.out, x.initial_value)

function JuMP.build_variable(
        errorf::Function, info::JuMP.VariableInfo, ::Type{BinaryState};
        initial_value = NaN, kwargs...)
    state_info = JuMP.build_variable(
        errorf, info, BinaryState; initial_value = initial_value, kwargs...)
    return BinaryStateInfo(state_info)
end

function JuMP.add_variable(
        subproblem::JuMP.Model, state_info::BinaryStateInfo, name::String)
    state = JuMP.add_variable(subproblem, StateInfo(state_info), name)
    return state
end

#

function _set_binary_solve_hook(subproblem::JuMP.Model)

end

function _make_dual_infeasible(sense::Symbol, λ::Float64, x::Float64)
    y = sense == :Max ? 1.0 - x : x
    sign = y ≈ 0.0 ? 1.0 : -1.0
    if λ > 1e-6 * sign
        return 1.5 * λ
    elseif λ < -1e-6 * sign
        return 0.0
    else
        return 1.0 * sign
    end
end

#

function set_lagrangian_objective(model::JuMP.Model, d::LinearProgramData, π::Vector{Float64})
    obj_term = length(d.obj.qvars1) == 0 ? d.obj.aff : d.obj
    if JuMP.objective_sense(model) == MOI.MIN_SENSE
        @objective(model, Min, obj_term + dot(π, d.slacks))
    else
        @objective(model, Max, obj_term - dot(π, d.slacks))
    end
end

# For a fixed π, solve minₓ{L = cᵀx + πᵀ(Ax-b)} or maxₓ{L = cᵀx - πᵀ(Ax-b)}
function solve_primal(model::JuMP.Model, d::LinearProgramData, π::Vector{Float64})
    set_lagrangian_objective(m, d, π)
    subgradient = JuMP.objective_sense(model) == MOI.MIN_SENSE ? d.slacks : -d.slacks
    optimize!(model, ignore_solve_hook = true)
    return JuMP.objective_value(model), JuMP.value.(subgradient)
end

function solve_lagrangian(model::JuMP.Model, π::Vector{Float64}; initial_bound, tol, iteration_limit)
    # To make things easier, make primal always min, dual always max.
    reversed = false
    if JuMP.objective_sense(model) == MOI.MAX_SENSE
        lp.obj *= -1.0
        initial_bound *= -1.0
        JuMP.set_objective_sense(model, MOI.MIN_SENSE)
        reversed = true
    end

    # We will sample the Lagrangian at pi = 0
    bound, direction = solve_primal(model, lp, π)

    # This setup makes it possible to cycle when 0 is one of many subgradients.
    # Need to test not if abs(direction[i]) < 1e-6, but if meeting relaxed i^th
    # constraint doesn't change objective.
    iter = 0
    while isapprox(initial_bound, bound, atol=tol) && iter < iteration_limit
        iter += 1
        @assert bound <= initial_bound + 1e-6  # Santity check
        for i in 1:length(π)
            # If dual=0 is feasible, let it be TODO: improve on this.
            if abs(direction[i]) < 1e-6
                continue
            end
            π[i] += (initial_bound - bound) / direction[i]
        end
        bound, direction = solve_primal(model, lp, π)
    end

    if reversed
        π .*= -1
        lp.obj *= -1
        bound *= -1
        JuMP.set_objective_sense(model, MOI.MAX_SENSE)
    end
    return bound
end

function SDDiPsolve!(sp::JuMP.Model; require_duals::Bool=false, iteration::Int=-1, kwargs...)
    @assert !(require_duals && iteration == -1)
    solvers = sp.ext[:solvers]
    if require_duals && SDDP.ext(sp).stage > 1
        # Update the objective we cache in case the objective has noises
        l = lagrangian(sp)
        l.obj = getobjective(sp)
        cuttype = getcuttype(iteration, sp.ext[:pattern])

            # Update initial bound of the dual problem
            @assert solve(sp) == :Optimal
            l.method.initialbound = getobjectivevalue(sp)
            # Slacks have a new RHS each iteration, so update them
            l.slacks = getslack.(l.constraints)
            # Somehow choose duals to start with
            π0 = zeros(length(l.constraints)) # or rand, or ones
            initialize_dual!(sp, π0)
            # Lagrangian objective and duals
            setsolver(sp, solvers.MIP)
            status, _, kelleymodel = lagrangiansolve!(l, sp, π0)
            strengthendual!(l, sp, π0, kelleymodel)
        end
        sp.obj = l.obj
    else
        # We are in the forward pass, or we are in stage 1
        setsolver(sp, solvers.MIP)
        status = JuMP.solve(sp, ignore_solve_hook=true)
    end
    status
end
