#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

"""
    DefaultCutOracle()

# Description

Initialize the default cut oracle.

This oracle keeps every cut discovered and does not perform cut selection.
"""
struct DefaultCutOracle <: AbstractCutOracle
    cuts::Vector{Cut}
end
DefaultCutOracle() = DefaultCutOracle(Cut[])

storecut!(oracle::DefaultCutOracle, m::SDDPModel, sp::JuMP.Model, cut::Cut) = push!(oracle.cuts, cut)
validcuts(oracle::DefaultCutOracle) = oracle.cuts
allcuts(oracle::DefaultCutOracle) = oracle.cuts
