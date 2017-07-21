importall POMDPs
using StaticArrays
using POMDPToolbox
using Parameters # for @with_kw
using ParticleFilters # for AbstractParticleBelief
using LightDarkPOMDPs
using LPDM
# Typealias appropriately
typealias LDState   SVector{2,Float64}
typealias LDAction  SVector{2,Float64}
typealias LDObs     SVector{2,Float64}
typealias LDBelief  LPDMBelief

include("/Users/tomer/.julia/v0.5/LightDarkPOMDPs/src/LPDMBounds.jl")


p = LightDark2DTarget()
config = LPDMConfig();
config.n_particles = 100;
config.sim_len = -1;
bu = LPDMBeliefUpdater{LDState, LDAction, LDObs}(p, n_particles = config.n_particles);
# create initial belief distribution and allocate an updated belief object
initial_states = state_distribution(p, LDState(π, e), config)
current_belief = LPDM.create_belief(bu)
println(current_belief)
# initialize belief
LPDM.initialize_belief(bu, initial_states, current_belief)
updated_belief = LPDM.create_belief(bu)
custom_bounds = LDBounds()

#LPDM.bounds(LDBounds, p, current_belief.particles, config)
#
solver = LPDMSolver{LDState,
LDAction,
LDObs,
LDBounds}(bounds = custom_bounds)
