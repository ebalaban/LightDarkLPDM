include("LightDarkPOMDPs.jl")

# using Plots
import POMDPs: action, generate_o

using Combinatorics
# using Plots
using POMDPToolbox, Parameters, ParticleFilters, StaticArrays
using LPDM
using LightDarkPOMDPs
using StatsFuns
import LPDM: init_bounds!


# Typealias appropriately
const LDState  = SVector{2,Float64}
const LDAction = SVector{2,Float64}
const LDObs    = SVector{2,Float64}
const LDBelief = LPDMBelief
include("LPDMBounds.jl")

############
# place all of these where they belong once you figure out where that is.
###########

# POMDPs.generate_o(p::AbstractLD2, sp::Vec2, rng::LPDM.RNGVector) = LightDarkPOMDPs.generate_o(p, sp, rng)


function LPDM.init_bounds!(::LDBounds, ::AbstractLD2, ::LPDM.LPDMConfig)
end


function state_distribution(p::AbstractLD2, s::Vec2, config::LPDMConfig)
    randx = randn(config.n_particles);
    randy = randn(config.n_particles);
    states = Vector{POMDPToolbox.Particle{Vec2}}();
    particle = POMDPToolbox.Particle{Vec2}([0,0],1)

    for rx in randx, ry in randy
        particle = POMDPToolbox.Particle{Vec2}([s[1]+rx, s[2]+ry],1)
        push!(states, particle)
    end
    return states
end

Base.rand(p::AbstractLD2, s::Vec2, rng::LPDM.RNGVector) = rand(rng, SymmetricNormal(s, p.init_dist.std))

function Base.rand(rng::LPDM.RNGVector,
                   d::SymmetricNormal2)

    # 2 random numbers selected from normal distribution
    r1 = norminvcdf(d.mean[1], d.std, rand(rng))
    r2 = norminvcdf(d.mean[2], d.std, rand(rng))

    return Vec2(r1, r2)
end



function execute()#n_sims::Int64 = 100)

    p = LightDark2DTarget()
    config = LPDMConfig();
    config.n_particles = 100;
    config.sim_len = 100;
    config.search_depth = 10;

    s::LDState                  = LDState(π, e);
    rewards::Array{Float64}     = Array{Float64}(0)
#---------------------------------------------------------------------------------
    # Belief
    bu = LPDMBeliefUpdater(p, n_particles = config.n_particles);  # initialize belief updater
    initial_states = state_distribution(p, s, config)                                   # create initial  distribution
    current_belief = LPDM.create_belief(bu)                                                 # allocate an updated belief object

    LPDM.initialize_belief(bu, initial_states, current_belief)                              # initialize belief
    updated_belief = LPDM.create_belief(bu)                                                 # update belief now that it has been initialized
#---------------------------------------------------------------------------------


    sim_seed = UInt32(91)
    sim_rng  = RNGVector(config.n_particles, sim_seed)
    custom_bounds = LDBounds()    # bounds object

    solver = LPDMSolver{LDState, LDAction, LDObs, LDBounds, RNGVector}( bounds = custom_bounds,
                                                                        rng = sim_rng)


    init_solver!(solver, p)

    policy::LPDMPolicy = POMDPs.solve(solver, p)

    seed  ::UInt32   = convert(UInt32, 42)
    world_rng = RNGVector(1, seed)
    LPDM.set!(world_rng, 1)

    sim_steps::Int64 = 1
    r::Float64 = 0.0

    println("updated belief: $(current_belief)")

    tic() # start the clock
    while !isterminal(p, s) && (solver.config.sim_len == -1 || sim_steps < solver.config.sim_len)
        a = POMDPs.action(policy, current_belief)
        println("action: $a")

        s, o, r = POMDPs.generate_sor(p, s, a, world_rng)
        push!(rewards, r)
        println("rewards: $rewards")
        # update belief
        POMDPs.update(bu, current_belief, a, o, updated_belief)
        current_belief = deepcopy(updated_belief)
        sim_steps += 1
    end
    run_time::Float64 = toq() # stop the clock

    # Compute discounted reward
    discounted_reward::Float64 = 0.0
    multiplier::Float64 = 1.0
    for r in rewards
        discounted_reward += multiplier * r
        multiplier *= p.discount
    end

    return sim_steps, sum(rewards), discounted_reward, run_time
end
