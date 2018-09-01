using LPDM
using POMDPToolbox, Parameters, ParticleFilters, StaticArrays, Plots, D3Trees

include("LightDarkPOMDPs.jl")
using LightDarkPOMDPs

# using Plots
import POMDPs: action, generate_o

using Combinatorics
# using Plots

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

# Just use the initial distribution in the POMDP
function state_distribution(pomdp::AbstractLD2, config::LPDMConfig, rng::RNGVector)
    states = Vector{POMDPToolbox.Particle{Vec2}}();
    weight = 1/(config.n_particles^2) # weight of each individual particle
    particle = POMDPToolbox.Particle{Vec2}([0,0], weight)

    for i = 1:config.n_particles^2 #TODO: Inefficient, possibly improve. Maybe too many particles
        particle = POMDPToolbox.Particle{Vec2}(rand(rng, pomdp.init_dist), weight)
        push!(states, particle)
    end
    println("n states: $(length(states))")
    return states
end

# function state_distribution(p::AbstractLD2, s::Vec2, config::LPDMConfig)
#     randx = randn(config.n_particles);
#     randy = randn(config.n_particles);
#     states = Vector{POMDPToolbox.Particle{Vec2}}();
#     weight = 1/(config.n_particles^2) # weight of each individual particle
#     particle = POMDPToolbox.Particle{Vec2}([0,0], weight)
#
#     for rx in randx, ry in randy
#         particle = POMDPToolbox.Particle{Vec2}([s[1]+rx, s[2]+ry], weight)
#         push!(states, particle)
#     end
#     println("n states: $(length(states))")
#     println(states)
#     error("done")
#     return states
# end

Base.rand(p::AbstractLD2, s::Vec2, rng::LPDM.RNGVector) = rand(rng, SymmetricNormal2(s, p.init_dist.std))

function Base.rand(rng::LPDM.RNGVector,
                   d::SymmetricNormal2)

    # 2 random numbers selected from normal distribution
    r1 = norminvcdf(d.mean[1], d.std, rand(rng))
    r2 = norminvcdf(d.mean[2], d.std, rand(rng))

    return Vec2(r1, r2)
end



function execute()#n_sims::Int64 = 100)

    p = LightDark2DDespot()
    # gr()
    # plot(p)
    # Base.invokelatest(gui()) #HACK: this is to get rid of the "The applicable method may be too new:
    #                     #running in world age 22059, while current world is 22060"

    # config.n_particles = 100;
    # n_particles = 10;
    # sim_seed = UInt32(1)
    # sim_rng  = RNGVector(n_particles, sim_seed)

    world_seed  ::UInt32   = convert(UInt32, 42)
    world_rng = RNGVector(1, world_seed)
    LPDM.set!(world_rng, 1)

    s::LDState                  = LDState(π, e);    # initial state
    rewards::Array{Float64}     = Array{Float64}(0)

    custom_bounds = LDBounds{LDAction}()    # bounds object

    # TODO: consider if using floats as observations is better
    solver = LPDMSolver{LDState, LDAction, LDObs, LDBounds, RNGVector}( bounds = custom_bounds,
                                                                        # rng = sim_rng,
                                                                        debug = 2,
                                                                        time_per_move = 10.0,  #sec
                                                                        sim_len = 1,
                                                                        search_depth = 10,
                                                                        n_particles = 10,
                                                                        seed = UInt32(1),
                                                                        # max_trials = 10)
                                                                        max_trials = 2)

#---------------------------------------------------------------------------------
    # Belief
    bu = LPDMBeliefUpdater(p, n_particles = solver.config.n_particles);  # initialize belief updater
    initial_states = state_distribution(p, solver.config, world_rng)     # create initial  distribution
    current_belief = LPDM.create_belief(bu)                       # allocate an updated belief object

    LPDM.initialize_belief(bu, initial_states, current_belief)    # initialize belief
    # show(current_belief)
    updated_belief = LPDM.create_belief(bu)
#---------------------------------------------------------------------------------

    # solver = LPDMSolver{LDState, LDAction, LDObs, LDBounds, RNGVector}( bounds = custom_bounds,
    #                                                                     rng = sim_rng,
    #                                                                     debug = 2,
    #                                                                     time_per_move = 10.0,  #sec
    #                                                                     max_trials = 10)

    # Not needed for now
    # init_solver!(solver, p)

    policy::LPDMPolicy = POMDPs.solve(solver, p)

    sim_steps::Int64 = 1
    r::Float64 = 0.0

    # println("updated belief: $(current_belief)")
    println("actions: $(POMDPs.actions(p, true))")

    tic() # start the clock
    println("sim_len: $(solver.config.sim_len)")
    while !isterminal(p, s) && (solver.config.sim_len == -1 || sim_steps <= solver.config.sim_len)

        println("")
        println("=============== Step $sim_steps ================")
        println("s: $s")
        a = POMDPs.action(policy, current_belief)
        println("a: $a")

        s, o, r = POMDPs.generate_sor(p, s, a, world_rng)
        println("s': $s")
        println("o': $o")
        println("r: $r")
        push!(rewards, r)
        println("=======================================")

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

    println("root q nodes are: $(typeof(solver.root.q_nodes)) of length $(length(solver.root.q_nodes)), start is $(start(solver.root.q_nodes))")
    t = LPDM.d3tree(solver)
    inchrome(t)

    return sim_steps, sum(rewards), discounted_reward, run_time
end
