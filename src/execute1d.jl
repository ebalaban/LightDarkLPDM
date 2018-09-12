module execute1d
using LPDM
using POMDPs, POMDPToolbox, Parameters, ParticleFilters, StaticArrays, Plots, D3Trees, Distributions

include("LightDarkPOMDPs.jl")
using LightDarkPOMDPs

# using Plots
import POMDPs: action, generate_o

using Combinatorics
# using Plots

using StatsFuns
import LPDM: init_bounds!

# Typealias appropriately
const LDState  = Float64
const LDAction = Float64
const LDObs    = Float64
const LDBelief = LPDMBelief
include("LPDMBounds1d.jl")

############
# place all of these where they belong once you figure out where that is.
###########

# POMDPs.generate_o(p::AbstractLD1, sp::Float64, rng::LPDM.RNGVector) = LightDarkPOMDPs.generate_o(p, sp, rng)


function LPDM.init_bounds!(::LDBounds1d, ::LightDarkPOMDPs.AbstractLD1, ::LPDM.LPDMConfig)
end

# Just use the initial distribution in the POMDP
function state_distribution(pomdp::LightDarkPOMDPs.AbstractLD1, config::LPDMConfig, rng::RNGVector)
    states = Vector{POMDPToolbox.Particle{Float64}}();
    weight = 1/(config.n_particles^2) # weight of each individual particle
    particle = POMDPToolbox.Particle{Float64}(0.0, weight)

    for i = 1:config.n_particles^2 #TODO: Inefficient, possibly improve. Maybe too many particles
        particle = POMDPToolbox.Particle{Float64}(rand(rng, pomdp.init_dist), weight)
        push!(states, particle)
    end
    println("n states: $(length(states))")
    return states
end

Base.rand(p::LightDarkPOMDPs.AbstractLD1, s::Float64, rng::LPDM.RNGVector) = rand(rng, Normal(s, std(p.init_dist)))

function Base.rand(rng::LPDM.RNGVector,
                   d::Normal)

    # a random number selected from normal distribution
    r1 = norminvcdf(mean(d), std(d), rand(rng))
    return r1
end



function execute()#n_sims::Int64 = 100)

    p = LightDarkPOMDPs.LightDark1DDespot()
    world_seed  ::UInt32   = convert(UInt32, 42)
    world_rng = RNGVector(1, world_seed)
    LPDM.set!(world_rng, 1)
    s::LDState                  = LDState(π);    # initial state
    rewards::Array{Float64}     = Array{Float64}(0)
    custom_bounds = LDBounds1d{LDAction}()    # bounds object
    solver = LPDMSolver{LDState, LDAction, LDObs, LDBounds1d, RNGVector}(bounds = custom_bounds,
                                                                        # rng = sim_rng,
                                                                        debug = 1,
                                                                        time_per_move = 1.0,  #sec
                                                                        sim_len = 100,
                                                                        search_depth = 50,
                                                                        n_particles = 20,
                                                                        seed = UInt32(91),
                                                                        # max_trials = 10)
                                                                        max_trials = 100)
#---------------------------------------------------------------------------------
    # Belief
    bu = LPDMBeliefUpdater(p, n_particles = solver.config.n_particles);  # initialize belief updater
    initial_states = state_distribution(p, solver.config, world_rng)     # create initial  distribution
    current_belief = LPDM.create_belief(bu)                       # allocate an updated belief object
    LPDM.initialize_belief(bu, initial_states, current_belief)    # initialize belief
    # show(current_belief)
    updated_belief = LPDM.create_belief(bu)
#---------------------------------------------------------------------------------

    # solver = LPDMSolver{LDState, LDAction, LDObs, LDBounds1d, RNGVector}( bounds = custom_bounds,
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
    # println("actions: $(POMDPs.actions(p, true))")

    tic() # start the clock
    # println("sim_len: $(solver.config.sim_len)")
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

    t = LPDM.d3tree(solver)
    # # show(t)
    inchrome(t)
    # blink(t)

    return sim_steps, sum(rewards), discounted_reward, run_time
    # return t
end
end
