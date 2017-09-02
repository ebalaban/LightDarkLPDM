
# using Plots
import POMDPs: action

using Combinatorics
# using Plots
using POMDPToolbox, Parameters, ParticleFilters, StaticArrays
using LightDarkPOMDPs
using LPDM
import LPDM: init_bounds


# Typealias appropriately
const LDState  = SVector{2,Float64}
const LDAction = SVector{2,Float64}
const LDObs    = SVector{2,Float64}
const LDBelief = LPDMBelief
include("LPDMBounds.jl")


function execute(n_sims::Int64 = 100)

    p = LightDark2DTarget()
    config = LPDMConfig();
    config.n_particles = 100;
    config.sim_len = 100;
    p.n_rand = 2;
    config.search_depth = 10;

    state::LDState              = LDState(π, e);
    rewards::Array{Float64}     = Array(Float64, 0)

    bu = LPDMBeliefUpdater{LDState, LDAction, LDObs}(p, n_particles = config.n_particles);  # initialize belief updater
    initial_states = state_distribution(p, state, config)                                   # create initial  distribution
    current_belief = LPDM.create_belief(bu)                                                 # allocate an updated belief object

    LPDM.initialize_belief(bu, initial_states, current_belief)                              # initialize belief
    updated_belief = LPDM.create_belief(bu)                                                 # update belief now that it has been initialized
    custom_bounds = LDBounds()                                                              # bounds object

    solver = LPDMSolver{LDState,
                        LDAction,
                        LDObs,
                        LDBounds}(bounds = custom_bounds)
    init_solver(solver, p)

    policy::LPDMPolicy = POMDPs.solve(solver, p)
    seed  ::UInt32   = convert(UInt32, 42)                          # the main random seed that's used to set the other seeds

    rng = LPDMRandomVector([])                                      # random number vector
    rs = FiniteRandomStreamsCommonSeed(1, 100, 1, seed)             # seed(?) for the random vector 

    sim_steps::Int64 = 1
    r::Float64 = 0.0

    println(updated_belief)

    tic() # start the clock
    while !isterminal(p, state) && (solver.config.sim_len == -1 || sim_steps < solver.config.sim_len)
        action = POMDPs.action(policy, current_belief)
        println(action)
        rng.vector = rs.streams[1,sim_steps,:]
        state, obs, r = generate_sor(p, state, action, rng)
        push!(rewards, r)
        println(rewards)
        # update belief
        POMDPs.update(bu, current_belief, action, obs, updated_belief)
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





############
# place all of these where they belong once you figure out where that is.
###########

function LPDM.init_bounds(::LDBounds, ::AbstractLD2, ::LPDM.LPDMConfig)
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

# LPDM specific random function:
Base.rand(rng::LPDM.LPDMRandomVector, d::SymmetricNormal2) = d.mean + d.std*Vec2(Base.rand(rng.vector, 2))



# #  check if UB is ever lower than LB
# function foo(j::Int64 = 10000)                      
#     iswrong = Array{LDState ,1}(0)
#     for i in 1:j
#         st = LDState(rand(2)*10)
#         s = POMDPToolbox.Particle{Vec2}(st,1)
#         ub, lb  = upperBound(p,s), lowerBound(p,s)
#         if lb>ub
#             push!(iswrong, st)
#         end
#     end
#     return iswrong
# end
