module execute
using LPDM
using POMDPs, Parameters, StaticArrays, D3Trees, Distributions, SparseArrays
using Combinatorics
using StatsFuns
using Printf
using Dates

include("LightDarkPOMDPs.jl")
using LightDarkPOMDPs

include("LPDMBounds1d.jl")
include("LPDMBounds2d.jl")

struct LPDMTest
    # solver_mode::Symbol
    action_mode::Symbol
    obs_mode::Symbol
    reward_mode::Symbol
end

struct LPDMScenario{S}
    s0::S
end

# reward function options - :quadratic or :fixed
function batch_execute(;dims::Int64=1, n::Int64=1, debug::Int64=1, reward_mode=:quadratic)

    if dims == 1
        S = LD1State
        A = LD1Action
        O = LD1Obs
        B = LDBounds1d{S,A,O}
    elseif dims == 2
        S = LD2State
        A = LD2Action
        O = LD2Obs
        B = LDBounds2d{S,A,O}
    else
        error("Invalid number of dimensions $dims")
    end

    test=Array{LPDMTest}(undef,0)

    # DISCRETE OBSERVATIONS
    push!(test, LPDMTest(:standard, :discrete, reward_mode))
    push!(test, LPDMTest(:extended, :discrete, reward_mode))
    push!(test, LPDMTest(:blind_vl, :discrete, reward_mode))
    push!(test, LPDMTest(:adaptive, :discrete, reward_mode))

    # CONTINUOUS OBSERVATIONS
    # push!(test, LPDMTest(:standard, :continuous, reward_mode))
    # push!(test, LPDMTest(:extended, :continuous, reward_mode))
    # push!(test, LPDMTest(:blind_vl,   :continuous, reward_mode))
    # push!(test, LPDMTest(:adaptive, :continuous, reward_mode))

    scen=Array{LPDMScenario{S}}(undef,0)
    if dims == 1
        push!(scen, LPDMScenario(LD1State(-2*π)))
        # push!(scen, LPDMScenario(LD1State(π/2)))
        # push!(scen, LPDMScenario(LD1State(3/2*π)))
        # push!(scen, LPDMScenario(LD1State(2*π)))
    elseif dims == 2
        # push!(scen, LPDMScenario(LD2State(-2*π, π)))
        push!(scen, LPDMScenario(LD2State(-π, π)))
        # push!(scen, LPDMScenario(LD2State(π/2, π/2)))
        # push!(scen, LPDMScenario(LD2State(π, -π)))
        # push!(scen, LPDMScenario(LD2State(2*π,2*π)))
    end

    # Dummy execution, just to make sure all the code is compiled and loaded,
    # to improve uniformity of subsequent executions.
    # execute(solv_mode = :despot, action_space_type = :small, n_sims = 1, s0 = LD2State(π,π), output = 0)
    # execute(solv_mode = :lpdm, action_space_type = :adaptive, n_sims = 1, s0 = LD2State(π,π), output = 0)

    # General solver parameters
    steps::Int64                = -1
    time_per_move::Float64      = -1.0
    search_depth::Int64         = 30
    n_particles::Int64          = 50
    max_trials::Int64           = 1000

    f = open("results_" * Dates.format(now(),"yyyy-mm-dd_HH_MM") * ".txt", "w")

    Printf.@printf(f,"GENERAL SOLVER PARAMETERS\n")
    Printf.@printf(f,"\tsteps:\t\t\t%d\n", steps)
    Printf.@printf(f,"\ttime per move:\t\t\t%f\n", time_per_move)
    Printf.@printf(f,"\tsearch depth:\t\t\t%d\n", search_depth)
    Printf.@printf(f,"\tN particles:\t\t\t%d\n", n_particles)
    Printf.@printf(f,"\tmax trials:\t\t\t%d\n\n", max_trials)

    for i in 1:length(scen)
        if debug >= 0
            println("")
            println("SCENARIO $i, s0 = $(scen[i].s0)")
            println("------------------------")
        end

        Printf.@printf(f,"SCENARIO %d, s0 = %s\n", i, "$(scen[i].s0)")
        Printf.@printf(f,"================================================================\n")
        Printf.@printf(f,"ACT. MODE\t\tOBS. MODE\t\tSTEPS (STD)\t\t\tREWARD (STD)\n")
        Printf.@printf(f,"================================================================\n")
        for t in test
            if debug >= 0
                println("dimensions: $dims, actions: $(t.action_mode), observations: $(t.obs_mode), rewards: $(t.reward_mode)")
            end
            steps_avg, steps_std, reward_avg, reward_std =
                        run_scenario(scen[i].s0, A, O, B,
                                    dims              = dims,
                                    # solver_mode       = t.solver_mode,
                                    action_mode       = t.action_mode,
                                    obs_mode          = t.obs_mode,
                                    n_sims            = n,
                                    steps             = steps,
                                    time_per_move     = time_per_move,
                                    search_depth      = search_depth,
                                    n_particles       = n_particles,
                                    max_trials        = max_trials,
                                    output            = debug)

            Printf.@printf(f,"%s\t\t%s\t\t\t%05.2f (%06.2f)\t\t%06.2f (%06.2f)\n",
                                            string(t.action_mode), string(t.obs_mode), steps_avg, steps_std, reward_avg, reward_std)

            debug >=0 && println("STATS: actions=$(t.action_mode), observations=$(t.obs_mode), steps = $(steps_avg) ($steps_std), reward = $(reward_avg) ($reward_std)\n")
        end
        Printf.@printf(f,"==================================================================\n")
        Printf.@printf(f,"%d tests per scenario\n\n", n)
    end
    close(f)
end

function run_scenario(s0::S,
                A::Type,
                O::Type,
                B::Type;
                dims::Int64                 = 1,
                vis::Vector{Int64}          = Int64[],
                # solver_mode::Symbol         = :despot,
                action_mode::Symbol         = :standard,
                obs_mode::Symbol            = :discrete,
                reward_mode                 = :quadratic,
                n_sims::Int64               = 1,
                steps::Int64                = -1,
                time_per_move::Float64      = 5.0,
                search_depth::Int64         = 50,
                n_particles::Int64          = 50,
                max_trials::Int64           = -1,
                output::Int64               = 1
                ) where {S}

    if dims == 1
        p = LightDark1DLpdm(action_mode = action_mode,
                            obs_mode = obs_mode,
                            reward_mode = reward_mode)
    elseif dims == 2
        p = LightDark2DLpdm(action_mode = action_mode,
                            obs_mode = obs_mode,
                            reward_mode = reward_mode)
    end

    sim_rewards = Vector{Float64}(undef,n_sims)
    sim_steps   = Vector{Int64}(undef,n_sims)

    for sim in 1:n_sims

        world_rng = RNGVector(1, UInt32(sim))
        LPDM.set!(world_rng, 1)
        step_rewards::Array{Float64}     = Vector{Float64}(undef,0)

        solver = LPDM.LPDMSolver{S, A, O, B, RNGVector}(
                                                        # rng = sim_rng,
                                                        debug = output,
                                                        time_per_move = time_per_move,  #sec
                                                        # time_per_move = 1.0,  #sec
                                                        sim_len = steps,
                                                        #sim_len = 20,
                                                        search_depth = search_depth,
                                                        n_particles = n_particles,
                                                        # seed = UInt32(2),
                                                        seed = UInt32(2*sim+1),
                                                        # max_trials = 1000)
                                                        max_trials = max_trials,
                                                        # solver_mode = solver_mode,
                                                        action_mode = action_mode,
                                                        obs_mode    = obs_mode)

    #---------------------------------------------------------------------------------
        # Belief
        bu = LPDMBeliefUpdater(p,
                               n_particles = solver.config.n_particles,
                               seed = UInt32(3*sim+1));  # initialize belief updater
        initial_states = state_distribution(p, s0, solver.config, world_rng)     # create initial  distribution
        current_belief = LPDM.create_belief(bu)                       # allocate an updated belief object
        LPDM.initialize_belief(bu, initial_states, current_belief)    # initialize belief
        updated_belief = LPDM.create_belief(bu)

    #---------------------------------------------------------------------------------

        policy::LPDMPolicy = POMDPs.solve(solver, p)
        s = s0
        step::Int64 = 0
        r::Float64 = 0.0

        if output >= 0
            println("")
            println("*** SIM $sim (dimensions: $dims, action mode: $action_mode, obs mode: $obs_mode, s0: $s0)***")
            println("")
        end

        val, run_time, bytes, gctime, memallocs =
        @timed while !isterminal(p, s) && (solver.config.sim_len == -1 || step < solver.config.sim_len)
            step += 1

            if output >= 1
                println("")
                println("=============== Step $step ================")
                show(current_belief)
            end

            a = POMDPs.action(policy, current_belief)
            if output >= 1
                println("s: $s")
                println("a: $a")
            end

            s, o, r = POMDPs.generate_sor(p, s, a, world_rng)
            push!(step_rewards, r)
            if output >= 1
                println("s': $s")
                println("o': $o")
                println("r: $r")
                println("=======================================")
            end

            # update belief
            POMDPs.update(bu, current_belief, a, o, updated_belief)
            current_belief = deepcopy(updated_belief)
            # show(updated_belief) #NOTE: don't show for now

            if LPDM.isterminal(p, current_belief.particles)
                if output >= 1
                    println("Terminal belief. Execution completed.")
                    show(current_belief)
                end
                break
            end

            if step ∈ vis
                t = LPDM.d3tree(solver,
                                detect_repeat=false,
                                title="Step $step",
                                init_expand=10)
                # # show(t)
                inchrome(t)
                # blink(t)
            end
            if output >= 1
                println("root actions: $(solver.root.action_space)")
            end
        end

        sim_steps[sim] = step
        sim_rewards[sim] = sum(step_rewards)
        output >= 0 && println("steps=$(sim_steps[sim]), reward=$(sim_rewards[sim])")
    end

    return mean(sim_steps), std(sim_steps), mean(sim_rewards), std(sim_rewards)

end

end #module
