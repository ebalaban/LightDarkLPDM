module execute2d
using LPDM
using POMDPs, Parameters, StaticArrays, D3Trees, Distributions, SparseArrays
using Combinatorics
using StatsFuns
using Printf
using Dates

include("LightDarkPOMDPs.jl")
using LightDarkPOMDPs

include("LPDMBounds2d.jl")

struct LPDMTest
    mode::Symbol
    action_space::Symbol
    reward_func::Symbol
end

struct LPDMScenario
    s0::LD2State
end

# reward function options - :quadratic or :fixed
function batch_execute(;n::Int64=1, debug::Int64=1, reward_func=:quadratic)
    test=Array{LPDMTest}(undef,0)

    # push!(test, LPDMTest(:lpdm, :adapt, reward_func)) # simulated annealing
    # push!(test, LPDMTest(:despot, :small, reward_func))
    # push!(test, LPDMTest(:despot, :large, reward_func))
    push!(test, LPDMTest(:lpdm_bv, :bv, reward_func)) # blind value

    scen=Array{LPDMScenario}(undef,0)
    # push!(scen, LPDMScenario(LD2State(-2*π)))
    # push!(scen, LPDMScenario(LD2State(π/2)))
    push!(scen, LPDMScenario(LD2State(π, -π)))
    # push!(scen, LPDMScenario(LD2State(2*π)))

    # Dummy execution, just to make sure all the code is compiled and loaded,
    # to improve uniformity of subsequent executions.
    # execute(solv_mode = :despot, action_space_type = :small, n_sims = 1, s0 = LD2State(π,π), output = 0)
    # execute(solv_mode = :despot, action_space_type = :small, n_sims = 1, s0 = LD2State(3.70,7.34), output = debug)

    # General solver parameters
    steps::Int64                = -1
    time_per_move::Float64      = -1.0
    search_depth::Int64         = 30
    n_particles::Int64          = 50
    max_trials::Int64           = 500

    f = open("results_" * Dates.format(now(),"yyyy-mm-dd_HH_MM") * ".txt", "w")

    Printf.@printf(f,"GENERAL SOLVER PARAMETERS\n")
    Printf.@printf(f,"\treward function:\t\t\t%s\n", reward_func)
    Printf.@printf(f,"\tsteps:\t\t\t%d\n", steps)
    Printf.@printf(f,"\ttime per move:\t\t\t%f\n", time_per_move)
    Printf.@printf(f,"\tsearch depth:\t\t\t%d\n", search_depth)
    Printf.@printf(f,"\tN particles:\t\t\t%d\n", n_particles)
    Printf.@printf(f,"\tmax trials:\t\t\t%d\n\n", max_trials)

    for i in 1:length(scen)
        # write(f,"SCENARIO $i, s0 = $(scen[i].s0)\n\n")
        if debug >= 0
            println("")
            println("SCENARIO $i, s0 = $(scen[i].s0)")
            println("------------------------")
        end

        Printf.@printf(f,"SCENARIO %d, s0 = (%f,%f)\n", i, scen[i].s0[1], scen[i].s0[2])
        Printf.@printf(f,"==================================================================\n")
        # Printf.@printf(f,"mode\t\tact. space\t\tsteps(std)\t\treward(std)\n")
        Printf.@printf(f,"SOLVER\t\tACT. SPACE\t\tSTEPS (STD)\t\t\tREWARD (STD)\n")
        Printf.@printf(f,"==================================================================\n")
        for t in test
            if debug >= 0
                println("mode: $(t.mode), action space: $(t.action_space), reward: $(t.reward_func)")
            end
            steps_avg, steps_std, reward_avg, reward_std =
                        execute(solv_mode         = t.mode,
                                action_space_type = t.action_space,
                                n_sims            = n,
                                steps             = steps,
                                time_per_move     = time_per_move,
                                search_depth      = search_depth,
                                n_particles       = n_particles,
                                max_trials        = max_trials,
                                s0                = scen[i].s0,
                                output            = debug,
                                reward_func       = reward_func)

            Printf.@printf(f,"%s\t\t%s\t\t\t%05.2f (%06.2f)\t\t%06.2f (%06.2f)\n",
                            string(t.mode), string(t.action_space), steps_avg, steps_std, reward_avg, reward_std)
        end
        Printf.@printf(f,"==================================================================\n")
        Printf.@printf(f,"%d tests per scenario\n\n", n)
    end
    close(f)
end

function execute(;vis::Vector{Int64}        = Int64[],
                solv_mode::Symbol           = :despot,
                action_space_type::Symbol   = :small,
                reward_func::Symbol         = :quadratic,
                n_sims::Int64               = 1,
                steps::Int64                = -1,
                time_per_move::Float64      = -1.0,
                search_depth::Int64         = 30,
                n_particles::Int64          = 50,
                max_trials::Int64           = 500,
                s0::LD2State                = LD2State(0.5*π, 2*π),
                output::Int64               = 1
                )

    if solv_mode == :despot
        p = LightDark2DDespot(action_space_type, reward_func = reward_func)
    elseif (solv_mode == :lpdm || solv_mode == :lpdm_bv)
        p = LightDark2DLpdm(action_space_type, reward_func = reward_func)
    end

    sim_rewards = Vector{Float64}(undef,n_sims)
    sim_steps   = Vector{Int64}(undef,n_sims)

    for sim in 1:n_sims

        world_rng = RNGVector(1, UInt32(sim))
        LPDM.set!(world_rng, 1)

        # NOTE: restrict initial state to positive numbers only, for now
        # s::LD2State                  = LD2State(π);    # initial state
        step_rewards::Array{Float64}     = Vector{Float64}(undef,0)


        # NOTE: Original version
        # solver = LPDM.LPDMSolver{LD2State, LD2Action, LD2Obs, LDBounds2d{LD2State, LD2Action, LD2Obs}, RNGVector}(
        #                                                                     # rng = sim_rng,
        #                                                                     debug = output,
        #                                                                     time_per_move = 1.0,  #sec
        #                                                                     # time_per_move = 1.0,  #sec
        #                                                                     sim_len = -1,
        #                                                                     search_depth = 50,
        #                                                                     n_particles = 20,
        #                                                                     # seed = UInt32(5),
        #                                                                     seed = UInt32(2*sim+1),
        #                                                                     # max_trials = 10)
        #                                                                     max_trials = -1,
        #                                                                     mode = solv_mode)

        # NOTE: Test version
        solver = LPDM.LPDMSolver{LD2State, LD2Action, LD2Obs, LDBounds2d{LD2State, LD2Action, LD2Obs}, RNGVector}(
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
                                                                            mode = solv_mode)

        # #DEBUG: remove ###############################
        # # states=[Vec2(3.6971,7.3388), Vec2(3.5,-1.2), Vec2(7.1,7.3), Vec2(5.35,0.0)]
        # states=[Vec2(3.14,3.14)]
        # for st in states
        #     b = LDBounds2d{LD2State, LD2Action, LD2Obs}(p)
        #     lb, ub = LPDM.bounds(b,p,[LPDMParticle{Vec2}(st,1,1.0)],solver.config)
        #     println("s=$st, lb = $lb, ub = $ub, lba=$(best_lb_action(b)), uba=$(best_ub_action(b))")
        # end
        # error("stop for now")
        # ##############################################
    #---------------------------------------------------------------------------------
        # Belief
        bu = LPDMBeliefUpdater(p,
                               n_particles = solver.config.n_particles,
                               seed = UInt32(3*sim+1));  # initialize belief updater
        initial_states = state_distribution(p, s0, solver.config, world_rng)     # create initial  distribution
        # println(initial_states)
        current_belief = LPDM.create_belief(bu)                       # allocate an updated belief object
        LPDM.initialize_belief(bu, initial_states, current_belief)    # initialize belief
        # show(current_belief)
        updated_belief = LPDM.create_belief(bu)
        # println("$(length(actions(p))) actions: $(actions(p))")
        # error("")
    #---------------------------------------------------------------------------------

        policy::LPDMPolicy = POMDPs.solve(solver, p)
        s = s0
        step::Int64 = 0
        r::Float64 = 0.0

        # output >= 1 && println("=============== SIMULATION # $sim ================")
        if output >= 0
            println("")
            println("*** SIM $sim (mode: $solv_mode, action space: $action_space_type, s0: $s0)***")
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
                solv_mode == :lpdm && println("root actions: $(solver.root.action_space)")
            end
        end

        sim_steps[sim] = step
        sim_rewards[sim] = sum(step_rewards)
        output >= 0 && println("steps=$(sim_steps[sim]), reward=$(sim_rewards[sim])")
    end

    return mean(sim_steps), std(sim_steps), mean(sim_rewards), std(sim_rewards)

end

end #module
