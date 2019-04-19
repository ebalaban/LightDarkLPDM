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

struct ProblemConfig
    n_bins                  ::Int64
    max_actions             ::Int64
    n_new_actions           ::Int64 # number of new actions to select at the same time in SA
    action_range_fraction   ::Float64
    max_exploit_visits      ::Int64
    max_belief_clusters     ::Int64
end

struct LPDMTest
    action_mode         ::Symbol
    obs_mode            ::Symbol
    reward_mode         ::Symbol
    problem_config      ::ProblemConfig
    n_sims              ::Int64
end

struct LPDMScenario{S}
    s0::S
end

# reward function options - :quadratic or :fixed
function batch_execute(;dims::Int64=1)

    # General test parameters
    reward_mode                 = :quadratic
    n_sims                      = 50
    vis::Vector{Int64}          = Int64[]
    debug                       = 0

    # 1D problem configuration
    n_bins1d                    = 10
    max_actions1d               = 50
    n_new_actions1d             = 2
    action_range_fraction1d     = 0.25
    max_exploit_visits1d        = 0
    max_belief_clusters1d       = 100

    # 2D problem configuration
    n_bins2d                    = 10 # per dimension
    max_actions2d               = 100
    n_new_actions2d             = 2
    action_range_fraction2d     = 0.25
    max_exploit_visits2d        = 0
    max_belief_clusters2d       = 100

    pconfig1d = ProblemConfig(n_bins1d, max_actions1d, n_new_actions1d, action_range_fraction1d, max_exploit_visits1d, max_belief_clusters1d)
    pconfig2d = ProblemConfig(n_bins2d, max_actions2d, n_new_actions2d, action_range_fraction2d, max_exploit_visits2d, max_belief_clusters2d)

    # 1D solver configuration (some are left as defaults)
    sconfig1d = LPDMConfig(
            search_depth        = 30,
            seed                = 0xffffffff, # assigned per scenario (will cause an error if not assigned)
            time_per_move       = 1.0,
            n_particles         = 25,
            sim_len             = -1,
            max_trials          = -1,
            debug               = debug,
            action_mode         = :tbd,     # assigned per test (will cause an error if not assigned)
            obs_mode            = :tbd)     # assigned per test (will cause an error if not assigned)

    # 2D solver configuration
    sconfig2d = LPDMConfig(
            search_depth        = 30,
            seed                = 0xffffffff,  # assigned per scenario (will cause an error if not assigned)
            time_per_move       = 2.0,
            n_particles         = 50,
            sim_len             = -1,
            max_trials          = -1,
            debug               = debug,
            action_mode         = :tbd,     # assigned per test (will cause an error if not assigned)
            obs_mode            = :tbd)     # assigned per test (will cause an error if not assigned)


    if dims == 1
        S = LD1State
        A = LD1Action
        O = LD1Obs
        B = LDBounds1d{S,A,O}
        pconfig = pconfig1d
        sconfig = sconfig1d
    elseif dims == 2
        S = LD2State
        A = LD2Action
        O = LD2Obs
        B = LDBounds2d{S,A,O}
        pconfig = pconfig2d
        sconfig = sconfig2d
    else
        error("Invalid number of dimensions $dims")
    end

    test = Array{LPDMTest}(undef,0)

    # DISCRETE OBSERVATIONS
    # push!(test, LPDMTest(:standard, :discrete, reward_mode, pconfig, n_sims))
    # push!(test, LPDMTest(:extended, :discrete, reward_mode, pconfig, n_sims))
    # push!(test, LPDMTest(:blind_vl, :discrete, reward_mode, pconfig, n_sims))
    # push!(test, LPDMTest(:adaptive, :discrete, reward_mode, pconfig, n_sims))

    # CONTINUOUS OBSERVATIONS
    push!(test, LPDMTest(:standard, :continuous, reward_mode, pconfig, n_sims))
    # push!(test, LPDMTest(:extended, :continuous, reward_mode, pconfig, n_sims))
    # push!(test, LPDMTest(:blind_vl, :continuous, reward_mode, pconfig, n_sims))
    # push!(test, LPDMTest(:adaptive, :continuous, reward_mode, pconfig, n_sims))

    scen=Array{LPDMScenario{S}}(undef,0)

    if dims == 1
        # push!(scen, LPDMScenario(LD1State(-2*π)))
        # push!(scen, LPDMScenario(LD1State(π/2)))
        push!(scen, LPDMScenario(LD1State(3/2*π)))
        # push!(scen, LPDMScenario(LD1State(2*π)))
    elseif dims == 2
        push!(scen, LPDMScenario(LD2State(-2*π, π)))
        push!(scen, LPDMScenario(LD2State(π/2, -π/2)))
        push!(scen, LPDMScenario(LD2State(π, 2*π)))
        push!(scen, LPDMScenario(LD2State(2*π, -π)))
    end

    # Dummy execution, just to make sure all the code is compiled and loaded,
    # to improve uniformity of subsequent executions.
    # execute(solv_mode = :despot, action_space_type = :small, n_sims = 1, s0 = LD2State(π,π), debug = 0)
    # execute(solv_mode = :lpdm, action_space_type = :adaptive, n_sims = 1, s0 = LD2State(π,π), debug = 0)

    f = open("results_" * Dates.format(now(),"yyyy-mm-dd_HH_MM") * ".txt", "w")

    # Printf.@printf(f,"GENERAL SOLVER PARAMETERS\n")
    # Printf.@printf(f,"\tsteps:\t\t\t%d\n", steps)
    # Printf.@printf(f,"\ttime per move:\t\t\t%f\n", time_per_move)
    # Printf.@printf(f,"\tsearch depth:\t\t\t%d\n", search_depth)
    # Printf.@printf(f,"\tN particles:\t\t\t%d\n", n_particles)
    # Printf.@printf(f,"\tmax trials:\t\t\t%d\n\n", max_trials)
    print(f,sconfig)
    Printf.@printf(f,"\nPROBLEM PARAMETERS\n")
    Printf.@printf(f,"\tdimensions:\t\t\t%d\n",             dims)
    Printf.@printf(f,"\tN bins (per dim):\t\t\t%d\n",       pconfig.n_bins)
    Printf.@printf(f,"\tmax actions:\t\t\t%d\n",            pconfig.max_actions)
    Printf.@printf(f,"\tN new actions:\t\t\t%d\n",          pconfig.n_new_actions)
    Printf.@printf(f,"\taction range fraction:\t\t\t%f\n",  pconfig.action_range_fraction)
    Printf.@printf(f,"\tmax exploit. visits:\t\t\t%d\n",    pconfig.max_exploit_visits)
    Printf.@printf(f,"\tmax belief clusters:\t\t\t%d\n",    pconfig.max_belief_clusters)
    Printf.@printf(f,"\ttests per scenario:\t\t\t%d\n\n",   n_sims)

    for i in 1:length(scen)
        if debug >= 0
            println("")
            println("SCENARIO $i, s0 = $(scen[i].s0)")
            println("------------------------")
        end

        Printf.@printf(f,"SCENARIO %d, s0 = %s\n", i, "$(scen[i].s0)")
        Printf.@printf(f,"================================================================\n")
        Printf.@printf(f,"ACT. MODE\t\tOBS. MODE\t\t\tSTEPS (STD)\t\t\t\tREWARD (STD)\n")
        Printf.@printf(f,"================================================================\n")
        for t in test
            if debug >= 0
                println("dimensions: $dims, actions: $(t.action_mode), observations: $(t.obs_mode), rewards: $(t.reward_mode)")
            end
            steps_avg, steps_std, reward_avg, reward_std =
                        run_scenario(scen[i].s0, t, A, O, B, sconfig,
                                     dims      = dims,
                                     n_sims    = n_sims,
                                     vis       = vis)

            Printf.@printf(f,"%s\t\t%s\t\t\t%05.2f (%06.2f)\t\t%06.2f (%06.2f)\n",
                                            string(t.action_mode), string(t.obs_mode), steps_avg, steps_std, reward_avg, reward_std)
            flush(f)
            debug >=0 && println("\n*******************************************************************************************************")
            debug >=0 && println("STATS: actions=$(t.action_mode), observations=$(t.obs_mode), steps = $(steps_avg) ($steps_std), reward = $(reward_avg) ($reward_std)")
            debug >=0 && println("*******************************************************************************************************\n")
        end
        Printf.@printf(f,"================================================================\n\n")
    end
    close(f)
end

function run_scenario(s0::S, test::LPDMTest, A::Type, O::Type, B::Type, sconfig::LPDMConfig;
                      dims::Int64                 = 1,
                      n_sims::Int64               = 1,
                      vis::Vector{Int64}          = Int64[]
                      ) where {S}

    if dims == 1
        p = LightDark1DLpdm(action_mode             = test.action_mode,
                            obs_mode                = test.obs_mode,
                            reward_mode             = test.reward_mode,
                            n_bins                  = test.problem_config.n_bins,
                            max_actions             = test.problem_config.max_actions,
                            n_new_actions           = test.problem_config.n_new_actions,
                            action_range_fraction   = test.problem_config.action_range_fraction,
                            max_exploit_visits      = test.problem_config.max_exploit_visits,
                            max_belief_clusters     = test.problem_config.max_belief_clusters
                            )
    elseif dims == 2
        p = LightDark2DLpdm(action_mode             = test.action_mode,
                            obs_mode                = test.obs_mode,
                            reward_mode             = test.reward_mode,
                            n_bins                  = test.problem_config.n_bins,
                            max_actions             = test.problem_config.max_actions,
                            n_new_actions           = test.problem_config.n_new_actions,
                            action_range_fraction   = test.problem_config.action_range_fraction,
                            max_exploit_visits      = test.problem_config.max_exploit_visits,
                            max_belief_clusters     = test.problem_config.max_belief_clusters
                            )
    end

    sim_rewards = Vector{Float64}(undef,n_sims)
    sim_steps   = Vector{Int64}(undef,n_sims)

    for sim in 1:n_sims

        world_rng = RNGVector(1, UInt32(sim))
        LPDM.set!(world_rng, 1)
        step_rewards::Array{Float64}     = Vector{Float64}(undef,0)

        sconfig.seed        = UInt32(2*sim+1)
        sconfig.action_mode = test.action_mode
        sconfig.obs_mode    = test.obs_mode
        solver = LPDM.LPDMSolver{S, A, O, B, RNGVector}(config = sconfig)

    #---------------------------------------------------------------------------------
        # Belief
        bu = LPDMBeliefUpdater(p,
                               n_particles = sconfig.n_particles,
                               seed = UInt32(3*sim+1));  # initialize belief updater
        initial_states = state_distribution(p, s0, sconfig, world_rng)     # create initial  distribution
        current_belief = LPDM.create_belief(bu)                       # allocate an updated belief object
        LPDM.initialize_belief(bu, initial_states, current_belief)    # initialize belief
        updated_belief = LPDM.create_belief(bu)

    #---------------------------------------------------------------------------------

        policy::LPDMPolicy = POMDPs.solve(solver, p)
        s = s0
        step::Int64 = 0
        r::Float64 = 0.0

        if sconfig.debug >= 0
            println("")
            println("*** SIM $sim (dimensions: $dims, action mode: $(test.action_mode), obs mode: $(test.obs_mode), s0: $s0)***")
            println("")
        end

        val, run_time, bytes, gctime, memallocs =
        @timed while !isterminal(p, s) && (sconfig.sim_len == -1 || step < sconfig.sim_len)
            step += 1

            if sconfig.debug >= 1
                println("")
                println("=============== Step $step ================")
                show(current_belief)
            end

            a = POMDPs.action(policy, current_belief)
            if sconfig.debug >= 1
                println("s: $s")
                println("a: $a")
            end

            s, o, r = POMDPs.generate_sor(p, s, a, world_rng)
            push!(step_rewards, r)
            if sconfig.debug >= 1
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
                if sconfig.debug >= 1
                    println("Terminal belief. Execution completed.")
                    show(current_belief)
                end
                break
            end

            if step ∈ vis
                t = LPDM.d3tree(solver,
                                detect_repeat=false,
                                title="Step $step",
                                init_expand=2)
                # # show(t)
                inchrome(t)
                # blink(t)
            end
            if sconfig.debug >= 1
                println("root actions: $(solver.root.action_space)")
            end
        end

        sim_steps[sim] = step
        sim_rewards[sim] = sum(step_rewards)
        sconfig.debug >= 0 && println("steps=$(sim_steps[sim]), reward=$(sim_rewards[sim])")
    end

    return mean(sim_steps), std(sim_steps), mean(sim_rewards), std(sim_rewards)

end

end #module
