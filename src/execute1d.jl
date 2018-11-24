module execute1d
using LPDM
using POMDPs, Parameters, StaticArrays, D3Trees, Distributions, SparseArrays
using Combinatorics
using StatsFuns
using Printf
using Dates

include("LightDarkPOMDPs.jl")
using LightDarkPOMDPs

include("LPDMBounds1d.jl")

struct LPDMTest
    mode::Symbol
    action_space::Symbol
end

struct LPDMScenario
    s0::LD1State
end

function batch_execute(;n::Int64=1, debug::Int64=1)
    test=Array{LPDMTest}(undef,0)
    push!(test, LPDMTest(:despot, :small))
    # push!(test, LPDMTest(:despot, :large))
    # push!(test, LPDMTest(:lpdm, :bv)) # blind value
    # push!(test, LPDMTest(:lpdm, :sa)) # simulated annealing

    scen=Array{LPDMScenario}(undef,0)
    push!(scen, LPDMScenario(LD1State(-5.1)))
    # push!(scen, LPDMScenario(LD1State(0.9)))
    # push!(scen, LPDMScenario(LD1State(4.7)))
    # push!(scen, LPDMScenario(LD1State(2*π)))

    f = open("test_results_" * Dates.format(now(),"yyyy-mm-dd_HH_MM") * ".txt", "w")
    for i in 1:length(scen)
        # write(f,"SCENARIO $i, s0 = $(scen[i].s0)\n\n")
        if debug >= 0
            println("")
            println("SCENARIO $i, s0 = $(scen[i].s0)")
            println("------------------------")
        end

        Printf.@printf(f,"SCENARIO %d, s0 = %f\n", i, scen[i].s0)
        Printf.@printf(f,"==================================================================\n")
        # Printf.@printf(f,"mode\t\tact. space\t\tsteps(std)\t\treward(std)\n")
        Printf.@printf(f,"MODE\t\tACT. SPACE\t\tSTEPS (STD)\t\t\tREWARD (STD)\n")
        Printf.@printf(f,"==================================================================\n")
        for t in test
            if debug >= 0
                println("mode: $(t.mode), action space: $(t.action_space)")
            end
            steps, steps_std, reward, reward_std =
                        execute(solv_mode         = t.mode,
                                action_space_type = t.action_space,
                                n_sims            = n,
                                s0                = scen[i].s0,
                                output            = debug)
            Printf.@printf(f,"%s\t\t%s\t\t\t%.2f(%.2f)\t\t\t%.2f (%.2f)\n",
                            string(t.mode), string(t.action_space), steps, steps_std, reward, reward_std)
        end
        Printf.@printf(f,"==================================================================\n")
        Printf.@printf(f,"%d simulations per test\n\n", n)
    end
    close(f)
end

function execute(;vis::Vector{Int64}=Int64[],
                solv_mode::Symbol=:lpdm,
                action_space_type::Symbol=:sa,
                n_sims::Int64=1,
                s0::LD1State=LD1State(1.9),
                output::Int64=1)#n_sims::Int64 = 100)

    if solv_mode == :despot
        p = LightDark1DDespot(action_space_type)
    elseif solv_mode == :lpdm
        p = LightDark1DLpdm(action_space_type)
    end

    sim_rewards = Vector{Float64}(undef,n_sims)
    sim_steps   = Vector{Int64}(undef,n_sims)

    for sim in 1:n_sims
        # world_seed  ::UInt32   = convert(UInt32, 42)

        world_rng = RNGVector(1, UInt32(sim))
        LPDM.set!(world_rng, 1)

        # NOTE: restrict initial state to positive numbers only, for now
        # s::LD1State                  = LD1State(π);    # initial state
        step_rewards::Array{Float64}     = Vector{Float64}(undef,0)
        # println("$(supertype(LightDark1DDespot)))")
        solver = LPDM.LPDMSolver{LD1State, LD1Action, LD1Obs, LDBounds1d{LD1State, LD1Action, LD1Obs}, RNGVector}(
                                                                            # rng = sim_rng,
                                                                            debug = output,
                                                                            time_per_move = -1.0,  #sec
                                                                            # time_per_move = 1.0,  #sec
                                                                            sim_len = -1,
                                                                            search_depth = 50,
                                                                            n_particles = 20,
                                                                            # seed = UInt32(5),
                                                                            seed = UInt32(2*sim+1),
                                                                            # max_trials = 10)
                                                                            max_trials = 20000,
                                                                            mode = solv_mode)

    #---------------------------------------------------------------------------------
        # Belief
        bu = LPDMBeliefUpdater(p,
                               n_particles = solver.config.n_particles,
                               seed = UInt32(3*sim+1));  # initialize belief updater
        initial_states = state_distribution(p, s0, solver.config, world_rng)     # create initial  distribution
        current_belief = LPDM.create_belief(bu)                       # allocate an updated belief object
        LPDM.initialize_belief(bu, initial_states, current_belief)    # initialize belief
        # show(current_belief)
        updated_belief = LPDM.create_belief(bu)
    #---------------------------------------------------------------------------------

        policy::LPDMPolicy = POMDPs.solve(solver, p)
        s = s0
        step::Int64 = 0
        r::Float64 = 0.0

        # output >= 1 && println("=============== SIMULATION # $sim ================")
        if output >= 0
            println("")
            println("*** SIM $sim ***")
            println("")
        end

        val, run_time, bytes, gctime, memallocs =
        @timed while !isterminal(p, s) && (solver.config.sim_len == -1 || step <= solver.config.sim_len)
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

            # error("$(@which(POMDPs.update(bu, current_belief, a, o, updated_belief)))")
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
                t = LPDM.d3tree(solver)
                # # show(t)
                inchrome(t)
                # blink(t)
            end
        end

        sim_steps[sim] = step
        sim_rewards[sim] = sum(step_rewards)
        # # Compute discounted reward
        # discounted_reward::Float64 = 0.0
        # multiplier::Float64 = 1.0
        # for r in rewards
        #     discounted_reward += multiplier * r
        #     multiplier *= p.discount
        # end
        # println("Discounted reward: $discounted_reward")
    end
    # println("SIM_STEPS: $sim_steps, SIM_REWARDS: $sim_rewards")
    return mean(sim_steps), std(sim_steps), mean(sim_rewards), std(sim_rewards)
    # return t
end

function test_move()
    p = LightDarkPOMDPs.LightDark1DDespot()

    coordinates = Vector{Tuple{Float64,Float64}}()
    push!(coordinates,(5.0,0.0))
    push!(coordinates,(-5.0,0.0))
    push!(coordinates,(3.14,4.50))

    for c in coordinates
        r,a = move(p,c[1],c[2])
        println("x1=$(c[1]), x2=$(c[2]), r=$r, a=$a")
    end
end

end #module
