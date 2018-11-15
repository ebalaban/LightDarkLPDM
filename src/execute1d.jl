module execute1d
using LPDM
using POMDPs, Parameters, StaticArrays, D3Trees, Distributions, SparseArrays
using Combinatorics
using StatsFuns

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

function batch_execute(;sims::Int64=1)
    test=Array{LPDMTest}(undef,0)
    push!(test, LPDMTest(:despot, :small))
    push!(test, LPDMTest(:despot, :large))
    push!(test, LPDMTest(:lpdm, :bv)) # blind value
    push!(test, LPDMTest(:lpdm, :sa)) # simulated annealing

    scen=Array{LPDMTest}(undef,0)
    push!(scen, LPDMScenario(LD1State(-4.1)))
    push!(scen, LPDMScenario(LD1State(0.9)))
    push!(scen, LPDMScenario(LD1State(4.7)))
    push!(scen, LPDMScenario(LD1State(2*π)))

    f = open("test_results.txt", "w")
    for i in 1:length(scen)
        write(f,"SCENARIO $i, s0=$(scen[i].s0)\n")
        write(f,"mode\taction space\treward(std)\n")
        write(f,"=====================================================\n")
        for t in test
            reward, std = execute(solver       = t.mode,
                                  action_space = t.action_space,
                                  n_sims       = sims,
                                  s0           = scen[i].s0)
            write(f,"$(t.mode)\t$(t.action_space)\t$reward($std)\n")
        end
        write(f,"=====================================================\n")
        write(f,"$n simulations per test\n\n")
    end
    close(f)
end

function execute(;vis::Vector{Int64}=Int64[],
                solv_mode::Symbol=:lpdm,
                n_sims::Int64=1,
                s0::LD1State=LD1State(1.9))#n_sims::Int64 = 100)

    if solver == :despot
        p = LightDark1DDespot()
    elseif solver == :lpdm
        p = LightDark1DLpdm()
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
                                                                            debug = 1,
                                                                            time_per_move = 5.0,  #sec
                                                                            sim_len = -1,
                                                                            search_depth = 50,
                                                                            n_particles = 20,
                                                                            # seed = UInt32(5),
                                                                            seed = UInt32(2*sim+1),
                                                                            # max_trials = 10)
                                                                            max_trials = -1,
                                                                            mode = solv_mode)

    #---------------------------------------------------------------------------------
        # Belief
        bu = LPDMBeliefUpdater(p, n_particles = solver.config.n_particles);  # initialize belief updater
        initial_states = state_distribution(p, s, solver.config, world_rng)     # create initial  distribution
        current_belief = LPDM.create_belief(bu)                       # allocate an updated belief object
        LPDM.initialize_belief(bu, initial_states, current_belief)    # initialize belief
        # show(current_belief)
        updated_belief = LPDM.create_belief(bu)
    #---------------------------------------------------------------------------------

        policy::LPDMPolicy = POMDPs.solve(solver, p)

        sim_steps::Int64 = 1
        r::Float64 = 0.0

        println("=============== REPETITION # $sim ================")

        val, run_time, bytes, gctime, memallocs =
        @timed while !isterminal(p, s) && (solver.config.sim_len == -1 || sim_steps <= solver.config.sim_len)

            println("")
            println("=============== Step $sim_steps ================")
            show(current_belief)
            println("s: $s")
            a = POMDPs.action(policy, current_belief)
            println("a: $a")

            s, o, r = POMDPs.generate_sor(p, s, a, world_rng)
            println("s': $s")
            println("o': $o")
            println("r: $r")
            push!(rewards, r)
            println("=======================================")

            # error("$(@which(POMDPs.update(bu, current_belief, a, o, updated_belief)))")
            # update belief
            POMDPs.update(bu, current_belief, a, o, updated_belief)
            current_belief = deepcopy(updated_belief)
            # show(updated_belief) #NOTE: don't show for now

            if LPDM.isterminal(p, current_belief.particles)
                println("Terminal belief. Execution completed.")
                show(current_belief)
                break
            end

            if sim_steps ∈ vis
                t = LPDM.d3tree(solver)
                # # show(t)
                inchrome(t)
                # blink(t)
            end
            sim_steps += 1
        end


        # # Compute discounted reward
        # discounted_reward::Float64 = 0.0
        # multiplier::Float64 = 1.0
        # for r in rewards
        #     discounted_reward += multiplier * r
        #     multiplier *= p.discount
        # end
        # println("Discounted reward: $discounted_reward")
    end
    return sim_steps, sum(step_rewards), discounted_reward, run_time
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
