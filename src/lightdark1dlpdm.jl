using Discretizers
import LPDM: default_action, next_actions, equivalent
import POMDPs: rand, actions

mutable struct LightDark1DLpdm <: AbstractLD1
# @with_kw mutable struct LightDark1DLpdm <: AbstractLD1
    min_noise::Float64
    min_noise_loc::Float64
    Q::Float64
    R::Float64
    term_radius::Float64
    n_bins::Int         # per linear dimension
    max_x::Float64     # assume symmetry in x and y for simplicity
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    lindisc::LinearDiscretizer
    discount::Float64
    count::Int
    n_rand::Int
    resample_std::Float64
    exploit_visits::Int64
    max_actions::Int64
    n_new_actions::Int64
    action_range_fraction::Float64
    max_belief_clusters::Int64
    action_limits::Tuple{Float64,Float64}
    action_mode::Symbol
    obs_mode::Symbol
    obs_epsilon::Float64
    reward_mode::Symbol
    base_action_space::Vector{LD1Action}
    standard_action_space::Vector{LD1Action}
    extended_action_space::Vector{LD1Action}


    function LightDark1DLpdm(;
                             action_mode            ::Symbol   = :adaptive,
                             obs_mode               ::Symbol   = :continuous,
                             reward_mode            ::Symbol   = :quadratic,
                             n_bins                 ::Int64    = 10, # per linear dimension
                             max_actions            ::Int64    = 50,
                             n_new_actions          ::Int64    = 1,# number of new actions to select at the same time in SA
                             action_range_fraction  ::Float64  = 0.5,
                             max_exploit_visits     ::Int64    = 25,
                             max_belief_clusters    ::Int64    = 4
                             )
        this = new()
        this.min_noise               = 0.0
        this.min_noise_loc           = 5.0
        this.Q                       = 0.5
        this.R                       = 0.5
        this.term_radius             = 0.05
        this.n_bins                  = n_bins
        this.max_x                   = 10     # assume symmetry in x and y for simplicity
        this.bin_edges               = collect(-this.max_x:(2*this.max_x)/this.n_bins:this.max_x)
        this.bin_centers             = [(this.bin_edges[i]+this.bin_edges[i+1])/2 for i=1:this.n_bins]
        this.lindisc                 = LinearDiscretizer(this.bin_edges)
        this.discount                = 1.0
        this.count                   = 0
        this.n_rand                  = 0
        this.resample_std            = 0.5 # st. deviation for particle resampling
        this.max_actions             = max_actions
        this.n_new_actions           = n_new_actions
        this.action_range_fraction   = action_range_fraction
        this.action_limits           = (-5.0,5.0)
        this.action_mode             = action_mode
        this.obs_mode                = obs_mode
        this.obs_epsilon             = 0.5
        this.exploit_visits          = max_exploit_visits
        this.max_belief_clusters     = max_belief_clusters
        this.standard_action_space    = [1.0, 0.1, 0.01]
        this.extended_action_space   = vcat(1*this.standard_action_space,
                                            2*this.standard_action_space,
                                            3*this.standard_action_space,
                                            4*this.standard_action_space,
                                            5*this.standard_action_space)
        this.reward_mode                  = reward_mode
        return this
    end
end

# POMDPs.actions(p::LightDark1DLpdm) = vcat(-POMDPs.actions(p, true), [0.0], POMDPs.actions(p,true))
LPDM.default_action(p::LightDark1DLpdm) = 0.00
LPDM.default_action(p::LightDark1DLpdm, ::Vector{LPDMParticle{LD1State}}) = LPDM.default_action(p)

POMDPs.rand(p::LightDark1DLpdm, s::LD1State, rng::LPDM.RNGVector) = norminvcdf(s, p.resample_std, rand(rng)) # for resampling


function POMDPs.actions(p::LightDark1DLpdm)
    if p.action_mode == :standard
        # println("POMDP 1D actions: standard actions")
        return vcat(-p.standard_action_space, p.standard_action_space)
    elseif p.action_mode ∈ [:extended, :blind_vl, :adaptive] # the latter two use this for bounds calculations
        # println("POMDP 1D actions: extended actions")
        return vcat(-p.extended_action_space, p.extended_action_space)
    else
        error("Action space $(p.action_mode) is not valid for POMDP of type $(typeof(p))")
    end
end
# POMDPs.actions(pomdp::LightDark1DLpdm) = vcat(-pomdp.extended_action_space, pomdp.extended_action_space)

LPDM.max_actions(pomdp::LightDark1DLpdm) = pomdp.max_actions

# For "simulated annealing"
function LPDM.next_actions(pomdp::LightDark1DLpdm,
                           depth::Int64,
                           current_action_space::Vector{LD1Action},
                           a_star::LD1Action,
                           n_visits::Int64,
                           T_solver::Float64, # "temperature"
                           rng::RNGVector)::Vector{LD1Action}

    # println("POMDP: adaptive actions")
    new_actions = Vector{LD1Action}(undef,pomdp.n_new_actions)
    initial_space = vcat(-pomdp.standard_action_space, pomdp.standard_action_space)

    # simulated annealing temperature
    if isempty(current_action_space) # initial request
        # return vcat(-pomdp.standard_action_space, [0], pomdp.standard_action_space)
        return initial_space
    end

    l_initial = length(initial_space)

    # don't count initial "seed" actions in computing T_actions
    T_actions = 1 - (length(current_action_space) - l_initial)/(LPDM.max_actions(pomdp) - l_initial)
    T = minimum([T_solver T_actions]) # use the lowest "temperature"

    # adj_exploit_visits = pomdp.exploit_visits * (1-T) # exploit more as T decreases

    # generate new action(s)
    # if (n_visits > adj_exploit_visits) && (length(current_action_space) < LPDM.max_actions(pomdp))
    if length(current_action_space) < LPDM.max_actions(pomdp)

        # Use the full range as initial radius to accomodate points at the edges of it
        radius = abs(pomdp.action_limits[2]-pomdp.action_limits[1]) * pomdp.action_range_fraction * T # DEBUG: testing 0.5

        in_set = true
        a = NaN
        for i in 1:pomdp.n_new_actions
            in_set = true
            while in_set
                a = (rand(rng, Uniform(a_star - radius, a_star + radius)))
                a = clamp(a, pomdp.action_limits[1], pomdp.action_limits[2]) # if outside action space limits, clamp to them
                in_set = a ∈ current_action_space
            end
            new_actions[i]=a
        end
        # println("a_star: $a_star, T: $T, radius: $radius, a: $a")
        return new_actions # New action, returned as a one element vector.
    else
        return LD1Action[]
    end
end

# version for Blind Value
function LPDM.next_actions(pomdp::LightDark1DLpdm,
                           current_action_space::Vector{LD1Action},
                           Q::Vector{Float64},
                           n_visits::Int64,
                           rng::RNGVector)::Vector{LD1Action}

    # println("POMDP: blind value actions")
     initial_space = vcat(-pomdp.standard_action_space, pomdp.standard_action_space)
     # initial_space = [0.0]
     M = pomdp.max_actions

       # simulated annealing temperature
     if isempty(current_action_space) # initial request
           return initial_space
     end

    if (n_visits > pomdp.exploit_visits) && (length(current_action_space) < LPDM.max_actions(pomdp))
        # TODO: Create a formal sampler for RNGVector when there is time
        Apool = [rand(rng, Uniform(pomdp.action_limits[1], pomdp.action_limits[2])) for i in 1:M]
        σ_known = std(Q)
        σ_pool = std(Apool) # in our case distance to 0 (center of the domain) is just the abs. value of an action
        ρ = σ_known/σ_pool
        bv_vector = [bv(a,ρ,current_action_space,Q) for a in Apool]

        return [Apool[argmax(bv_vector)]] # New action, returned as a one element vector.
    else
        return []
    end
end

# Blind Value function
function bv(a::LD1Action, ρ::Float64, Aexpl::Vector{LD1Action}, Q::Vector{Float64})::LD1Action
    scores = [ρ*abs(a-Aexpl[i])+Q[i] for i in 1:length(Aexpl)]
    return Aexpl[argmin(scores)]
end

LPDM.max_belief_clusters(p::LightDark1DLpdm) = p.max_belief_clusters
LPDM.equivalent(pomdp::LightDark1DLpdm, o1::LD1Obs, o2::LD1Obs)::Bool = abs(o2-o1) < obs_epsilon


# NOTE: OLD VERSION. implements "fast" simulated annealing
# function LPDM.next_actions(pomdp::LightDark1DLpdm,
#                            current_action_space::Vector{LD1Action},
#                            a_star::LD1Action,
#                            n_visits::Int64,
#                            rng::RNGVector)::Vector{LD1Action}
#
#     initial_space = vcat(-pomdp.standard_action_space, pomdp.standard_action_space)
#
#     # simulated annealing temperature
#     if isempty(current_action_space) # initial request
#         # return vcat(-pomdp.standard_action_space, [0], pomdp.standard_action_space)
#         return initial_space
#     end
#
#     l_initial = length(initial_space)
#
#     # don't count initial "seed" actions in computing T
#     T = 1 - (length(current_action_space) - l_initial)/(LPDM.max_actions(pomdp) - l_initial)
#     adj_exploit_visits = pomdp.exploit_visits * (1-T) # exploit more as T decreases
#
#     # generate new action(s)
#     if (n_visits > adj_exploit_visits) && (length(current_action_space) < LPDM.max_actions(pomdp))
#         # NOTE: use the actual point for now, convert to a distribution around it later
#         left_d = abs(pomdp.action_limits[1]-a_star)
#         right_d = abs(pomdp.action_limits[2]-a_star)
#         d = left_d > right_d ? -left_d : right_d
#         # println("a_star: $a_star, T: $T, left_d: $left_d, right_d: $right_d, d: $d, a: $(a_star + d*T)")
#         return [a_star + d*T] # New action, returned as a one element vector. Value is scaled by T.
#     else
#         return []
#     end
# end

# # Hard-coded version for now for debugging
# function LPDM.next_actions(pomdp::LightDark1DLpdm, current_action_space::Vector{LD1Action})::Vector{LD1Action}
#     if isempty(current_action_space) # initial request
#         return vcat(-pomdp.standard_action_space, [0], pomdp.standard_action_space)
#     end
#
#     # index of the new action in the extended_action_space
#     n = round(Int64, 0.5*(length(current_action_space) - (2*length(pomdp.standard_action_space) + 1))) + 1
#     if (length(current_action_space) < pomdp.max_actions -1) && (n <= length(pomdp.extended_action_space))
#         # println("current: $current_action_space")
#         # accounting for zero with the first +1; 0.5 because we add in pairs.
#
#         return [-pomdp.extended_action_space[n], pomdp.extended_action_space[n]] # return as a 2-element vector
#     else
#         return []
#     end
# end
