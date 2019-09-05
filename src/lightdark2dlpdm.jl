using Discretizers
import LPDM: default_action, next_actions, isterminal, bv_action_pool, adaptive_actions
import POMDPs: rand, actions

mutable struct LightDark2DLpdm <: AbstractLD2
# @with_kw mutable struct LightDark2DLpdm <: AbstractLD2
    min_noise::Float64
    min_noise_loc::Float64
    Q::Matrix{Float64}
    R::Matrix{Float64}
    term_radius::Float64
    n_bins::Int         # per linear dimension
    max_xy::Float64     # assume symmetry in x and y for simplicity
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    lindisc::LinearDiscretizer
    init_dist::Any
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
    reward_mode::Symbol
    action_space_type::Symbol
    standard_moves::Vector{Float64}
    extended_moves::Vector{Float64}
    base_action_space::Vector{LD2Action}
    standard_action_space::Vector{LD2Action}
    extended_action_space::Vector{LD2Action}

    function LightDark2DLpdm(;
                             action_mode            ::Symbol   = :adaptive,
                             obs_mode               ::Symbol   = :continuous,
                             reward_mode            ::Symbol   = :quadratic,
                             n_bins                 ::Int64    = 10, # per linear dimension
                             max_actions            ::Int64    = 150,
                             n_new_actions          ::Int64    = 1,# number of new actions to select at the same time in SA
                             action_range_fraction  ::Float64  = 0.5,
                             max_exploit_visits     ::Int64    = 25,
                             max_belief_clusters    ::Int64    = 8
                             )
        this = new()
        this.min_noise               = 0.0
        this.min_noise_loc           = 5.0
        this.Q                       = diagm(0=>[0.5, 0.5])
        this.R                       = diagm(0=>[0.5, 0.5])
        this.term_radius             = 0.05
        this.n_bins                  = n_bins # per linear dimension
        this.max_xy                  = 10.0     # assume symmetry in x and y for simplicity
        this.bin_edges               = collect(-this.max_xy:(2*this.max_xy)/this.n_bins:this.max_xy)
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
        this.exploit_visits          = max_exploit_visits
        this.max_belief_clusters     = max_belief_clusters
        this.standard_moves          = [1.0, 0.1, 0.01]
        this.extended_moves          = vcat(1*this.standard_moves,
                                       5*this.standard_moves)
        this.standard_action_space   = permute(this.standard_moves)
        this.extended_action_space   = permute(this.extended_moves)
        this.reward_mode             = reward_mode
        return this
    end
end

LPDM.max_belief_clusters(p::LightDark2DLpdm) = p.max_belief_clusters

# Version with discrete observations
function generate_o(p::LightDark2DLpdm, sp::Float64, rng::AbstractRNG)
    o = rand(rng, observation(p, sp))
    o_disc = p.bin_centers[encode(p.lindisc,o)]
    return o_disc
end

# For bounds calculations
# POMDPs.actions(p::LightDark2DLpdm, ::Bool) = p.extended_moves
function POMDPs.actions(p::LightDark2DLpdm, ::Bool)
    if p.action_mode == :standard
        # println("POMDP 2D actions: standard actions")
        # return vcat(-p.standard_action_space, [0.0], p.standard_action_space)
        return p.standard_moves
    elseif p.action_mode ∈ [:extended, :blind_vl, :adaptive]
        # println("POMDP 2D actions: extended actions")
        # return vcat(-p.extended_action_space, [0.0], p.extended_action_space)
        return p.extended_moves
    else
        error("Action space type $(p.action_mode) is not valid for POMDP of type $(typeof(p))")
    end
end

# # for tree construction
# function POMDPs.actions(p::LightDark2DLpdm)
#      if p.action_mode == :standard
#          return p.standard_action_space
#      elseif p.action_mode == :extended
#          return p.extended_action_space
#      elseif p.action_mode ∈ [:blind_vl, :adaptive]
#          return []
#      else
#          error("Action space type $(p.action_mode) is not valid for POMDP of type $(typeof(p))")
#      end
# end

# for tree construction
function POMDPs.actions(p::LightDark2DLpdm)
     if p.action_mode == :standard
         return p.standard_action_space
     elseif p.action_mode == :extended
         return p.extended_action_space
     elseif p.action_mode == :blind_vl
         return p.standard_action_space
     elseif p.action_mode == :adaptive
         return []
     else
         error("Action space type $(p.action_mode) is not valid for POMDP of type $(typeof(p))")
     end
end

POMDPs.actions(p::LightDark2DLpdm, ::LD2State) = actions(p::LightDark2DLpdm)

LPDM.max_actions(pomdp::LightDark2DLpdm) = pomdp.max_actions

# For adaptive action selection ("simulated annealing")
function LPDM.next_actions(pomdp::LightDark2DLpdm,
                           depth::Int64,
                           current_action_space::Vector{LD2Action},
                           a_star::LD2Action,
                           n_visits::Int64,
                           T_solver::Float64, # "temperature"
                           rng::RNGVector)::Vector{LD2Action}

    # simulated annealing temperature
    if isempty(current_action_space) # initial request
        # return vcat(-pomdp.standard_action_space, [0], pomdp.standard_action_space)
        if depth > 1
            return pomdp.standard_action_space
            # return pomdp.extended_action_space #DEBUG
        else
            # return pomdp.extended_action_space   #DEBUG, let's try this...
            return pomdp.standard_action_space
        end
    end

    l_initial = length(pomdp.standard_action_space)

    # don't count initial "seed" actions in computing T
    T_actions = 1 - (length(current_action_space) - l_initial)/(LPDM.max_actions(pomdp) - l_initial)
    T = minimum([T_solver T_actions])
    # adj_exploit_visits = pomdp.exploit_visits * (1-T) # exploit more as T decreases

    # generate new action(s)
    # if (n_visits > adj_exploit_visits) && (length(current_action_space) < LPDM.max_actions(pomdp))
    if length(current_action_space) < LPDM.max_actions(pomdp)
        return LPDM.adaptive_actions(pomdp, T, a_star, current_action_space, rng)
    else
        return Vec2[]
    end
end

function LPDM.adaptive_actions(pomdp::LightDark2DLpdm,
                               T::Float64,
                               a_star::LD2Action,
                               current_action_space::Vector{LD2Action},
                               rng::RNGVector)

    # Use the full range as initial radius to accomodate points at the edges of it
    radius = abs(pomdp.action_limits[2]-pomdp.action_limits[1]) * pomdp.action_range_fraction * T
    new_actions = Vector{LD2Action}(undef,pomdp.n_new_actions)

    a_x = NaN
    a_y = NaN
    for i in 1:pomdp.n_new_actions
       in_set = true
       while in_set
           a_x = (rand(rng, Uniform(a_star[1] - radius, a_star[1] + radius)))
           a_y = (rand(rng, Uniform(a_star[2] - radius, a_star[2] + radius)))
           a_x = clamp(a_x, pomdp.action_limits[1], pomdp.action_limits[2]) # if outside action space limits, clamp to them
           a_y = clamp(a_y, pomdp.action_limits[1], pomdp.action_limits[2])
           in_set = (Vec2(a_x,a_y) ∈ current_action_space) || (Vec2(a_x,a_y) ∈ new_actions)
       end
       new_actions[i]=Vec2(a_x,a_y)
    end
    return new_actions
end

function LPDM.bv_action_pool(pomdp::LightDark2DLpdm,
                             M::Int64,  # the number of actions to return in the pool
                             rng::RNGVector)

    # TODO: Create a formal sampler for RNGVector when there is time
    a_pool_x = [rand(rng, Uniform(pomdp.action_limits[1], pomdp.action_limits[2])) for i ∈ 1:M]
    a_pool_y = [rand(rng, Uniform(pomdp.action_limits[1], pomdp.action_limits[2])) for i ∈ 1:M]
    a_pool = [LD2Action(a_pool_x[i],a_pool_y[i]) for i in 1:M]
    σ_pool = std2d(hcat(a_pool_x, a_pool_y),[0.0,0.0]) # in our case distance to 0 (center of the domain) is just the abs. value of an action

    return a_pool, σ_pool
end

# TODO: Find a standard function for this and replace
function std2d(points2d::Array{Float64}, mean::Vector{Float64})
    norm_sum = 0.0 # L2 norm sum
    # println(points2d)
    N=size(points2d)[1]
    for i in 1:N
        norm_sum += norm(mean-points2d[i,:])
    end
    return norm_sum/sqrt(N-1)
end
