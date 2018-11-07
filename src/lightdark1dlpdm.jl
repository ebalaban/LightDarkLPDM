using Discretizers
import LPDM: default_action, next_actions, isterminal
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
    init_dist::Any
    discount::Float64
    count::Int
    n_rand::Int
    resample_std::Float64
    max_actions::Int64
    base_action_space::Vector{LD1Action}
    nominal_action_space::Vector{LD1Action}
    extended_action_space::Vector{LD1Action}

    function LightDark1DLpdm()
        this = new()
        this.min_noise               = 0.0
        this.min_noise_loc           = 5.0
        this.Q                       = 0.5
        this.R                       = 0.5
        this.term_radius             = 0.05
        this.n_bins                  = 100 # per linear dimension
        this.max_x                   = 10     # assume symmetry in x and y for simplicity
        this.bin_edges               = collect(-this.max_x:(2*this.max_x)/this.n_bins:this.max_x)
        this.bin_centers             = [(this.bin_edges[i]+this.bin_edges[i+1])/2 for i=1:this.n_bins]
        this.lindisc                 = LinearDiscretizer(this.bin_edges)
        this.discount                = 1.0
        this.count                   = 0
        this.n_rand                  = 0
        this.resample_std            = 0.5 # st. deviation for particle resampling
        this.max_actions             = 10
        this.base_action_space       = [1.0 0.1 0.01]
        this.nominal_action_space    = vcat(-base_action_space, [0.0], base_action_space)
        this.extended_action_space   = vcat(2*nominal_action_space, 3*nominal_action_space)
        return this
    end
end

# POMDPs.actions(p::LightDark1DLpdm) = vcat(-POMDPs.actions(p, true), [0.0], POMDPs.actions(p,true))
LPDM.default_action(p::LightDark1DLpdm) = 0.00
POMDPs.rand(p::LightDark1DLpdm, s::LD1State, rng::LPDM.RNGVector) = norminvcdf(s, p.resample_std, rand(rng)) # for resampling

# Replaces the default call
function LPDM.isterminal(pomdp::LightDark1DLpdm, particles::Vector{LPDMParticle{LD1State}})
    expected_state = 0.0
    for p in particles
        expected_state += p.state*p.weight # NOTE: assume weights are normalized
    end
    return isterminal(pomdp,expected_state)
end

# Version with discrete observations
function generate_o(p::LightDark1DLpdm, sp::Float64, rng::AbstractRNG)
    o = rand(rng, observation(p, sp))
    o_disc = p.bin_centers[encode(p.lindisc,o)]
    # println("$o -> $o_disc")
    return o_disc
    # return obs_index(p,o_disc) # return a single combined obs index
end

POMDPs.actions(pomdp::LightDark1DLpdm) = pomdp.extended_action_space
LPDM.max_actions(pomdp::LightDark1DLpdm) = pomdp.max_actions

# Implement a simple hard-coded version for now for debugging
function LPDM.next_actions(pomdp::LightDark1DLpdm, current_space::Vector{LD1Action})::Vector{LD1Action}

    if isempty(current_actions) # initial request
        return nominal_space
    else
        n = length(current_space) - length(nominal_space) + 1
        return [extended_space[n]] # return as a 1-element vector
    end
end
