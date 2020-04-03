import Base.show, Base.rand
import POMDPs:
        isterminal,
        pdf,
        observation,
        generate_s,
        generate_o,
        initialstate_distribution,
        discount,
        reward

import LPDM: isterminal, max_belief_clusters
using StatsFuns
using Random

# Typealias appropriately
const LD1State  = Float64
const LD1Action = Float64
const LD1Obs    = Float64
const LD1Belief = LPDMBelief

Base.show(io::IO, x::Float64) = print(io,"$(@sprintf("%.4f", x))")

abstract type AbstractLD1 <: POMDP{Float64, Float64, Float64} end

"""
1-Dimensional light-dark problem

    min_noise::Float64
        minimum standard deviation of observation noise

    min_noise_loc::Float64
        x location of minimum observation noise

    Q::Matrix{Float64}
    R::Matrix{Float64}
        cost function matrices: reward = -s'Qs -a'Ra

    term_radius::Float64
        problem will terminate if the state is within this radius of zero

    init_dist::Any
        initial state distribution

    discount::Float64
        discount factor
"""
@with_kw mutable struct LightDark1D <: LightDarkLPDM.AbstractLD1
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    Q::Float64              = 0.5
    R::Float64              = 0.5
    term_radius::Float64    = 1e-5
    # init_dist::Any          = Normal(2.0, 0.5)
    #init_dist::Any          = Normal(-3.0, 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
end

# POMDPs.jl API functions:
# POMDPs.generate_s(p::AbstractLD1, s::Float64, a::Float64, rng::AbstractRNG) = generate_s(p, s, a, rng, det = false)
POMDPs.generate_s(p::AbstractLD1, s::Float64, a::Float64, rng::AbstractRNG) = generate_s(p, s, a, rng, false)
POMDPs.generate_o(p::AbstractLD1, s::Float64, a::Float64, sp::Float64, rng::AbstractRNG) = generate_o(p, sp, rng)
POMDPs.observation(p::AbstractLD1, sp::Float64) = Normal1D(sp,obs_std(p,sp))
POMDPs.observation(p::AbstractLD1, a::Float64, sp::Float64) = observation(p, sp)
POMDPs.observation(p::AbstractLD1, s::Float64, a::Float64, sp::Float64) = observation(p,sp)

function generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG)
    d = observation(p, sp)
    o = rand(rng, Distributions.Normal(d.mean, d.std))
    if p.obs_mode == :discrete
        o_disc = p.bin_centers[encode(p.lindisc,o)]
        return o_disc
    elseif p.obs_mode == :continuous
    #     o=clamp(o, -p.max_x, p.max_x) #TODO: EB, 02/11/2020: disable clamping for now, see if it stops clustering outliers into wrong clusters
        return o
    else
        error("Invalid obs_mode = $(p.obs_mode)")
    end
    # return obs_index(p,o_disc) # return a single combined obs index
end


# generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG) = rand(rng, observation(p, sp))

# POMDPs.reward(p::AbstractLD1, s::Float64, a::Float64)              = -1.0

POMDPs.reward(p::AbstractLD1, s::Float64, a::Float64) =
                        (p.reward_mode == :quadratic) ? -(p.Q*s^2 + p.R*a^2) : -1

POMDPs.reward(p::AbstractLD1, s::Float64, a::Float64, sp::Float64) = POMDPs.reward(p,s,a)
POMDPs.discount(p::AbstractLD1) = p.discount

# Replaces the default call
function LPDM.isterminal(pomdp::AbstractLD1, particles::Vector{LPDMParticle{LD1State}})
    expected_state = 0.0
    weight_sum = 0.0
    # LPDM.normalize!(particles)
    for p in particles
        expected_state += p.state*p.weight
        weight_sum += p.weight
    end
    expected_state /= weight_sum
    return isterminal(pomdp,expected_state)
end

POMDPs.isterminal(p::AbstractLD1, s::Float64) = (abs(s) <= p.term_radius)

struct Normal1D
    mean::Float64
    std::Float64
end

# Base.rand(p::AbstractLD1, s::Float64, rng::LPDM.RNGVector) = rand(rng, Normal(s, std(p.init_dist)))

# function Base.rand(rng::LPDM.RNGVector,
#                    d::Normal)
#
#     # error("IN HERE!!!")
#     # a random number selected from normal distribution
#     r1 = norminvcdf(mean(d), std(d), rand(rng))
#     return r1
# end

# rand(rng::AbstractRNG, d::Normal) = d.mean + d.std*Float64(randn(rng, 2))
# POMDPs.pdf(d::Normal, o::Float64) = exp(-0.5*sum((o-d.mean).^2)/d.std^2)/(2*pi*d.std^2); println("observing!")
# function POMDPs.pdf(d::Normal, o::Float64) #DEBUG version
#     # error("In my pdf")
#     return exp(-0.5*sum((o-d.mean).^2)/d.std^2)/(2*pi*d.std^2)
# end
#
# mean(d::Normal) = d.mean
# mode(d::Normal) = d.mean
# Base.eltype(::Type{Normal}) = Float64

# chose this on 2/6/17 because I like the bowtie particle patterns it produces
# unclear which one was actually used in the paper
# Masters Thesis by Pas assumes the sqrt version
# obs_std(p::AbstractLD1, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)

# obs_std(p::AbstractLD1, x::Float64) = 0.5*(p.min_noise_loc-x)^2 + p.min_noise

# EB 10/20/18, implementing the version from Platt et al:
obs_std(p::AbstractLD1, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)

# NOTE: the whole point of defining Normal1D is to avoid the so-called type piracy when defining this POMDPs.pdf method
POMDPs.pdf(d::Normal1D, o::Float64) = Distributions.pdf(Distributions.Normal(d.mean, d.std),o)

# function POMDPs.pdf(d::Normal1D, o::Float64)
#     # pdf=Distributions.pdf(d,o)
#     println("In pdf: d=$d, o=$o")
#     println("pdf=$(Distributions.pdf(Distributions.Normal(d.mean, d.std),o))")
#     return nothing
# end

# function generate_s(p::AbstractLD1, s::Float64, a::Float64, rng::RNGVector; det::Bool = false)
function generate_s(p::AbstractLD1, s::Float64, a::Float64, rng::RNGVector, det::Bool = false)
    p.count += 1
    if p.action_std == 0.0 || det
        return s + a
    else
        return s + LPDM.rand(rng, Distributions.Normal(a, p.action_std))
    end
end

# function generate_s_det(p::AbstractLD1, s::Float64, a::Float64)
#     p.count += 1
#     return s+a
# end

# generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG) = sp #DEBUG: trying no noise at all

# function generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG)
#     println("sp=$sp")
#     return sp #DEBUG: trying no noise at all
# end

# Initial distribution corresponds to how the observations would look
function state_distribution(pomdp::AbstractLD1, s0::LD1State, config::LPDMConfig, rng::RNGVector)
    states = Vector{LPDMParticle{Float64}}();
    weight = 1/(config.n_particles^2) # weight of each individual particle
    particle = LPDMParticle{LD1State}(0.0, 1, weight)
    d = observation(pomdp,s0)
    for i = 1:config.n_particles^2 #TODO: Inefficient, possibly improve. Maybe too many particles
        particle = LPDMParticle{Float64}(LPDM.rand(rng, Distributions.Normal(d.mean, d.std)), i, weight)
        push!(states, particle)
    end
    # println("n states: $(length(states))")
    return states
end

function POMDPs.generate_sor(p::AbstractLD1, s::Float64, a::Float64, rng::RNGVector, det::Bool = false)
# function POMDPs.generate_sor(p::AbstractLD1, s::Float64, a::Float64, rng::RNGVector; det::Bool = false)

    s = generate_s(p, s, a, rng, det)
    o = generate_o(p, s, rng)
    r = reward(p, s, a)
    # println("o! $o")
    return s, o, r
end
