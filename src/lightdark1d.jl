import Base.show, Base.rand
import POMDPs:
        isterminal,
        pdf,
        observation,
        generate_s,
        generate_o,
        initial_state_distribution,
        discount

import LPDM.isterminal
using StatsFuns

Base.show(io::IO, x::Float64) = print(io,"$(@sprintf("%.2f", x))")

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
@with_kw mutable struct LightDark1D <: LightDarkPOMDPs.AbstractLD1
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    Q::Float64              = 0.5
    R::Float64              = 0.5
    term_radius::Float64    = 1e-5
    init_dist::Any          = Normal(2.0, 0.5)
    #init_dist::Any          = Normal(-3.0, 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
end

# POMDPs.jl API functions:
POMDPs.generate_s(p::AbstractLD1, s::Float64, a::Float64, rng::AbstractRNG) = generate_s(p, s, a)
POMDPs.generate_o(p::AbstractLD1, s::Float64, a::Float64, sp::Float64, rng::AbstractRNG) = generate_o(p, sp, rng)
POMDPs.observation(p::AbstractLD1, a::Float64, sp::Float64) = observation(p, sp)
POMDPs.initial_state_distribution(p::AbstractLD1) = p.init_dist
POMDPs.reward(p::AbstractLD1, s::Float64, a::Float64, sp::Float64) = -(p.Q*s^2 + p.R*a^2)
POMDPs.reward(p::AbstractLD1, s::Float64, a::Float64)              = -(p.Q*s^2 + p.R*a^2)
POMDPs.discount(p::AbstractLD1) = p.discount

POMDPs.isterminal(p::AbstractLD1, s::Float64) = (abs(s) <= p.term_radius)
# function POMDPs.isterminal(p::AbstractLD1, s::Float64) #DEBUG
#     println("_______ISTERMINAL(s=$s)=$(abs(s) <= p.term_radius)_______")
#     return abs(s) <= p.term_radius
# end

# Compute the expected state and determine whether it is terminal
# function LPDM.isterminal(pomdp::AbstractLD1, particles::Vector{LPDMParticle{Float64}})
#     wt_sum = 0.0
#     exp_s  = 0.0 # expected state
#     for p in particles
#         exp_s += p.state*p.weight
#         wt_sum += p.weight
#     end
#     # println("ISTERMINAL($(POMDPs.isterminal(pomdp, exp_s/wt_sum))): E(s)=$(exp_s/wt_sum). Particles:")
#     # show(particles); println("")
#     # println("THIS BLOODY ISTERMINAL: $(@which(POMDPs.isterminal(pomdp, exp_s/wt_sum)))")
#     return POMDPs.isterminal(pomdp, exp_s/wt_sum)
# end

struct Normal
    mean::Float64
    std::Float64
end

Base.rand(p::LightDarkPOMDPs.AbstractLD1, s::Float64, rng::LPDM.RNGVector) = rand(rng, Normal(s, std(p.init_dist)))

function Base.rand(rng::LPDM.RNGVector,
                   d::Normal)

    # error("IN HERE!!!")
    # a random number selected from normal distribution
    r1 = norminvcdf(d.mean, d.std, rand(rng))
    return r1
end

# rand(rng::AbstractRNG, d::Normal) = d.mean + d.std*Float64(randn(rng, 2))
# POMDPs.pdf(d::Normal, o::Float64) = exp(-0.5*sum((o-d.mean).^2)/d.std^2)/(2*pi*d.std^2); println("observing!")
function POMDPs.pdf(d::Normal, o::Float64) #DEBUG version
    # error("In my pdf")
    return exp(-0.5*sum((o-d.mean).^2)/d.std^2)/(2*pi*d.std^2)
end

mean(d::Normal) = d.mean
mode(d::Normal) = d.mean
Base.eltype(::Type{Normal}) = Float64

# chose this on 2/6/17 because I like the bowtie particle patterns it produces
# unclear which one was actually used in the paper
# Masters Thesis by Pas assumes the sqrt version
# obs_std(p::AbstractLD1, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)

# obs_std(p::AbstractLD1, x::Float64) = 0.5*(p.min_noise_loc-x)^2 + p.min_noise

# EB 10/20/18, implementing the version from Platt et al:
obs_std(p::AbstractLD1, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)


function generate_s(p::AbstractLD1, s::Float64, a::Float64)
    p.count += 1
    return s+a
end

POMDPs.observation(p::AbstractLD1, sp::Float64) = Normal(sp,obs_std(p,sp))
POMDPs.observation(p::AbstractLD1, s::Float64, a::Float64, sp::Float64) = observation(p,sp)

generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG) = rand(rng, observation(p, sp))
# generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG) = sp #DEBUG: trying no noise at all

# function generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG)
#     println("sp=$sp")
#     return sp #DEBUG: trying no noise at all
# end

function POMDPs.generate_sor(p::AbstractLD1, s::Float64, a::Float64, rng::AbstractRNG)

    s = generate_s(p,s,a,rng)
    o = generate_o(p,s,rng)
    r = reward(p,s,a)
    # println("o! $o")
    return s, o, r
end

### Shouldn't need this:
# generate_o(p::AbstractLD1, sp::Float64, rng::LPDM.RNGVector) = generate_o(p, sp, rng)
