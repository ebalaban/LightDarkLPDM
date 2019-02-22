import Base.show, Base.rand
import POMDPs:
        isterminal,
        pdf,
        observation,
        generate_s,
        generate_o,
        initial_state_distribution,
        discount,
        reward,
        rand
using Random
import Random: rand
import LPDM.isterminal
using StatsFuns
using LinearAlgebra

const Vec2 = SVector{2,Float64}
Vec2() = Vec2(0.0,0.0)
Vec2Iter(S::Array{Array{Float64,1},1}) = [Vec2(s) for s in S]

# Typealias appropriately
const LD2State  = Vec2
const LD2Action = Vec2
const LD2Obs    = Vec2
const LD2Belief = LPDMBelief

Base.show(io::IO, x::Vec2) = print(io,"[$(@sprintf("%.2f", x[1])),$(@sprintf("%.2f",x[2]))]")

abstract type AbstractLD2 <: POMDP{LD2State, LD2Action, LD2Obs} end

"""
2-Dimensional light-dark problem

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
@with_kw mutable struct LightDark2D <: AbstractLD2
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    Q::Matrix{Float64}      = diagm([0.5, 0.5])
    R::Matrix{Float64}      = diagm([0.5, 0.5])
    term_radius::Float64    = 1e-5
    init_dist::Any          = SymmetricNormal2([2.0, 2.0], 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
end

# POMDPs.jl API functions:
POMDPs.generate_s(p::AbstractLD2, s::Vec2, a::Vec2, rng::AbstractRNG) = generate_s(p, s, a)
POMDPs.generate_o(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2, rng::AbstractRNG) = generate_o(p, sp, rng)
POMDPs.observation(p::AbstractLD2, a::Vec2, sp::Vec2) = observation(p, sp)
POMDPs.isterminal(p::AbstractLD2, s::Vec2) = LinearAlgebra.norm(s) <= p.term_radius
POMDPs.initial_state_distribution(p::AbstractLD2) = p.init_dist
POMDPs.reward(p::AbstractLD2, s::Vec2, a::Vec2) =
                (p.reward_func == :quadratic) ?  -(dot(s, p.Q*s) + dot(a, p.R*a)) : -1
POMDPs.reward(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2) = POMDPs.reward(p, s, a)

POMDPs.discount(p::AbstractLD2) = p.discount

struct SymmetricNormal2
    mean::Vec2
    std::Float64
end

Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{SymmetricNormal2}) = d[].mean + d[].std*Vec2(rand(rng)-0.5,rand(rng)-0.5)
POMDPs.pdf(d::SymmetricNormal2, s::Vec2) = exp(-0.5*sum((s-d.mean).^2)/d.std^2)/(2*pi*d.std^2)
mean(d::SymmetricNormal2) = d.mean
mode(d::SymmetricNormal2) = d.mean
Base.eltype(::Type{SymmetricNormal2}) = Vec2
POMDPs.rand(p::AbstractLD2, s::LD2State, rng::LPDM.RNGVector) = rand(rng, SymmetricNormal2(s,p.resample_std)) # for resampling

# chose this on 2/6/17 because I like the bowtie particle patterns it produces
# unclear which one was actually used in the paper
# Masters Thesis by Pas assumes the sqrt version
obs_std(p::AbstractLD2, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)
# obs_std(p::AbstractLD2, x::Float64) = 0.5*(p.min_noise_loc-x)^2 + p.min_noise

function generate_s(p::AbstractLD2, s::Vec2, a::Vec2)
    p.count += 1
    return s+a
end
POMDPs.observation(p::AbstractLD2, sp::Vec2) = SymmetricNormal2(sp, obs_std(p, sp[1]))
POMDPs.observation(p::AbstractLD1, s::Vec2, a::Vec2, sp::Vec2) = observation(p,sp)

generate_o(p::AbstractLD2, sp::Vec2, rng::AbstractRNG) = rand(rng, observation(p, sp))

# Initial distribution corresponds to how the observations would look
#TODO: possibly move to LightDarkPOMDPs
function state_distribution(pomdp::AbstractLD2, s0::LD2State, config::LPDMConfig, rng::RNGVector)
    states = Vector{LPDMParticle{LD2State}}();
    weight = 1/(config.n_particles^2) # weight of each individual particle
    particle = LPDMParticle{LD2State}(LD2State(0.0,0.0), 1, weight)

    for i = 1:config.n_particles^2 #TODO: Inefficient, possibly improve. Maybe too many particles
        particle = LPDMParticle{LD2State}(rand(rng, observation(pomdp,s0)), i, weight)
        push!(states, particle)
    end
    return states
end

function POMDPs.generate_sor(p::AbstractLD2, s::Vec2, a::Vec2, rng::AbstractRNG)

    s = generate_s(p,s,a,rng)
    o = generate_o(p,s,rng)
    r = reward(p,s,a)

    return s, o, r
end
