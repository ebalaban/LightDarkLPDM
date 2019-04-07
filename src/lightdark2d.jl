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

# Base.show(io::IO, x::Vec2) = print(io,"[$(@sprintf("%.2f", x[1])),$(@sprintf("%.2f",x[2]))]")
Base.show(io::IO, x::Vec2) = print(io,"[$(@sprintf("%.4f", x[1])),$(@sprintf("%.4f",x[2]))]")

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
    Q::Matrix{Float64}      = diagm(0=>[0.5, 0.5])
    R::Matrix{Float64}      = diagm(0=>[0.5, 0.5])
    term_radius::Float64    = 0.05
    init_dist::Any          = SymmetricNormal2([2.0, 2.0], 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
end

# POMDPs.jl API functions:
POMDPs.generate_s(p::AbstractLD2, s::Vec2, a::Vec2, rng::AbstractRNG) = generate_s(p, s, a)
POMDPs.generate_o(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2, rng::AbstractRNG) = generate_o(p, sp, rng)
POMDPs.observation(p::AbstractLD2, a::Vec2, sp::Vec2) = observation(p, sp)
# POMDPs.isterminal(p::AbstractLD2, s::Vec2) = LinearAlgebra.norm(s) <= p.term_radius
function POMDPs.isterminal(p::AbstractLD2, s::Vec2) #DEBUG
    #NOTE: yes, this is not strictly accurate, but makes things easier
    if abs(s[1]) < p.term_radius && abs(s[2]) < p.term_radius
        return true
    else
        return false
    end
end

# Replaces the default call
function LPDM.isterminal(pomdp::AbstractLD2, particles::Vector{LPDMParticle{LD2State}})
    expected_state = Vec2(0.0,0.0)
    weight_sum = 0.0
    # LPDM.normalize!(particles)
    for p in particles
        expected_state += p.state*p.weight
        weight_sum += p.weight
    end
    expected_state /= weight_sum
    # println("expected state: $expected_state")
    return isterminal(pomdp,expected_state)
end


POMDPs.initial_state_distribution(p::AbstractLD2) = p.init_dist

function POMDPs.reward(p::AbstractLD2, s::Vec2, a::Vec2)
    if isterminal(p,s)
        r = 0.0
    else
        r = (p.reward_mode == :quadratic) ?  -(dot(s, p.Q*s) + dot(a, p.R*a)) : -1
    end
    return r
end

POMDPs.reward(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2) = POMDPs.reward(p, s, a)
POMDPs.discount(p::AbstractLD2) = p.discount

LPDM.default_action(p::AbstractLD2) = Vec2(0.0,0.0)
LPDM.default_action(p::AbstractLD2, ::Vector{LPDMParticle{Vec2}}) = LPDM.default_action(p)

struct SymmetricNormal2D
    mean::Vec2
    std::Float64
end

# Compose action spaces via permutations of available moves
function permute(moves::Vector{Float64})::Vector{LD2Action}
    actions = Vector{Vec2}(undef,0)
    all_moves = vcat(-moves, [0.0], moves)
    for i in 1:length(all_moves)
        for j in 1:length(all_moves)
            if all_moves[i] != 0.0 || all_moves[j] != 0.0 # don't include 'idle'
                push!(actions,Vec2(all_moves[i], all_moves[j]))
            end
        end
    end
    return actions
end

# Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{SymmetricNormal2}) = d[].mean + d[].std*Vec2(rand(rng)-0.5,rand(rng)-0.5)
# Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{SymmetricNormal2}) = Vec2(rand(rng)-0.5,rand(rng)-0.5)
# POMDPs.pdf(d::SymmetricNormal2, s::Vec2) = exp(-0.5*sum((s-d.mean).^2)/d.std^2)/(2*pi*d.std^2)
POMDPs.pdf(d::SymmetricNormal2D, o::LD2Obs) = Distributions.pdf(Distributions.MvNormal([d.mean[1],d.mean[2]],[d.std,d.std]), o)
mean(d::SymmetricNormal2D) = d.mean
mode(d::SymmetricNormal2D) = d.mean
Base.eltype(::Type{SymmetricNormal2D}) = Vec2
POMDPs.rand(p::AbstractLD2, s::LD2State, rng::LPDM.RNGVector) = rand(rng, SymmetricNormal2D(s,p.resample_std)) # for resampling

# chose this on 2/6/17 because I like the bowtie particle patterns it produces
# unclear which one was actually used in the paper
# Masters Thesis by Pas assumes the sqrt version
obs_std(p::AbstractLD2, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)
# obs_std(p::AbstractLD2, x::Float64) = 0.5*(p.min_noise_loc-x)^2 + p.min_noise

function generate_s(p::AbstractLD2, s::Vec2, a::Vec2)
    p.count += 1
    return s+a
end
POMDPs.observation(p::AbstractLD2, sp::Vec2) = SymmetricNormal2D(sp, obs_std(p, sp[1]))
POMDPs.observation(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2) = observation(p,sp)

# generate_o(p::AbstractLD2, sp::Vec2, rng::AbstractRNG) = rand(rng, observation(p, sp))
function generate_o(p::AbstractLD2, sp::Vec2, rng::AbstractRNG)
    d = observation(p, sp)
    println("sp: $sp")
    println("d: $d")
    o = rand(rng, Distributions.MvNormal([d.mean[1],d.mean[2]],[d.std,d.std]))
    if p.obs_mode == :discrete
        o_disc = Vec2(p.bin_centers[encode(p.lindisc,o[1])],
                      p.bin_centers[encode(p.lindisc,o[2])])
        return o_disc
    elseif p.obs_mode == :continuous
        println("o: $o")
        return Vec2(o)
    else
        error("Invalid obs_mode = $(p.obs_mode)")
    end
    # return obs_index(p,o_disc) # return a single combined obs index
end

# Initial distribution corresponds to how the observations would look
#TODO: possibly move to LightDarkPOMDPs
function state_distribution(pomdp::AbstractLD2, s0::LD2State, config::LPDMConfig, rng::RNGVector)
    states = Vector{LPDMParticle{LD2State}}();
    weight = 1/(config.n_particles^2) # weight of each individual particle
    particle = LPDMParticle{LD2State}(LD2State(0.0,0.0), 1, weight)

    d = observation(pomdp,s0)
    println("state distribution d: $d")
    for i = 1:config.n_particles^2 #TODO: Inefficient, possibly improve. Maybe too many particles
        s = Vec2(rand(rng, Distributions.MvNormal([d.mean[1],d.mean[2]],[d.std,d.std])))
        particle = LPDMParticle{LD2State}(s,
                                          i,
                                          weight)
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
