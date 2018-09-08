import Base.show

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
@with_kw type LightDark1D <: AbstractLD1
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    Q::Float64              = 0.5
    R::Float64              = 0.5
    term_radius::Float64    = 1e-5
    init_dist::Any          = Normal(2.0, 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
end

# POMDPs.jl API functions:
generate_s(p::AbstractLD1, s::Float64, a::Float64, rng::AbstractRNG) = generate_s(p, s, a)
generate_o(p::AbstractLD1, s::Float64, a::Float64, sp::Float64, rng::AbstractRNG) = generate_o(p, sp, rng)
observation(p::AbstractLD1, a::Float64, sp::Float64) = observation(p, sp)
isterminal(p::AbstractLD1, s::Float64) = norm(s) <= p.term_radius
initial_state_distribution(p::AbstractLD1) = p.init_dist
reward(p::AbstractLD1, s::Float64, a::Float64, sp::Float64) = -(p.Q*s^2 + p.R*a^2)
reward(p::AbstractLD1, s::Float64, a::Float64)              = -(p.Q*s^2 + p.R*a^2)
discount(p::AbstractLD1) = p.discount


immutable Normal
    mean::Float64
    std::Float64
end
# rand(rng::AbstractRNG, d::Normal) = d.mean + d.std*Float64(randn(rng, 2))
pdf(d::Normal, s::Float64) = exp(-0.5*sum((s-d.mean).^2)/d.std^2)/(2*pi*d.std^2)
mean(d::Normal) = d.mean
mode(d::Normal) = d.mean
Base.eltype(::Type{Normal}) = Float64

# chose this on 2/6/17 because I like the bowtie particle patterns it produces
# unclear which one was actually used in the paper
# Masters Thesis by Pas assumes the sqrt version
obs_std(p::AbstractLD1, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)
# obs_std(p::AbstractLD1, x::Float64) = 0.5*(p.min_noise_loc-x)^2 + p.min_noise

function generate_s(p::AbstractLD1, s::Float64, a::Float64)
    p.count += 1
    return s+a
end
observation(p::AbstractLD1, sp::Float64) = Normal(sp, obs_std(p, sp))
generate_o(p::AbstractLD1, sp::Float64, rng::AbstractRNG) = rand(rng, observation(p, sp))

function POMDPs.generate_sor(p::AbstractLD1, s::Float64, a::Float64, rng::AbstractRNG)

    s = generate_s(p,s,a,rng)
    o = generate_o(p,s,rng)
    r = reward(p,s,a)

    return s, o, r
end

### Shouldn't need this:
# generate_o(p::AbstractLD1, sp::Float64, rng::LPDM.RNGVector) = generate_o(p, sp, rng)
