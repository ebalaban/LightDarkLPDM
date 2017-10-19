
const Vec2 = SVector{2,Float64}
Vec2() = Vec2(0, 0)

abstract type AbstractLD2 <: POMDP{Vec2, Vec2, Vec2} end

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
@with_kw type LightDark2D <: AbstractLD2
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
generate_s(p::AbstractLD2, s::Vec2, a::Vec2, rng::AbstractRNG) = generate_s(p, s, a)
generate_o(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2, rng::AbstractRNG) = generate_o(p, sp, rng)
observation(p::AbstractLD2, a::Vec2, sp::Vec2) = observation(p, sp)
isterminal(p::AbstractLD2, s::Vec2) = norm(s) <= p.term_radius
initial_state_distribution(p::AbstractLD2) = p.init_dist
reward(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2) = -(dot(s, p.Q*s) + dot(a, p.R*a))
reward(p::AbstractLD2, s::Vec2, a::Vec2)           = -(dot(s, p.Q*s) + dot(a, p.R*a))
discount(p::AbstractLD2) = p.discount


immutable SymmetricNormal2
    mean::Vec2
    std::Float64
end
# rand(rng::AbstractRNG, d::SymmetricNormal2) = d.mean + d.std*Vec2(randn(rng, 2))
pdf(d::SymmetricNormal2, s::Vec2) = exp(-0.5*sum((s-d.mean).^2)/d.std^2)/(2*pi*d.std^2)
mean(d::SymmetricNormal2) = d.mean
mode(d::SymmetricNormal2) = d.mean
Base.eltype(::Type{SymmetricNormal2}) = Vec2

# chose this on 2/6/17 because I like the bowtie particle patterns it produces
# unclear which one was actually used in the paper
# Masters Thesis by Pas assumes the sqrt version
obs_std(p::AbstractLD2, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)
# obs_std(p::AbstractLD2, x::Float64) = 0.5*(p.min_noise_loc-x)^2 + p.min_noise

function generate_s(p::AbstractLD2, s::Vec2, a::Vec2)
    p.count += 1
    return s+a
end
observation(p::AbstractLD2, sp::Vec2) = SymmetricNormal2(sp, obs_std(p, sp[1]))
generate_o(p::AbstractLD2, sp::Vec2, rng::AbstractRNG) = rand(rng, observation(p, sp))

function POMDPs.generate_sor(p::AbstractLD2, s::Vec2, a::Vec2, rng::AbstractRNG)

    s = generate_s(p,s,a,rng)
    o = generate_o(p,s,rng)
    r = reward(p,s,a)

    return s, o, r
end

### Shouldn't need this:
# generate_o(p::AbstractLD2, sp::Vec2, rng::LPDM.RNGVector) = generate_o(p, sp, rng)
