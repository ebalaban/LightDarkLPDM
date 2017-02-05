
typealias Vec2 SVector{2,Float64}

abstract AbstractLD2 <: POMDP{Vec2, Vec2, Vec2}

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
"""
@with_kw type LightDark2D <: AbstractLD2
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    Q::Matrix{Float64}      = diagm([0.5, 0.5])
    R::Matrix{Float64}      = diagm([0.5, 0.5])
    term_radius::Float64    = 1e-5
    init_dist::Any          = SymmetricNormal2([2.0, 2.0], 0.5)
    discount::Float64       = 1.0
end

generate_s(::AbstractLD2, s::Vec2, a::Vec2) = s + a
generate_s(p::AbstractLD2, s::Vec2, a::Vec2, rng::AbstractRNG) = generate_s(p, s, a)

# changed on 2/2/17
obs_std(p::AbstractLD2, x::Float64) = sqrt(0.5*(p.min_noise_loc-x)^2 + p.min_noise)
# obs_std(p::AbstractLD2, x::Float64) = 0.5*(p.min_noise_loc-x)^2 + p.min_noise

generate_o(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2, rng::AbstractRNG) = generate_o(p, sp, rng)
generate_o(p::AbstractLD2, sp::Vec2, rng::AbstractRNG) = rand(rng, observation(p, sp))

observation(p::AbstractLD2, a::Vec2, sp::Vec2) = observation(p, sp)
observation(p::AbstractLD2, sp::Vec2) = SymmetricNormal2(sp, obs_std(p, sp[1]))

reward(p::AbstractLD2, s::Vec2, a::Vec2, sp::Vec2) = -(dot(s, p.Q*s) + dot(a, p.R*a))
discount(p::AbstractLD2) = p.discount

immutable SymmetricNormal2
    mean::Vec2
    std::Float64
end
rand(rng::AbstractRNG, d::SymmetricNormal2) = d.mean + d.std*Vec2(randn(rng, 2))
pdf(d::SymmetricNormal2, s::Vec2) = exp(-0.5*sum((s-d.mean).^2)/d.std^2)/(2*pi*d.std^2)
mean(d::SymmetricNormal2) = d.mean
mode(d::SymmetricNormal2) = d.mean
Base.eltype(::Type{SymmetricNormal2}) = Vec2

initial_state_distribution(p::AbstractLD2) = p.init_dist

isterminal(p::AbstractLD2, s::Vec2) = norm(s) <= p.term_radius

