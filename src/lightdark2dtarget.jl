
@with_kw type LightDark2DTarget <: AbstractLD2
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    term_radius::Float64    = 0.05
    init_dist::Any          = SymmetricNormal2([2.0, 2.0], 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
end

reward(p::LightDark2DTarget, s::Vec2) = -1.0
reward(p::LightDark2DTarget, s::Vec2, a::Vec2) = reward(p, s)
reward(p::LightDark2DTarget, s::Vec2, a::Vec2, sp::Vec2) = reward(p, s)
