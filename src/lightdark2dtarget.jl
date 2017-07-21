
@with_kw type LightDark2DTarget <: AbstractLD2
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    term_radius::Float64    = 0.05
    init_dist::Any          = SymmetricNormal2([2.0, 2.0], 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
    n_rand::Int = 0
end

reward(p::LightDark2DTarget, s::Vec2) = -1.0
reward(p::LightDark2DTarget, s::Vec2, a::Vec2) = reward(p, s)
reward(p::LightDark2DTarget, s::Vec2, a::Vec2, sp::Vec2) = reward(p, s)


# actions() with an additional boolean argument returns the available steps
# actions() without a boolean returns all combinations of steps in x and y
POMDPs.actions(p::AbstractLD2, ::Bool) = [1, 0.5, 0.1, 0.01];
POMDPs.actions(p::AbstractLD2) = Vec2Iter(vcat(collect(permutations(POMDPs.actions(p, true), 2)), 
                                              collect(permutations(-POMDPs.actions(p, true), 2))
                                              )
                                          )
Vec2Iter(S::Array{Array{Float64,1},1}) = [Vec2(s) for s in S]