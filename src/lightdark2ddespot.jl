using Discretizers

@with_kw type LightDark2DDespot <: AbstractLD2
    min_noise::Float64      = 0.0
    min_noise_loc::Float64  = 5.0
    term_radius::Float64    = 0.05
    init_dist::Any          = SymmetricNormal2([2.0, 2.0], 0.5)
    discount::Float64       = 1.0
    count::Int              = 0
    n_rand::Int = 0
    n_bins::Int = 5 # per linear dimension
    max_xy = 10     # assume symmetry in x and y for simplicity
    bin_edges = [-max_xy:(2*max_xy)/n_bins:max_xy]
    lindisc = LinearDiscretizer(bin_edges)
end

# Version with discrete observations
function generate_o(p::LightDark2DDespot, sp::Vec2, rng::AbstractRNG)
    o_disc = (0,0)
    o = rand(rng, observation(p, sp))
    o_disc[1] = encode(p.lindisc,o[1])
    o_disc[2] = encode(p.lindisc,o[2])
    return obs_index(p,o_disc) # return a single combined obs index
end

obs_index(p::LightDark2DDespot, o_disc::Vec2) = p.n_bins*o_disc[1] + o_disc[2]
