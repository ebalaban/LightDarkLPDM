using Discretizers

@with_kw type LightDark2DDespot <: AbstractLD2
    min_noise::Float64
    min_noise_loc::Float64
    term_radius::Float64
    n_bins::Int         # per linear dimension
    max_xy::Float64     # assume symmetry in x and y for simplicity
    bin_edges::Vector{Float64}
    lindisc::LinearDiscretizer
    init_dist::Any
    discount::Float64
    count::Int
    n_rand::Int


    function LightDark2DDespot(n_bins = 5)
        this = new()
        this.min_noise               = 0.0
        this.min_noise_loc           = 5.0
        this.term_radius             = 0.05
        this.n_bins                  = n_bins # per linear dimension
        this.max_xy                  = 10     # assume symmetry in x and y for simplicity
        this.bin_edges               = collect(-this.max_xy:(2*this.max_xy)/this.n_bins:this.max_xy)
        this.lindisc                 = LinearDiscretizer(this.bin_edges)
        this.init_dist               = SymmetricNormal2([2.0, 2.0], 0.5)
        this.discount                = 1.0
        this.count                   = 0
        this.n_rand                  = 0

        return this
    end
end
# POMDPs.actions(p::LightDark2DDespot, ::Bool) = [1.0, 0.5, 0.1, 0.01, 0.0];
POMDPs.actions(p::LightDark2DDespot, ::Bool) = [0.1, 0.01]

reward(p::LightDark2DDespot, s::Vec2) = -1.0
reward(p::LightDark2DDespot, s::Vec2, a::Vec2) = reward(p, s)
reward(p::LightDark2DDespot, s::Vec2, a::Vec2, sp::Vec2) = reward(p, s)

# Version with discrete observations
function generate_o(p::LightDark2DDespot, sp::Vec2, rng::AbstractRNG)
    o_disc = Vec2()
    o = rand(rng, observation(p, sp))
    # show(o); println("")
    o_disc = Vec2(encode(p.lindisc,o[1]), encode(p.lindisc,o[2]))
    return o_disc
    # return obs_index(p,o_disc) # return a single combined obs index
end

#TODO: may not be needed
obs_index(p::LightDark2DDespot, o_disc::Vec2) = p.n_bins*(o_disc[1]-1) + o_disc[2]
