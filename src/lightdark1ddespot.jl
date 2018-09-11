using Discretizers

@with_kw type LightDark1DDespot <: AbstractLD1
    min_noise::Float64
    min_noise_loc::Float64
    term_radius::Float64
    n_bins::Int         # per linear dimension
    max_x::Float64     # assume symmetry in x and y for simplicity
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    lindisc::LinearDiscretizer
    init_dist::Any
    discount::Float64
    count::Int
    n_rand::Int


    function LightDark1DDespot(n_bins = 10)
        this = new()
        this.min_noise               = 0.0
        this.min_noise_loc           = 5.0
        this.term_radius             = 0.05
        this.n_bins                  = n_bins # per linear dimension
        this.max_x                   = 10     # assume symmetry in x and y for simplicity
        this.bin_edges               = collect(-this.max_x:(2*this.max_x)/this.n_bins:this.max_x)
        this.bin_centers             = [(this.bin_edges[i]+this.bin_edges[i+1])/2 for i=1:n_bins]
        this.lindisc                 = LinearDiscretizer(this.bin_edges)
        this.init_dist               = Distributions.Normal(2.0, 0.5)
        this.discount                = 1.0
        this.count                   = 0
        this.n_rand                  = 0
        println(this.bin_edges)
        println(this.bin_centers)
        return this
    end
end
POMDPs.actions(p::LightDark1DDespot) = [1.0, 0.5, 0.1, 0.01, 0.0];


# POMDPs.actions(p::LightDark1DDespot, ::Bool) = [0.1, 0.01]
#POMDPs.actions(p::LightDark1DDespot) = Float64Iter(collect(permutations(vcat(POMDPs.actions(p, true), -POMDPs.actions(p,true)), 2)))

# reward(p::LightDark1DDespot, s::Float64) = -1.0
# reward(p::LightDark1DDespot, s::Float64, a::Float64) = reward(p, s)
# reward(p::LightDark1DDespot, s::Float64, a::Float64, sp::Float64) = reward(p, s)

# Version with discrete observations
function generate_o(p::LightDark1DDespot, sp::Float64, rng::AbstractRNG)
    o = rand(rng, observation(p, sp))
    o_disc = p.bin_centers[encode(p.lindisc,o)]
    # println("$o -> $o_disc")
    return o_disc
    # return obs_index(p,o_disc) # return a single combined obs index
end