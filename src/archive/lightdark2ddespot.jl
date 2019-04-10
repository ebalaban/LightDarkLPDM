using Discretizers
import LPDM: default_action

# @with_kw mutable struct LightDark2DDespot <: AbstractLD2
mutable struct LightDark2DDespot <: AbstractLD2
    min_noise::Float64
    min_noise_loc::Float64
    term_radius::Float64
    Q::Matrix{Float64}
    R::Matrix{Float64}
    n_bins::Int         # per linear dimension
    max_xy::Float64     # assume symmetry in x and y for simplicity
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    lindisc::LinearDiscretizer
    init_dist::Any
    discount::Float64
    count::Int
    n_rand::Int
    resample_std::Float64
    action_space_type::Symbol
    nominal_moves::Vector{Float64}
    extended_moves::Vector{Float64}
    nominal_action_space::Vector{LD2Action}
    extended_action_space::Vector{LD2Action}
    reward_func::Symbol

    function LightDark2DDespot(action_space_type::Symbol; reward_func = :quadratic)
        this = new()
        this.min_noise               = 0.0
        this.min_noise_loc           = 5.0
        this.term_radius             = 0.05
        # this.term_radius             = 0.1
        this.Q                       = diagm(0=>[0.5, 0.5])
        this.R                       = diagm(0=>[0.5, 0.5])
        this.n_bins                  = 100 # per linear dimension
        this.max_xy                  = 10     # assume symmetry in x and y for simplicity
        this.bin_edges               = collect(-this.max_xy:(2*this.max_xy)/this.n_bins:this.max_xy)
        this.bin_centers             = [(this.bin_edges[i]+this.bin_edges[i+1])/2 for i=1:this.n_bins]
        this.lindisc                 = LinearDiscretizer(this.bin_edges)
        this.init_dist               = SymmetricNormal2([2.0, 2.0], 0.5)
        this.discount                = 0.9
        this.count                   = 0
        this.n_rand                  = 0
        this.resample_std            = 0.5 # st. deviation for particle resampling
        # this.nominal_action_space    = [1.0, 0.1, 0.01]
        this.nominal_moves    = [1.0, 0.1, 0.01]
        this.extended_moves   = vcat(1*this.nominal_moves,
                                    # 2*this.nominal_action_space,
                                    # 3*this.nominal_action_space,
                                    # 4*this.nominal_action_space,
                                     5*this.nominal_moves)
        this.nominal_action_space  = permute(this.nominal_moves)
        this.extended_action_space = permute(this.extended_moves)
        this.action_space_type       = action_space_type
        this.reward_func                  = reward_func
        return this
    end
end

function POMDPs.actions(p::LightDark2DDespot, ::Bool)
    if p.action_space_type == :small
        # return vcat(-p.nominal_action_space, [0.0], p.nominal_action_space)
        return p.nominal_moves
    elseif p.action_space_type == :large
        # return vcat(-p.extended_action_space, [0.0], p.extended_action_space)
        return p.extended_moves
    else
        error("Action space $(p.action_space_type) is not valid for POMDP of type $(typeof(p))")
    end
end

POMDPs.actions(p::LightDark2DDespot) = p.action_space_type == :small ? p.nominal_action_space : p.extended_action_space

# Version with discrete observations
function generate_o(p::LightDark2DDespot, sp::Vec2, rng::AbstractRNG)
    # o_disc = Vec2()
    o = rand(rng, observation(p, sp))
    o_disc = Vec2(p.bin_centers[encode(p.lindisc,o[1])],
                  p.bin_centers[encode(p.lindisc,o[2])])
    # println("$o -> $o_disc")
    return o_disc
    # return obs_index(p,o_disc) # return a single combined obs index
end

#TODO: may not be needed
obs_index(p::LightDark2DDespot, o_disc::Vec2) = p.n_bins*(o_disc[1]-1) + o_disc[2]
