# __precompile__()

module LightDarkPOMDPs

# importall POMDPs
using POMDPs
using StaticArrays
using Combinatorics
using Distributions
# using Plots
# using POMDPToolbox
using Parameters # for @with_kw
using Printf
using Random
# using ParticleFilters # for AbstractParticleBelief
using LPDM

include("lightdark1d.jl")
include("lightdark2d.jl")
include("lightdark2dtarget.jl")
include("lightdark2dfilter.jl")
include("lightdark1ddespot.jl")
include("lightdark1dlpdm.jl")
include("lightdark2ddespot.jl")

# include("lightdark2dvis.jl")
export
    AbstractLD1,
    AbstractLD2,
    LightDark1D,
    LightDark2D,
    LightDark2DTarget,
    LightDark1DDespot,
    LightDark1DLpdm,
    LightDark2DDespot,
    LightDark2DKalman,
    SymmetricNormal2,
    Vec2,
    obs_std,
    rand,
    state_distribution,
    LD1State,
    LD1Action,
    LD1Obs
end # module
