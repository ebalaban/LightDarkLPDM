# __precompile__()

module LightDarkPOMDPs

importall POMDPs

using StaticArrays
using Combinatorics
# using Plots
using POMDPToolbox
using Parameters # for @with_kw
using ParticleFilters # for AbstractParticleBelief

include("lightdark2d.jl")
include("lightdark2dtarget.jl")
include("lightdark2dfilter.jl")
# include("lightdark2dvis.jl")
export 
    AbstractLD2,
    LightDark2D,
    LightDark2DTarget,
    LightDark2DKalman,
    SymmetricNormal2,
    Vec2
end # module
