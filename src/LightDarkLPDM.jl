# __precompile__()

module LightDarkLPDM

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

include("1D/lightdark1d.jl")
include("2D/lightdark2d.jl")
# include("lightdark2dtarget.jl")
# include("lightdark2dfilter.jl")
# include("lightdark1ddespot.jl")
include("1D/lightdark1dlpdm.jl")
# include("lightdark2ddespot.jl")
include("2D/lightdark2dlpdm.jl")

# include("lightdark2dvis.jl")
export
    AbstractLD1,
    AbstractLD2,
    LightDark1D,
    LightDark2D,
    LightDark2DTarget,
    # LightDark1DDespot,
    LightDark1DLpdm,
    # LightDark2DDespot,
    LightDark2DLpdm,
    LightDark2DKalman,
    Normal1D,
    SymmetricNormal2D,
    Vec2,
    obs_std,
    rand,
    state_distribution,
    generate_sor_det,
    LD1State,
    LD1Action,
    LD1Obs,
    LD2State,
    LD2Action,
    LD2Obs
end # module
