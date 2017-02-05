using LightDarkPOMDPs
using Base.Test
using POMDPs
using POMDPToolbox
using StaticArrays
using Plots

using Gallium
# Gallium.breakpoint(Pkg.dir("POMDPToolbox","src","simulators","history_recorder.jl"), 109)

p = LightDark2D()

# slow_feedback(o::SVector) = -0.1*o
# slow_feedback(b::SymmetricNormal2) = -0.1*b.mean

pol = FunctionPolicy(o -> -0.01*o)
 
hr = HistoryRecorder(max_steps=100)

h = sim(p, simulator=hr) do o
    @show o
    action(pol, o)
end

# plot(p, h)
bpol = FunctionPolicy(b -> -0.5*b.mean)

up = LightDark2DKalman(p)

h = simulate(hr, p, bpol, up)

# plot(p, h, plot_belief=true, plot_observations=true)

#=
plot(p)
up = SIRParticleUpdater(p, 100)
b = initialize_belief(up, initial_state_distribution(p))
plot!(b)
=#

gui()
