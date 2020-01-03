using LightDarkLPDM
using Base.Test
using POMDPs
using POMDPToolbox
using StaticArrays
using Plots
using ParticleFilters

#### LightDark2D Problem ####

p = LightDark2D()

# simulate with a policy that moves -0.01 times the observation
hr = HistoryRecorder(max_steps=10)
h = sim(p, simulator=hr) do o
    @show o
    return -0.01*o
end
@show p.count


# simulate with a kalman filter
bpol = FunctionPolicy(b -> -0.5*mean(b))
up = LightDark2DKalman(p)
h = simulate(hr, p, bpol, up)


pl = plot(p);
plot!(pl, h)


#### LightDark2DTarget Problem ####

p = LightDark2DTarget()

pl = plot(p);
up = SIRParticleFilter(p, 1000)
b = initialize_belief(up, initial_state_distribution(p))
plot!(pl, b)
gui()

hist = simulate(hr, p, bpol, up)
println("state trajectory")
for s in state_hist(hist)
    @show s
end
