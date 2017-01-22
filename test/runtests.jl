using LightDarkPOMDPs
using Base.Test
using POMDPs
using POMDPToolbox
using StaticArrays
using Plots

p = LightDark2D(0.0, 5.0, diagm([0.5, 0.5]), diagm([0.5, 0.5]))
pol = FunctionPolicy(s -> rand(2) - SVector(0.5,0.5))

sim = HistoryRecorder(max_steps=20)

h = simulate(sim, p, pol)

plot(p, h)

gui()
