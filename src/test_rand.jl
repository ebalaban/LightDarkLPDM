using Revise
using LPDM
using Distributions
using POMDPs
import POMDPs.pdf

const LD1Obs    = Float64

d = Distributions.Normal(10.0,2.0)
rng = RNGVector(1,UInt32(42))
a = rand(rng,d)

POMDPs.pdf(d::Distributions.Normal, o::LD1Obs) = Distributions.pdf(d,o)
POMDPs.pdf(d,11.0)
