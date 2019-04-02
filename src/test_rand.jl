using Revise
using LPDM
using Distributions

d = Distributions.Normal(10.0,2.0)
rng = RNGVector(1,UInt32(42))
a = rand(rng,d)
