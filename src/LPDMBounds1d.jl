import LPDM: bounds
using LightDarkPOMDPs

mutable struct LDBounds1d{A}
    lb    ::Float64
    ub    ::Float64
    best_action::A

    function LDBounds1d{A}() where A
        this = new()
        this.lb = +Inf
        this.ub = -Inf
        this.best_action = 0.0 #TODO: let's try doing nothing as default
        return this
    end
end

function LPDM.bounds(b::LDBounds1d,
                     pomdp::LightDarkPOMDPs.AbstractLD1,
                     particles::Vector{LPDMParticle{LDState}},
                     config::LPDMConfig)

    ub = Array{Float64}(0);
    lb = Array{Float64}(0);

    for s in particles
        push!(ub, upperBound(pomdp, s))
        push!(lb, lowerBound(pomdp, s))
    end

    b.lb = minimum(lb)
    b.ub = maximum(ub)

    return b.lb, b.ub
end

# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach
function lowerBound(p::LightDarkPOMDPs.LightDark1DDespot, particle::POMDPToolbox.Particle{Float64})

    r1 = move_reward(p, particle.state, p.min_noise_loc)         # estimate cost of moving x to low noise x
    r2 = move_reward(p, p.min_noise_loc, p.term_radius)         # estimate cost of moving x to target region from the low noise region

    return r1+r2
end
lowerBound(p::LightDarkPOMDPs.LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = lowerBound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))

# computes the reward for the straight-line path to target
upperBound(p::LightDarkPOMDPs.LightDark1DDespot, particle::POMDPToolbox.Particle{Float64}) = move_reward(p, particle.state, p.term_radius)

upperBound(p::LightDarkPOMDPs.LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = upperBound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))

function move_reward(p::LightDarkPOMDPs.AbstractLD1, x1::Float64, x2::Float64)
    #direction = x2 > x1 ? 1 : -1
    actions = Base.findnz(POMDPs.actions(p,true)')[3] # get only positive non-zero actions
    min_a = minimum(actions)
    # compute just with positive case - the two cases are symmetrical
    orig = minimum(abs.([x1,x2]))
    dest = maximum(abs.([x1,x2]))

    r = 0.0
    Δx = 0.0
    x = orig

    while dest-x > min_a
        Δx = dest - x
        a = maximum(actions[actions .<= Δx]) # maximum action not exceeding Δx
        r += reward(p, x, a) # use current state for computing the reward (NOTE: assumes reward symmetry)
        println("BOUNDS: x=$x, Δx=$(Δx), x1=$x1, x2=$x2, orig=$orig, dest=$dest, actions=$(actions[actions .< Δx]), a=$a,  r=$r ")
        x += a # take the step
        # error("done")
    end
    return r
end


# NOTE: Not needed
# function take_action(x::Float64, terminal::Float64, actions::Array{Float64,1})
#     r::Int8 = 0;
#     x = abs(x);
#     first_a = -1.0
#     if x > terminal && terminal > minimum(actions)
#         for a in actions
#             if first_a < 0
#                 first_a = a
#             end
#             if x > a && x > terminal
#                 steps = floor(x/a)
#                 x -= steps*a;
#                 r += steps
#             end
#         end
#     elseif terminal < minimum(actions)
#         warn("""
#             The set of available actions will never allow x to arrive within the target radius
#                     x = $x
#                     terminal radius = $terminal
#                     actions = $actions
#             """)
#     end
#     return r, x, first_a
# end
