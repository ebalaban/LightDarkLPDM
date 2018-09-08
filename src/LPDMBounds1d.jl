import LPDM: bounds

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
                     pomdp::AbstractLD1,
                     particles::Vector{LPDMParticle{LDState}},
                     config::LPDMConfig)

    ub = Array{Int8}(0);
    lb = Array{Int8}(0);

    for s in particles
        push!(ub, upperBound(pomdp, s))
        push!(lb, lowerBound(pomdp, s))
    end

    b.lb = minimum(lb)
    b.ub = maximum(ub)

    return b.lb, b.ub
end

function lowerBound(p::LightDark1DDespot, particle::POMDPToolbox.Particle{Float64})
# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach

    s = particle.state
    actions = POMDPs.actions(p)
    r::Int8 = 0.0;
    remx::Float64 = 0.0;
    remy::Float64 = 0.0;
    first_a::Float64 = -1.0 #TODO: consider removing

    r1,remx,first_a = take_action(p.min_noise_loc-s, p.term_radius, actions)         # calculate cost for moving x to low noise region
    # r2,remy,first_a = take_action(s[2], p.term_radius, actions)                         # calculate cost for moving y to target coordinate
    r3,remx,first_a = take_action(p.min_noise_loc-remx, p.term_radius, actions)         # calculate cost for moving x to target coordinate from where it reached in the low noise region

    return -(r1 + r3)
end
lowerBound(p::LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = lowerBound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))


function upperBound(p::LightDark1DDespot, particle::POMDPToolbox.Particle{Float64})
    # computes the reward for the straight-line path to target
    s = particle.state
    actions = POMDPs.actions(p)
    rx::Int8 = 0

    remx::Float64 = 0

    rx,remx,first_a = take_action(s, p.term_radius, actions)        ##  for x: cost to move x in a straight line to within target region
    # r = rx > ry ? rx : ry                                  ## pick the larger of the two
    return -rx
end
upperBound(p::LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = upperBound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))


# function take_action(x::Float64, terminal::Float64, actions::Array{Float64,1})
#     r::Int8 = 0;
#     x = abs(x);
#     if x > terminal
#         r = floor(x);               # largest available step is 1
#         remx = x - r;
#         if remx > terminal
#             for a in actions
#                 while remx > a && remx > terminal
#                     remx -= a;
#                     r += 1
#                 end
#             end
#         end
#     end
#     return r
# end

function take_action(x::Float64, terminal::Float64, actions::Array{Float64,1})
    r::Int8 = 0;
    x = abs(x);
    first_a = -1.0
    if x > terminal && terminal > minimum(actions)
        for a in actions
            if first_a < 0
                first_a = a
            end
            if x > a && x > terminal
                steps = floor(x/a)
                x -= steps*a;
                r += steps
            end
        end
    elseif terminal < minimum(actions)
        warn("""
            The set of available actions will never allow x to arrive within the target radius
                    x = $x
                    terminal radius = $terminal
                    actions = $actions
            """)
    end
    return r, x, first_a
end
