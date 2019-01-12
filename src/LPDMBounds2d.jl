import LPDM: bounds, best_lb_action, best_ub_action

mutable struct LDBounds2d{S,A,O}
    lb    ::Float64
    ub    ::Float64
    best_lb_action_::Vec2
    best_ub_action_::Vec2

    function LDBounds2d{S,A,O}(::POMDP{S,A,O}) where {S,A,O}
        this = new{S,A,O}()
        this.lb_ = +Inf
        this.ub_ = -Inf
        this.best_lb_action_ = Vec2(NaN,NaN)
        this.best_ub_action_ = Vec2(NaN,NaN)
        return this
    end
end

function LPDM.bounds(b::LDBounds2d{S,A,O},
                     pomdp::AbstractLD2,
                     particles::Vector{LPDMParticle{S}},
                     config::LPDMConfig) where {S,A,O}

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

#TODO: these functions make sense for step-wise rewards, but not as much for quadratic rewards. May need to redo.
function lowerBound(p::LightDark2DDespot, particle::LPDMParticle{Vec2})
# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach
   #       s__________
   #                  |
   #                  |
   #           x______|

    s = particle.state
    actions = POMDPs.actions(p, true);
    r::Int8 = 0.0;
    remx::Float64 = 0.0;
    remy::Float64 = 0.0;
    first_a::Float64 = -1.0 #TODO: consider removing

    r1,remx,first_a = take_action(p.min_noise_loc-s[1], p.term_radius, actions)         # calculate cost for moving x to low noise region
    r2,remy,first_a = take_action(s[2], p.term_radius, actions)                         # calculate cost for moving y to target coordinate
    r3,remx,first_a = take_action(p.min_noise_loc-remx, p.term_radius, actions)         # calculate cost for moving x to target coordinate from where it reached in the low noise region

    return -(r1 + r2 + r3)
end
lowerBound(p::LightDark2DDespot, particle::LPDM.LPDMParticle{Vec2}) = lowerBound(p, POMDPToolbox.Particle{Vec2}(particle.state, particle.weight))


function upperBound(p::LightDark2DDespot, particle::LPDMParticle{Vec2})
    # computes the reward for the straight-line path to target
    s = particle.state
    actions = POMDPs.actions(p, true);
    rx::Int8 = 0
    ry::Int8 = 0
    remx::Float64 = 0
    remy::Float64 = 0
    rx,remx,first_a = take_action(s[1], p.term_radius, actions)        ##  for x: cost to move x in a straight line to within target region
    ry,remy,first_a = take_action(s[2], p.term_radius, actions)        ##  same for y

    r = rx > ry ? rx : ry                                  ## pick the larger of the two
    return -r
end
upperBound(p::LightDark2DDespot, particle::LPDM.LPDMParticle{Vec2}) = upperBound(p, POMDPToolbox.Particle{Vec2}(particle.state, particle.weight))


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
