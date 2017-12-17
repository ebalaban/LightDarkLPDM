import LPDM: bounds, default_action

type LDBounds
    a_star::Tuple{Float64,Float64}

    function LDBounds()
        this = new()
        this.a_star = (0.0,0.0)
        return this
    end
end

function LPDM.bounds(b::LDBounds,
                     pomdp::AbstractLD2,
                     particles::Vector{LPDMParticle{LDState}},
                     config::LPDMConfig)

    ub = Vector{Int8}(0)
    lb = Vector{Int8}(0)
    a_star = Vector{Tuple{Float64,Float64}}()
    ub_value = Int8(0)
    lb_value = Int8(0)
    a_star_value::Tuple{Float64,Float64}=(0.0,0.0)
    a_star_void::Tuple{Float64,Float64}=(0.0,0.0)

    for s in particles
        ub_value, a_star_value = upperBound(pomdp, s)
        lb_value, a_star_void = lowerBound(pomdp, s)
        push!(ub, ub_value)
        push!(a_star, a_star_value)
        push!(lb, lb_value)
    end
    b.a_star = a_star[indmax(ub)]

    return minimum(lb), maximum(ub)
end

function lowerBound(p::LightDark2DTarget, particle::POMDPToolbox.Particle{Vec2})
# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach
   #       s__________
   #                  |
   #                  |
   #           x______|

    s = particle.state
    actions = POMDPs.actions(p, true);
    # r::Int8 = 0;
    remx::Float64 = 0;
    remy::Float64 = 0;
    a_star_x::Float64 = 0.0;
    a_star_y::Float64 = 0.0;
    a_void::Float64 = 0.0;

    r1,remx, a_star_x = take_action(p.min_noise_loc-s[1], p.term_radius, actions)         # calculate cost for moving x to low noise region
    r2,remy, a_star_y = take_action(s[2], p.term_radius, actions)                         # calculate cost for moving y to target coordinate
    r3,remx, a_void = take_action(p.min_noise_loc-remx, p.term_radius, actions)         # calculate cost for moving x to target coordinate from where it reached in the low noise region

    return -(r1 + r2 + r3), (a_star_x, a_star_y)
end
lowerBound(p::LightDark2DTarget, particle::LPDM.LPDMParticle{Vec2}) = lowerBound(p, POMDPToolbox.Particle{Vec2}(particle.state, particle.weight))


function upperBound(p::LightDark2DTarget, particle::POMDPToolbox.Particle{Vec2})
    # computes the reward for the straight-line path to target
    s = particle.state
    actions = POMDPs.actions(p, true);
    rx::Int8 = 0
    ry::Int8 = 0
    remx::Float64 = 0
    remy::Float64 = 0
    a_star_x::Float64 = 0
    a_star_y::Float64 = 0

    rx, remx, a_star_x = take_action(s[1], p.term_radius, actions)        ##  for x: cost to move x in a straight line to within target region
    ry, remy, a_star_y = take_action(s[2], p.term_radius, actions)        ##  same for y

    r = rx > ry ? rx : ry                                  ## pick the larger of the two
    return -r, (a_star_x, a_star_y)
end
upperBound(p::LightDark2DTarget, particle::LPDM.LPDMParticle{Vec2}) = upperBound(p, POMDPToolbox.Particle{Vec2}(particle.state, particle.weight))


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
    a_star::Float64 = 0.0

    if x > terminal && terminal > minimum(actions)
        for a in actions
            if x > a && x > terminal
                steps = floor(x/a)
                x -= steps*a;
                r += steps
                a_star == 0.0 && (a_star = a) # record the first action chosen (to be used as default)
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
    return r, x, a_star
end

LPDM.default_action(b::LDBounds,
                     ::LightDark2DTarget,
                     ::Vector{LPDMParticle{LDState}},
                     ::LPDMConfig)::Tuple{Float64,Float64} = b.a_star
