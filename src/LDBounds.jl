immutable LDBounds end

POMDPs.actions(p::AbstractLD2) = [1, 0.5, 0.1, 0.01];

function LPDM.bounds{S,A,O}(b::LDBounds,
    pomdp::AbstractLD2,
    particles::Vector{LPDMParticle{S}},
    config::LPDMConfig)


    ub = Array{Int8}(0);
    lb = Array{Int8}(0);
    for s in particles
        push!(ub, upperBound(pomdp, s))
        push!(lb, lowerBound(pomdp, s))
    end

    return maximum(ub), minimum(lb)
end

function lowerBound(p::LightDark2DTarget, s::LDState)
# computes the cost of traveling to the low noise region and then towards target
   #       s__________
   #                  |
   #                  |
   #           x______|
    actions = POMDPs.actions(p);
    x::Float64, y::Float64 = s[1], s[2];
    r::Int8 = 0;
    ##  Move x to low noise region
    r += take_action(p.min_noise_loc-x, p.term_radius, actions)         # calculate cost for moving x to low noise region
    ##  Move y to within range of target coordinate
    r += take_action(y, p.term_radius, actions)                         # calculate cost for moving y to target coordinate
    ## Move x to within range of target coordinate
    r += take_action(p.min_noise_loc, p.term_radius, actions)           # calculate cost for moving x to target coordinate

    return -r
end


function upperBound(p::LightDark2DTarget, s::LDState)
    # computes the reward for the straight-line path to target
    actions = POMDPs.actions(p);
    x::Float64, y::Float64 = s[1], s[2];
    rx::Int8 = 0;
    ry::Int8 = 0;
    ##  for x
    rx = take_action(x, p.term_radius, actions);
    ## for y
    ry = take_action(y, p.term_radius, actions);

    r = rx > ry ? rx : ry;
    return -r
end

function take_action(x::Float64, terminal::Float64, actions::Array{Float64,1})
    r::Int8 = 0;
    x = abs(x);
    if x > terminal
        r = floor(x);               # largest available step is 1
        remx = x - r;
        if remx > terminal
            for a in actions
                while remx > a && remx > terminal
                    remx -= a;
                    r += 1
                end
            end
        end
    end
    return r
end



function state_distribution(p::AbstractLD2, s::Vec2, config::LPDMConfig)
    randx = randn(config.n_particles);
    randy = randn(config.n_particles);
    B = Array{Vec2}(0);

    for rx in randx, ry in randy
        push!(B, Vec2(s[1] + rx, s[2] + ry))
    end
    return B
end
