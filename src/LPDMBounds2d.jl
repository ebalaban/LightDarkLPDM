import LPDM: bounds, best_lb_action, best_ub_action

mutable struct LDBounds2d{S,A,O}
    lb_    ::Float64
    ub_    ::Float64
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

    # reset every time bounds are about to be recomputed
    b.lb_ = +Inf
    b.ub_ = -Inf
    b.best_lb_action_ = Vec2(NaN,NaN)
    b.best_ub_action_ = Vec2(NaN,NaN)

    tmp_lb = 0.0
    tmp_ub = 0.0
    tmp_lb_action = Vec2(0.0,0.0)
    tmp_ub_action = Vec2(0.0,0.0)

    for p in particles
        tmp_ub, tmp_ub_action = upper_bound(pomdp,p)
        if p.state[1] > pomdp.min_noise_loc # assume min_noise_loc > 0
            # both will have to do roughly the same thing (+/- discretization differences),
            # so make them the same
            #TODO: review for lb > ub
            tmp_lb, tmp_lb_action = tmp_ub, tmp_ub_action
        else
            tmp_lb, tmp_lb_action = lower_bound(pomdp,p)
        end

        if tmp_lb < b.lb_
            b.lb_             = tmp_lb
            b.best_lb_action_ = tmp_lb_action
        end
        if tmp_ub > b.ub_
            b.ub_             = tmp_ub
            b.best_ub_action_ = tmp_ub_action
        end
    end

    # Sanity check
    if b.ub_ < b.lb_
        show(particles); println("")
        error("BOUNDS: ub=$(b.ub_) < lb=$(b.lb_), lba = $(b.best_lb_action_), uba = $(b.best_ub_action_)")
    end
    return b.lb_, b.ub_
end



LPDM.best_lb_action(b::LDBounds2d) = isnan(b.best_lb_action_) ? error("best_lb_action undefined ($(b.best_lb_action_)). Call bounds() first.") : b.best_lb_action_
# LPDM.best_ub_action(b::LDBounds2d) = isnan(b.best_ub_action_) ? error("best_ub_action undefined. Call bounds() first.") : b.best_ub_action_
LPDM.best_ub_action(b::LDBounds2d) = isnan(b.best_ub_action_) ? error("best_ub_action undefined ($(b.best_ub_action_)). Call bounds() first.") : b.best_ub_action_


#TODO: these functions make sense for step-wise rewards, but not as much for quadratic rewards. May need to redo.

# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach
   #       s__________
   #                  |
   #                  |
   #           (0,0) _|

function lower_bound(p::LightDark2DDespot, particle::LPDMParticle{Vec2})

    s = particle.state

    r1, a1_x = coarse_move(p, s, Vec2(p.min_noise_loc,s[2]), 1)
    r2, a1_y = coarse_move(p, Vec2(p.min_noise_loc,s[2]), Vec2(p.min_noise_loc,0.0), 2)
    r3, a2_x = coarse_move(p, Vec2(p.min_noise_loc,0.0), Vec2(0.0,0.0), 1)

    return r1 + r2 + r3, Vec2(a1_x,a1_y)
end
# lowerBound(p::LightDark2DDespot, particle::LPDM.LPDMParticle{Vec2}) = lowerBound(p, POMDPToolbox.Particle{Vec2}(particle.state, particle.weight))
#
# function lower_bound(p::LightDark2DDespot, particle::LPDMParticle{Vec2})
#
#     s = particle.state
#     actions = POMDPs.actions(p, true);
#     r::Int8 = 0.0;
#     remx::Float64 = 0.0;
#     remy::Float64 = 0.0;
#
#     r1,remx,first_ax = take_action(p.min_noise_loc-s[1], p.term_radius, actions)        # calculate cost for moving x to low noise region
#     r2,remy,first_ay = take_action(s[2], p.term_radius, actions)                         # calculate cost for moving y to target coordinate
#     r3,remx,temp_a = take_action(p.min_noise_loc-remx, p.term_radius, actions)         # calculate cost for moving x to target coordinate from where it reached in the low noise region
#
#     return -(r1 + r2 + r3), Vec2(first_ax,first_ay)
# end

# computes the reward for the straight-line path to target
function upper_bound(p::LightDark2DDespot, particle::LPDMParticle{Vec2})
    s = particle.state

    rx, a1_x = coarse_move(p, s, Vec2(0.0,s[2]), 1)
    ry, a1_y = coarse_move(p, Vec2(0.0,s[2]), Vec2(0.0,0.0), 2)

    # println("s=$s, rx=$rx, ry=$ry, a1_x=$a1_x, a1_y=$a1_y")
    # error("done, for now")
    r = rx < ry ? rx : ry                                  ## pick the smaller (worst) of the two
    return r, Vec2(a1_x,a1_y)
end


#w1 and w2 are start and end waypoints, coord is 1 or 2 for x and y, respectively
# NOTE: this function computes a rough estimate to avoid discretization issues
function coarse_move(p::AbstractLD2, w1::Vec2, w2::Vec2, c::Int64)
    direction = w2[c] > w1[c] ? 1 : -1
    all_actions = POMDPs.actions(p, true)
    pos_actions = all_actions[all_actions .> 0]

    r = 0.0
    w = [w1[1],w1[2]]
    a = NaN
    a_dir = NaN

    a = maximum(pos_actions[pos_actions .< abs(w2[c]-w[c])+p.term_radius]) # maximum action not exceeding Δx + p.term_radius
    # n_moves = round(Int64,(abs(w2[c]-w[c])+p.term_radius)/a
    n_moves = ceil(Int64,(abs(w2[c]-w[c])+p.term_radius)/a)
    a_dir = direction * a

    for i in 1:n_moves
        r += reward(p, Vec2(w[1],w[2]), Vec2(a_dir,0.0)) # use current state for computing the reward; (0.0,a_dir) and (a_dir,0.0) are equivalent for reward purposes
        w[c] += a_dir # take the step
    end

    if (abs(w2[1]) <= p.term_radius) && (abs(w2[2]) <= p.term_radius)
        r += reward(p,w2,Vec2(0.0,0.0)) #termination reward
    end
    # println("move: w1=$w1, w2=$w2, n_moves = $n_moves, a_dir = $a_dir, r=$r")
    return r, a_dir #action sign depends on the direction
end

 #w1 and w2 are start and end waypoints, coord is 1 or 2 for x and y, respectively
function move(p::AbstractLD2, w1::Vec2, w2::Vec2, c::Int64)
    direction = w2[c] > w1[c] ? 1 : -1
    all_actions = POMDPs.actions(p, true)
    # I = findall(!iszero, all_actions)
    pos_actions = all_actions[all_actions .> 0]
    # nz_actions = all_actions[I]
    min_a = minimum(pos_actions)

    r = 0.0
    w = [w1[1],w1[2]]
    first_a = NaN
    a_dir = NaN

    # if (abs(w1[1]) <= p.term_radius) && (abs(w1[2]) <= p.term_radius)
    #     return reward(p,w1,Vec2(0.0,0.0)), 0.0
    # end

    if (abs(w1[c]) <= p.term_radius)
        return reward(p,w1,Vec2(0.0,0.0)), 0.0
    end

    if abs(w2[c]-w1[c])+p.term_radius < min_a #too close to take any action
        return reward(p,w1,Vec2(0.0,0.0)), 0.0
    end

    while (abs(w2[c]-w[c])+p.term_radius > min_a) && (abs(w2[c]-w[c]) > p.term_radius)
        a = maximum(pos_actions[pos_actions .< abs(w2[c]-w[c])+p.term_radius]) # maximum action not exceeding Δx + p.term_radius
        a_dir = direction * a
        if isnan(first_a)
            first_a = a_dir # assign first action (directional)
        end
        r += reward(p, Vec2(w[1],w[2]), Vec2(a_dir,0.0)) # use current state for computing the reward; (0.0,a_dir) and (a_dir,0.0) are equivalent for reward purposes
        # println("BOUNDS: w=$w, w1=$w1, w2=$w2, min_a = $min_a, all_actions=$all_actions, pos_actions=$pos_actions, av_actions=$(all_actions[all_actions .< abs(w2[c]-w[c]) + p.term_radius]), a_dir=$a_dir,  r=$r ")
        w[c] += a_dir # take the step
        println("    w=($(w[1]),$(w[2])), a_dir=$a_dir, r = $r")
    end

    if (abs(w2[1]) <= p.term_radius) && (abs(w2[2]) <= p.term_radius)
        # x1 < 0.7 && println("TERMINATION REWARD #2 FOR x2=$x")
        r += reward(p,w2,Vec2(0.0,0.0)) #termination reward
        # first_a = 0.0
    end

    # x1 < 0.7 && println("exiting move $x1 -> $x2")
    # if isnan(first_a) #DEBUG: remove
    #     error("move: first_a = $first_a, w1=$w1, w2=$w2")
    # end
    println("move: w1=$w1, w2=$w2, first_a = $first_a, r=$r")
    # error("done, for now")
    return r, first_a #action sign depends on the direction
end

# NOTE: not for direct calling
# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach
function lower_bound(p::AbstractLD1, particle::LPDMParticle{Float64})

    if abs(particle.state) < p.term_radius
        return reward(p, particle.state, 0.0), 0.0
    else
        r1,a1 = move(p, particle.state, p.min_noise_loc)         # estimate cost of moving x to low noise x
        # estimate cost of moving x to the target from the low noise region (will terminate earlier)
        r2,a2 = move(p, p.min_noise_loc, 0.0)
    end
    return r1+r2, a1
end


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
