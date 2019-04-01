import LPDM: bounds, best_lb_action, best_ub_action

mutable struct LDBounds1d{S,A,O}
    lb_    ::Float64
    ub_    ::Float64
    best_lb_action_::Float64
    best_ub_action_::Float64

    function LDBounds1d{S,A,O}(::POMDP{S,A,O}) where {S,A,O}
        this = new()
        this.lb_ = +Inf
        this.ub_ = -Inf
        this.best_lb_action_ = NaN
        this.best_ub_action_ = NaN
        return this
    end
end

function LPDM.bounds(b::LDBounds1d{S,A,O},
                     pomdp::AbstractLD1,
                     particles::Vector{LPDMParticle{S}},
                     config::LPDMConfig) where {S,A,O}

    # reset every time bounds are about to be recomputed
    b.lb_ = +Inf
    b.ub_ = -Inf
    b.best_lb_action_ = NaN
    b.best_ub_action_ = NaN

    tmp_lb = NaN
    tmp_ub = NaN
    tmp_lb_action = NaN
    tmp_ub_action = NaN

    for p in particles
        tmp_ub, tmp_ub_action = upper_bound(pomdp,p)
        if p.state >= pomdp.min_noise_loc # assume min_noise_loc > 0
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
            # if abs(tmp_lb + 30.0) < 0.1 # DEBUG: remove
            #     println("particles: $particles")
            #     println("s: $(p.state), b.lb_=$(b.lb_), tmp_lb_action = $(tmp_lb_action), b.lba_ = $(b.best_lb_action_)")
            # end
        end
        if tmp_ub > b.ub_
            b.ub_             = tmp_ub
            b.best_ub_action_ = tmp_ub_action
        end
    end

    # Sanity check
    if b.ub_ < b.lb_
        # show(particles); println("")
        # error("BOUNDS: ub=$(b.ub_) < lb=$(b.lb_), lba = $(b.best_lb_action_), uba = $(b.best_ub_action_), tmp_lb = $(tmp_lb), tmp_ub = $(tmp_ub), tmp_lb_action = $(tmp_lb_action), tmp_ub_action = $(tmp_ub_action)")
        error("BOUNDS: ub=$(b.ub_) < lb=$(b.lb_)")
    end
    return b.lb_, b.ub_
end

LPDM.best_lb_action(b::LDBounds1d) = isnan(b.best_lb_action_) ? error("best_lb_action undefined. Call bounds() first") : b.best_lb_action_
LPDM.best_ub_action(b::LDBounds1d) = isnan(b.best_ub_action_) ? error("best_ub_action undefined. Call bounds() first") : b.best_ub_action_

function move(p::AbstractLD1, x1::Float64, x2::Float64)#::(Float64,Float64)
    direction = x2 > x1 ? 1 : -1
    all_actions = POMDPs.actions(p)
    # I = findall(!iszero, all_actions)
    pos_actions = all_actions[all_actions .> 0]
    # nz_actions = all_actions[I]
    min_a = minimum(pos_actions)

    r = 0.0
    x = x1
    first_a = NaN
    a_dir = NaN

    if abs(x1) <= p.term_radius
        return reward(p,x1,0.0), 0.0
    end

    if abs(x2-x1) < min_a #too close to take any action
        return reward(p,x1,0.0), 0.0
    end

    while (abs(x2-x) > min_a) && (abs(x) >= p.term_radius)
        a = maximum(pos_actions[pos_actions .<= abs(x2-x)]) # maximum action not exceeding Δx
        a_dir = direction * a
        if isnan(first_a)
            first_a = a_dir # assign first action (directional)
        end
        r += reward(p, x, a_dir) # use current state for computing the reward
        # println("BOUNDS: x=$x, x1=$x1, x2=$x2, min_a = $min_a, all_actions=$all_actions, pos_actions=$pos_actions, av_actions=$(all_actions[all_actions .< abs(x2-x)]), a_dir=$a_dir,  r=$r ")
        x += a_dir # take the step
    end
    # error("review below, might be a bug with assigning first_a = 0.0")
    if abs(x2) <= p.term_radius
        # x1 < 0.7 && println("TERMINATION REWARD #2 FOR x2=$x")
        r += reward(p,x2,0.0) #termination reward
        first_a = 0.0
    end
    # x1 < 0.7 && println("exiting move $x1 -> $x2")
    # if isnan(first_a) #DEBUG: remove
    #     error("move: first_a = $first_a, x1=$x1, x2=$x2")
    # end
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
# lower_bound(p::LightDarkPOMDPs.LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = lower_bound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))

# NOTE: not for direct calling
# computes the reward for the straight-line path to target
function upper_bound(p::AbstractLD1, particle::LPDMParticle{Float64})
    if abs(particle.state) < p.term_radius
        return reward(p, particle.state, 0.0), 0.0
    else
        return move(p, particle.state, 0.0)
    end
end
