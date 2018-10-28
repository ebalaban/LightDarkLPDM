import LPDM: bounds, best_lb_action, best_ub_action
# using LightDarkPOMDPs

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

    tmp_lb = 0.0
    tmp_ub = 0.0
    tmp_lb_action = 0.0
    tmp_ub_action = 0.0

    # lb = Array{Float64}(0)
    # ub = Array{Float64}(0)
    # lb_action = Array{Float64}(0)
    # ub_action = Array{Float64}(0)

    for p in particles
        # push!(ub, upper_bound(pomdp,p)[1])
        # push!(lb, lower_bound(pomdp,p)[1])
        tmp_lb, tmp_lb_action = lower_bound(pomdp,p)
        tmp_ub, tmp_ub_action = upper_bound(pomdp,p)

        if tmp_lb < b.lb_
            b.lb_             = tmp_lb
            b.best_lb_action_ = tmp_lb_action
        end
        if tmp_ub > b.ub_
            b.ub_             = tmp_ub
            b.best_ub_action_ = tmp_ub_action
        end
    end

    # b.lb_ = minimum(lb)
    # b.ub_ = maximum(ub)
    # config.debug >= 2 && println("s=$(particles[1].state), lb=$(b.lb_), ub=$(b.ub_)")
    if b.ub_ > 100.0 #DEBUG, remove
        error("BOUNDS: ub=$(b.ub_)")
    end
    return b.lb_, b.ub_
end

LPDM.best_lb_action(b::LDBounds1d) = isnan(b.best_lb_action_) ? error("best_lb_action undefined. Call bounds() first") : b.best_lb_action_
LPDM.best_ub_action(b::LDBounds1d) = isnan(b.best_ub_action_) ? error("best_ub_action undefined. Call bounds() first") : b.best_ub_action_

function move(p::AbstractLD1, x1::Float64, x2::Float64)#::(Float64,Float64)
    direction = x2 > x1 ? 1 : -1
    all_actions = POMDPs.actions(p,true)
    I = findall(!iszero, all_actions)
    nz_actions = all_actions[I]
    min_a = minimum(nz_actions)

    r = 0.0
    x = x1
    first_a = NaN
    a_dir = NaN

    if abs(x1) <= p.term_radius
        return reward(p,x1,0.0), 0.0
    end

    while (abs(x2-x) > min_a) && (abs(x) >= p.term_radius)
        a = maximum(nz_actions[nz_actions .<= abs(x2-x)]) # maximum action not exceeding Δx
        a_dir = direction * a
        if isnan(first_a)
            first_a = a_dir # assign first action (directional)
        end
        r += reward(p, x, a_dir) # use current state for computing the reward
        # println("BOUNDS: x=$x, x1=$x1, x2=$x2, actions=$(actions[actions .< abs(x2-x)]), a_dir=$a_dir,  r=$r ")
        x += a_dir # take the step
    end
    if abs(x2) <= p.term_radius
        # x1 < 0.7 && println("TERMINATION REWARD #2 FOR x2=$x")
        r += reward(p,x2,0.0) #termination reward
    end
    # x1 < 0.7 && println("exiting move $x1 -> $x2")
    return r, first_a #action sign depends on the direction
end

# NOTE: not for direct calling
# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach
function lower_bound(p::LightDark1DDespot, particle::LPDMParticle{Float64})

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
function upper_bound(p::LightDark1DDespot, particle::LPDMParticle{Float64})
    if abs(particle.state) < p.term_radius
        return reward(p, particle.state, 0.0), 0.0
    else
        return move(p, particle.state, 0.0)
    end
end
# upper_bound(p::LightDarkPOMDPs.LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = upper_bound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))


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
