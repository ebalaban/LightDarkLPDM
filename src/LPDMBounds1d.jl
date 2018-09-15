import LPDM: bounds
using LightDarkPOMDPs

mutable struct LDBounds1d{A}
    lb_    ::Float64
    ub_    ::Float64
    best_lb_action_::Float64
    best_ub_action_::Float64

    function LDBounds1d{A}() where A
        this = new()
        this.lb_ = +Inf
        this.ub_ = -Inf
        this.best_lb_action_ = NaN
        this.best_ub_action_ = NaN
        return this
    end
end

function LPDM.bounds(b::LDBounds1d,
                     pomdp::LightDarkPOMDPs.AbstractLD1,
                     particles::Vector{LPDMParticle{LDState}},
                     config::LPDMConfig)

    ub = Array{Float64}(0);
    lb = Array{Float64}(0);

    for p in particles
        show(upper_bound(pomdp,p))
        show(typeof(upper_bound(pomdp,p)))
        push!(ub, upper_bound(pomdp,p)[1])
        push!(lb, lower_bound(pomdp,p)[1])
    end

    b.lb_ = minimum(lb)
    b.ub_ = maximum(ub)

    return b.lb_, b.ub_
end

best_ub_action(b::LDBounds1d) = isnan(b.best_lb_action_) ? error("best_lb_action undefined. Call bounds() first") : b.best_lb_action_
best_ub_action(b::LDBounds1d) = isnan(b.best_ub_action_) ? error("best_ub_action undefined. Call bounds() first") : b.best_ub_action_

function move(p::LightDarkPOMDPs.AbstractLD1, x1::Float64, x2::Float64)#::(Float64,Float64)
    #direction = x2 > x1 ? 1 : -1
    actions = Base.findnz(POMDPs.actions(p,true)')[3] # get only positive non-zero actions
    min_a = minimum(actions)
    # compute just with positive case - the two cases are symmetrical
    orig = minimum(abs.([x1,x2]))
    dest = maximum(abs.([x1,x2]))

    r = 0.0
    Δx = 0.0
    x = orig
    first_a = NaN

    while dest-x > min_a
        Δx = dest - x
        a = maximum(actions[actions .<= Δx]) # maximum action not exceeding Δx
        if isnan(first_a)
            first_a = a # assign first action
        end
        r += reward(p, x, a) # use current state for computing the reward (NOTE: assumes reward symmetry)
        # println("BOUNDS: x=$x, Δx=$(Δx), x1=$x1, x2=$x2, orig=$orig, dest=$dest, actions=$(actions[actions .< Δx]), a=$a,  r=$r ")
        x += a # take the step
        # error("done")
    end
    return r, first_a
end

# NOTE: not for direct calling
# computes the cost of traveling to the low noise region and only then towards target. i.e. a slow approach
function lower_bound(p::LightDarkPOMDPs.LightDark1DDespot, particle::POMDPToolbox.Particle{Float64})

    r1,a1 = move(p, particle.state, p.min_noise_loc)         # estimate cost of moving x to low noise x
    r2,a2 = move(p, p.min_noise_loc, p.term_radius)         # estimate cost of moving x to target region from the low noise region

    return r1+r2, a1
end
lower_bound(p::LightDarkPOMDPs.LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = lower_bound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))

# NOTE: not for direct calling
# computes the reward for the straight-line path to target
upper_bound(p::LightDarkPOMDPs.LightDark1DDespot, particle::POMDPToolbox.Particle{Float64}) = move(p, particle.state, p.term_radius)
upper_bound(p::LightDarkPOMDPs.LightDark1DDespot, particle::LPDM.LPDMParticle{Float64}) = upper_bound(p, POMDPToolbox.Particle{Float64}(particle.state, particle.weight))


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
