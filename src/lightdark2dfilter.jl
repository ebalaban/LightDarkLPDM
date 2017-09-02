# Kalman filter for the problem

# See Optimal Estimation of Dynamic Systems by Crassidis and Junkins p.256

"""
LightDark2DKalman

A simple Kalman filter for a LightDark2D problem. This does not work very well at localizing.

Fields:
    pomdp::LightDark2D

    obs_noise_offset::Float64
        Makes the filter more conservative by calculating the observation
        at a point this fraction of a standard deviation away from the mean in
        the direction that will increase it.
"""
immutable LightDark2DKalman <: Updater
    pomdp::LightDark2D
    obs_noise_offset::Float64
end
LightDark2DKalman(pomdp::LightDark2D) = LightDark2DKalman(pomdp, 0.0)

function update(kf::LightDark2DKalman, b::SymmetricNormal2, a::Vec2, o::Vec2)
    x = b.mean
    s = b.std

    # propagation
    xpm = x + a
    # so = obs_std(kf.pomdp, x[1]) # VERY sketchy
    # (b.std stays the same in propagation)

    # update
    if xpm[1] > kf.pomdp.min_noise_loc
        so = obs_std(kf.pomdp, xpm[1]+kf.obs_noise_offset*s)
    else
        so = obs_std(kf.pomdp, xpm[1]-kf.obs_noise_offset*s)
    end
    k = s^2/(s^2+so^2)
    xp = xpm + k*(o-x)
    sp = sqrt(1-k)*s
    return SymmetricNormal2(xp, sp)
end
