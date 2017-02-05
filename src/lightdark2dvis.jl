# plots a problem and a trajectory

@recipe function f(h::AbstractPOMDPHistory{Vec2})
    x = collect(s[1] for s in state_hist(h))
    y = collect(s[2] for s in state_hist(h))
    @series begin
        label --> "true"
        x, y
    end
    xo = collect(o[1] for o in observation_hist(h))
    yo = collect(o[2] for o in observation_hist(h))
    @series begin
        label --> "observed"
        seriestype --> :scatter
        color --> :green
        xo, yo
    end
end

@recipe function f(p::LightDark2DTarget)
    kwargd = Dict{Symbol, Any}() # can't figure out how to intercept keywords
    xlim = get(kwargd, :xlim, (-1.0,11.0))
    X = linspace(xlim...)
    ylim = get(kwargd, :ylim, (-10.0,10.0))
    Y = linspace(ylim...)
    inv_grays = cgrad([RGB(.95,.95,.95),RGB(.05,.05,.05)])
    @series begin
        fill := true
        color := inv_grays
        seriestype := :contour
        X, Y, (x,y)->obs_std(p,x)
    end
    @series begin
        label := ""
        color --> :black
        pts = Plots.partialcircle(0, 2*pi, 100, p.term_radius)
        x, y = Plots.unzip(pts)
        x, y
    end
end

@recipe function f(p::LightDark2D)
    kwargd = Dict{Symbol, Any}() # can't figure out how to intercept keywords
    xlim = get(kwargd, :xlim, (-1.0,11.0))
    X = linspace(xlim...)
    ylim = get(kwargd, :ylim, (-10.0,10.0))
    Y = linspace(ylim...)
    inv_grays = cgrad([RGB(1.0,1.0,1.0),RGB(0.0,0.0,0.0)])
    fill := true
    color := inv_grays
    seriestype := :contour
    X, Y, (x,y)->obs_std(p,x)
end

@recipe function f(b::SymmetricNormal2)
    label := ""
    color --> :black
    pts = Plots.partialcircle(0, 2*pi, 100, 3*b.std)
    x, y = Plots.unzip(pts)
    x += b.mean[1]
    y += b.mean[2]
    x, y
end

@recipe function f(b::ParticleBelief{Vec2})
    label := ""
    color --> :black
    seriestype := :scatter
    x = [p.state[1] for p in b.particles]
    y = [p.state[2] for p in b.particles]
    markersize --> [10.0*sqrt(p.weight) for p in b.particles]
    x, y
end

@recipe function f(b::ParticleCollection{Vec2})
    label := ""
    color --> :black
    seriestype := :scatter
    x = [p[1] for p in particles(b)]
    y = [p[2] for p in particles(b)]
    markersize --> [10.0*sqrt(weight(b,i)) for i in 1:n_particles(b)]
    x, y
end
