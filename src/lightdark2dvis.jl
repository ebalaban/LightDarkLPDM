rect(w, h, x, y) = Shape(x + [0,w,w,0], y + [0,0,h,h])

@recipe function f(h::AbstractPOMDPHistory{Vec2})
    sh = state_hist(h)
    x = collect(s[1] for s in sh[1:end-1])
    y = collect(s[2] for s in sh[1:end-1])
    @series begin
        label --> "true state"
        color --> :orange
        x, y
    end
    ah = action_hist(h)
    xa = [last(x), last(x)+ah[end][1]]
    ya = [last(y), last(y)+ah[end][2]]
    @series begin
        label --> "action"
        color --> :blue
        linestyle --> :dash
        xa, ya
    end
    xo = collect(o[1] for o in observation_hist(h)[1:end-1])
    yo = collect(o[2] for o in observation_hist(h)[1:end-1])
    @series begin
        label --> "observed"
        seriestype --> :scatter
        color --> :green
        # markersize -->  
        marker --> :cross
        xo, yo
    end
end

@recipe function f(p::LightDark2DTarget)
    X = linspace(-1.0, 11.0)
    Y = linspace(-100.0, 100.0)
    inv_grays = cgrad([RGB(1.0, 1.0, 1.0),RGB(0.0,0.0,0.0)])
    bg_inside := :black
    xlim --> (-3,10)
    ylim --> (-4,8)
    @series begin
        fill := true
        color := inv_grays
        seriestype := :contour
        X, Y, (x,y)->obs_std(p,x)
    end
    @series begin
        label --> "target"
        color --> :red
        pts = Plots.partialcircle(0, 2*pi, 100, p.term_radius)
        x, y = Plots.unzip(pts)
        x, y
    end
end

@recipe function f(p::LightDark2D)
    X = linspace(-1.0, 11.0)
    Y = linspace(-100.0, 100.0)
    inv_grays = cgrad([RGB(1.0,1.0,1.0),RGB(0.0,0.0,0.0)])
    bg_inside := :black
    xlim --> (-3,10)
    ylim --> (-4,8)
    @series begin
        fill := true
        color := inv_grays
        seriestype := :contour
        X, Y, (x,y)->obs_std(p,x)
    end
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

Base.show(io::IO, mime::MIME"image/png", p::LightDark2DTarget) = show(io, mime, plot(p))

function Base.show(io::IO, mime::MIME"image/png", t::Tuple{LightDark2DTarget, AbstractPOMDPHistory})
    p = plot(first(t))
    plot!(p, last(t))
    show(io, mime, p)
end
