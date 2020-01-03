# LightDarkLPDM inspired and forked from LightDarkPOMDPs

![A trajectory of a good solution to a LightDark2DTarget problem](https://github.com/zsunberg/LightDarkPOMDPs.jl/raw/master/img/target_good_solution.gif)

[![Build Status](https://travis-ci.org/zsunberg/LightDarkPOMDPs.jl.svg?branch=master)](https://travis-ci.org/zsunberg/LightDarkPOMDPs.jl)

In the `LightDark2D` problem, cost is quadratic in the distance from the origin; In the `LightDark2DTarget` problem, there is a cost of -1 accrued for every step outside of the target region (a radius from the origin controlled by the `term_radius` field of `LightDark2DTarget`).

There is also a Kalman filter `LightDark2DKalman`, but Kalman filtering is rather ill-suited to this problem because the true beliefs are not similar to a Gaussian distribution. Instead, it is much better to use a particle filter from ParticleFilters.jl.

See [test/runtests.jl](test/runtests.jl) for usage examples.

<!--
[![Coverage Status](https://coveralls.io/repos/zsunberg/LightDarkPOMDPs.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/zsunberg/LightDarkPOMDPs.jl?branch=master)

[![codecov.io](http://codecov.io/github/zsunberg/LightDarkPOMDPs.jl/coverage.svg?branch=master)](http://codecov.io/github/zsunberg/LightDarkPOMDPs.jl?branch=master)
-->
# LightDarkLPDM
