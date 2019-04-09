using Revise
import LPDM, D3Trees, LightDarkPOMDPs, execute
steps = 1

# execute.batch_execute(dims=1, n=1, debug=1, reward_mode=:quadratic)
execute.batch_execute(dims=2, n=5, debug=0, reward_mode=:quadratic)

# Juno.@enter execute.batch_execute(dims=1, n=1, debug=1, reward_mode=:quadratic)
# execute.execute(1, s0=3/2*π, solver_mode = :lpdm, action_mode = :standard, obs_mode = :continuous, steps=steps, output=3)
