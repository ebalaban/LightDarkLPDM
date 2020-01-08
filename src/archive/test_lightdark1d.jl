using Revise
import LPDM, D3Trees, LightDarkLPDM, execute1d
# execute1d.execute(solver=:lpdm)
steps = 1
# execute1d.execute(s0=3/2*π, solver_mode = :lpdm, action_mode = :standard, obs_mode = :discrete, steps=steps, vis=Int64[], output=1)
# Juno.@enter execute1d.execute(s0=3/2*π, solver_mode = :lpdm, action_mode = :standard, obs_mode = :continuous, steps=steps, vis=Int64[1], output=3)
Juno.@run execute1d.execute(s0=3/2*π, solver_mode = :lpdm, action_mode = :standard, obs_mode = :discrete, steps=steps, vis=Int64[1], output=3)

# execute1d.batch_execute(n=50, debug=0, reward_func=:quadratic)
# execute1d.batch_execute(n=50, debug=0, reward_func=:quadratic)
# execute1d.batch_execute(n=30, debug=0)
