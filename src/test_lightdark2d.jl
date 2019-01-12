using Revise
import LPDM, D3Trees, LightDarkPOMDPs, execute2d
# execute2d.execute(solver=:lpdm)
# steps = 100
# execute2d.execute(s0=3/2*π, steps=steps, vis=[steps], output=2)
# execute2d.batch_execute(n=50, debug=0, reward_func=:quadratic)
execute2d.batch_execute(n=30, debug=0, reward_func=:fixed)
