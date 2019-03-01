using Revise
import LPDM, D3Trees, LightDarkPOMDPs, execute2d
# execute2d.execute(solver=:lpdm)
steps = 10
execute2d.execute(output=2, steps = steps)
# execute2d.execute(vis=[], output=2, steps = steps)
# execute2d.batch_execute(n=50, debug=0, reward_func=:quadratic)
# execute2d.batch_execute(n=1, debug=2, reward_func=:quadratic)
