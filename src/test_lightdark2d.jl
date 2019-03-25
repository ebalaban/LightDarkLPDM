using Revise
import LPDM, D3Trees, LightDarkPOMDPs, execute2d
# execute2d.execute(solver=:lpdm)
steps = 1
# execute2d.execute(output=2, steps = steps, reward_func = :quadratic)
execute2d.execute(vis=Int64[1], solv_mode=:lpdm, reward_func=:quadratic, action_space_type=:adaptive, output=1, steps = steps)
# execute2d.execute(vis=Int64[], solv_mode=:lpdm_bv, reward_func=:quadratic, action_space_type=:bv, output=0, steps = steps)
# execute2d.execute(vis=Int64[], solv_mode=:despot, reward_func=:quadratic, action_space_type=:small, output=0, steps = steps)
# execute2d.batch_execute(n=50, debug=0, reward_func=:quadratic)
# execute2d.batch_execute(n=50, debug=0, reward_func=:quadratic)
