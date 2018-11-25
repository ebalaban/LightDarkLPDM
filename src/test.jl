using Revise
import LPDM, LightDarkPOMDPs, execute1d
# execute1d.execute(solver=:lpdm)
# execute1d.execute(solver=:despot)
execute1d.batch_execute(n=50, debug=0)
