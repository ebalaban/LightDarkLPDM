using Revise
include("execute.jl")
using .execute
import LPDM, D3Trees, LightDarkLPDM

# execute.batch_execute(dims=2)
# Juno.@run execute.batch_execute(dims=1, debugger=true)
execute.batch_execute(dims=1, debugger=true)
