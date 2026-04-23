import torch

from pass_dir.shared_depthpro_dispatch import depthpro_dispatch


# Pattern matching function
# Match the reliable tail of the graph.
def pattern(a, b, c):
    tmp_0 = torch.cat([a, b, c], dim=0)
    tmp_1 = tmp_0.to(torch.float16)
    return tmp_1


# Argument extraction function
def replacement_args(a, b, c):
    return (a, b, c, "cat_cast")


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return depthpro_dispatch