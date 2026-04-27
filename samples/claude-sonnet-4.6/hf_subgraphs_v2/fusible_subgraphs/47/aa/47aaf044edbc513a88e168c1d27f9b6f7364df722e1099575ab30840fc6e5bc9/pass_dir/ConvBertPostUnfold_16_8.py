import torch
from pass_dir._shared_dispatch import convbert_dispatch


# Pattern: transpose + reshape + reshape (no unfold → no proxy tracing failure)
# in_0 matches the unfold output [1, C*9, L] in the target graph.
def pattern(in_0):
    tmp_3 = in_0.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0, "route_post_16_8")


def replacement_func():
    return convbert_dispatch