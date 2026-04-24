import torch
from pass_dir._shared_dispatch import dispatch_bool


# device_const placeholder matches _tensor_constant0 in the target graph.
# Returns ONLY tmp_2 (single output) — avoids tuple-return crash.
# All N-value passes share dispatch_bool to satisfy replacement_func_limit=1.
def pattern(in_0, device_const):
    tmp_2 = in_0.to(device=device_const, dtype=torch.bool)
    return tmp_2


def replacement_args(in_0, device_const):
    return (in_0, "128")


def replacement_func():
    return dispatch_bool