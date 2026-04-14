import torch
from pass_dir._shared_kernels import dispatch_kernel, _dt


# Universal GELU+add pass — matches all size variants (no size-specific views).
# Returns a SINGLE tensor (the residual sum = tmp_6 = logical tmp_10).

def pattern(in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    return in_3 + tmp_5


def replacement_args(in_2, in_3):
    # Hardcoded route "ga" — no .meta access, safe to trace
    return (in_2, in_3, in_2, "ga")


def replacement_func():
    return dispatch_kernel