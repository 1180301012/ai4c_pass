import torch
import triton
import triton.language as tl
from pass_dir.kernels import dispatch_replacement, _run_add_mean, _run_add_mean_bn


# ---------------------------------------------------------------------------
# Full-chain: add + mean + aten.batch_norm (cudnn_enabled=False variant).
# aten.batch_norm arg order:
#   input, weight, bias, running_mean, running_var, training,
#   momentum, eps, cudnn_enabled
# Returns single Tensor (not tuple).  No dropout (eliminated).
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_8 = torch.ops.aten.batch_norm.default(
        tmp_5, in_3, in_2, in_0, in_1, False, 0.1, 1e-05, False)
    return (tmp_8, tmp_5)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "route_full_bn")


def replacement_func():
    return dispatch_replacement