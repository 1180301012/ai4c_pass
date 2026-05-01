import torch
import triton
import triton.language as tl
from pass_dir.kernels import dispatch_replacement, _run_add_mean, _run_add_mean_bn


# ---------------------------------------------------------------------------
# Full-chain pattern: add + mean + BN arithmetic (sqrt + div form).
# Alternative decomposition: (mean - rm) / sqrt(rv + eps) * w + b
# No dropout (eliminated). Returns (tmp_8, tmp_5).
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4  = in_5 + in_4
    tmp_5  = tmp_4.mean((2, 3), keepdim=False)
    # BN arithmetic: (mean - rm) / sqrt(rv + eps) * w + b
    t_veps = in_1 + 1e-05          # aten.add.Scalar(rv, eps)
    t_std  = t_veps.sqrt()          # aten.sqrt.default
    t_sub  = tmp_5 - in_0           # aten.sub.Tensor
    t_norm = t_sub / t_std           # aten.div.Tensor
    t_scl  = t_norm * in_3           # aten.mul.Tensor  (weight)
    tmp_8  = t_scl + in_2            # aten.add.Tensor  (bias)
    return (tmp_8, tmp_5)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "route_legit_bn")


def replacement_func():
    return dispatch_replacement