import torch
from pass_dir.rmsnorm_kernel import rmsnorm_dispatch


def pattern(in_0, in_2):
    """Match: x -> float32 -> pow(2) -> mean(-1,keepdim=True)
                    + eps(1e-6) -> rsqrt -> * weight -> bfloat16
    This is the RMSNorm path in SmolLM3-3B (eps=1e-6)."""
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim = True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2, "1e-6")


def replacement_func():
    return rmsnorm_dispatch