import torch
from pass_dir.kernels import shared_dispatch


# ── pattern ──────────────────────────────────────────────────────────────────
def pattern(w, x):
    x_fp32 = x.to(torch.float32)
    x_sq   = x_fp32.pow(2)
    mean   = x_sq.mean(-1, keepdim=True)
    mean_eps  = mean + 1e-06
    rsqrt_val = torch.rsqrt(mean_eps)
    x_norm    = x_fp32 * rsqrt_val
    x_norm_cast = x_norm.to(torch.bfloat16)
    out = w * x_norm_cast
    return out


def replacement_args(w, x):
    return (w, x, "rmsnorm_1e6_bf16")


def replacement_func():
    return shared_dispatch