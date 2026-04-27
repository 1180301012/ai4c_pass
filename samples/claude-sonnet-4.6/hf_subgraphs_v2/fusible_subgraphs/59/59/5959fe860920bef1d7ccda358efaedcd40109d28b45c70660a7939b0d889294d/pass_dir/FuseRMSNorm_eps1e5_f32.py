import torch
from pass_dir.kernels import shared_dispatch


# ── pattern ──────────────────────────────────────────────────────────────────
# TinyLlama variant: eps=1e-5, intermediate cast to float32 (no-op),
# weight (bfloat16) * float32 → float32 output.
def pattern(w, x):
    x_fp32 = x.to(torch.float32)
    x_sq   = x_fp32.pow(2)
    mean   = x_sq.mean(-1, keepdim=True)
    mean_eps  = mean + 1e-05
    rsqrt_val = torch.rsqrt(mean_eps)
    x_norm    = x_fp32 * rsqrt_val
    x_norm_cast = x_norm.to(torch.float32)   # no-op float32→float32
    out = w * x_norm_cast                    # bfloat16 × float32 → float32
    return out


def replacement_args(w, x):
    return (w, x, "rmsnorm_1e5_f32")


def replacement_func():
    return shared_dispatch