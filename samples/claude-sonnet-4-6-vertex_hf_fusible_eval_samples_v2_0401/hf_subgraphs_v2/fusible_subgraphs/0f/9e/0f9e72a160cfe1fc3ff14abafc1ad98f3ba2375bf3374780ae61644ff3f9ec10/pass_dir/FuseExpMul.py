import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: elementwise  out = exp(scale) * x
# Hard-coded for N=512, single program, no mask, minimal args.
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def exp_mul_kernel(
    scale_ptr,          # 0-d or 1-element float tensor
    x_ptr,              # flat [N] input
    out_ptr,            # flat [N] output
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)       # [0 … 511], no mask needed
    scale   = tl.load(scale_ptr).to(tl.float32)
    exp_s   = tl.exp(scale)
    x       = tl.load(x_ptr + offsets)
    result  = exp_s * x.to(tl.float32)
    tl.store(out_ptr + offsets, result.to(x.dtype))


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper (must be @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_exp_mul_wrapper(scale, x):
    """
    Replacement for:
        s   = scale.exp()
        out = s * x
    Hard-coded for N=512 (x has shape [1,1,512] in this graph).
    """
    out = torch.empty_like(x)
    exp_mul_kernel[(1,)](
        scale, x, out,
        BLOCK_SIZE=512,
        num_warps=4, num_stages=1,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement_args / replacement_func
# ──────────────────────────────────────────────────────────────────────────────
def pattern(scale, x):
    """Matches: s = scale.exp(); out = s * x"""
    s   = scale.exp()
    out = s * x
    return out


def replacement_args(scale, x):
    return (scale, x)


def replacement_func():
    return fused_exp_mul_wrapper