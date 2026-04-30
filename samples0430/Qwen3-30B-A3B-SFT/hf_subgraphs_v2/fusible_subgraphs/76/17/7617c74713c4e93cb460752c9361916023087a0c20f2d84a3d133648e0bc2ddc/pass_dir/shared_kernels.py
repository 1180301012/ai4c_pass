"""
Shared Triton kernels and dispatch wrapper used by both
FuseAddMeanDropoutBatchNorm.py and FuseBatchNorm.py.

Using a single shared dispatch_wrapper ensures both passes return the
SAME function object from replacement_func(), which satisfies the
output_pass_replacement_func_limit constraint.
"""
import torch
import triton
import triton.language as tl


# ─── Kernel 1: fused add + spatial mean ──────────────────────────────────────

@triton.jit
def add_mean_kernel(
    in4_ptr, in5_ptr, out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per (n,c) pair; computes mean(in4+in5) over HW elements."""
    pid  = tl.program_id(0)    # = n*C + c
    base = pid * HW

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    x4 = tl.load(in4_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x5 = tl.load(in5_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    s = tl.sum(x4 + x5, axis=0)
    tl.store(out_ptr + pid, s / HW)   # store scalar


# ─── Kernel 2: batch-norm inference ──────────────────────────────────────────

@triton.jit
def bn_inference_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per (n,c) element of [N,C]. Applies BN affine transform."""
    pid  = tl.program_id(0)
    c    = pid % C

    x = tl.load(x_ptr    + pid).to(tl.float32)
    m = tl.load(mean_ptr + c   ).to(tl.float32)
    v = tl.load(var_ptr  + c   ).to(tl.float32)
    w = tl.load(weight_ptr + c ).to(tl.float32)
    b = tl.load(bias_ptr + c   ).to(tl.float32)

    y = w * (x - m) / tl.sqrt(v + eps) + b
    tl.store(out_ptr + pid, y.to(x_ptr.dtype.element_ty))


# ─── Shared dispatch wrapper ─────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_wrapper(a, b, c, d, e, route):
    """
    Route-based dispatch for all fused kernels.

    route == "add_mean"  : a=in4 [N,C,H,W], b=in5 [N,C,H,W]
                           c,d,e unused (None).
    route == "bn_inference": a=x [N,C], b=running_mean [C],
                           c=running_var [C], d=weight [C], e=bias [C].
    """
    if route == "add_mean":
        N, C, H, W = a.shape
        HW = H * W
        NC = N * C
        BLOCK_SIZE = 1
        while BLOCK_SIZE < HW:
            BLOCK_SIZE <<= 1
        out = torch.empty((N, C), dtype=a.dtype, device=a.device)
        add_mean_kernel[(NC,)](a, b, out, HW=HW, BLOCK_SIZE=BLOCK_SIZE)
        return out
    else:  # "bn_inference"
        N, C = a.shape[0], a.shape[1]
        NC   = N * C
        out  = torch.empty_like(a)
        bn_inference_kernel[(NC,)](
            a, b, c, d, e, out,
            C=C, eps=1e-5, BLOCK_SIZE=1,
        )
        return out