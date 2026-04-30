import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: fuse  x * sigmoid(y)  into a single Triton kernel.
#
# This matches BOTH sigmoid×mul occurrences in the model:
#   1. in_3 * sigmoid(interpolate(in_4))   → branch A
#   2. in_2 * sigmoid(conv2d(in_5,...))   → branch B
# ─────────────────────────────────────────────────────────────────────────────
def pattern(x, y):
    sig = torch.sigmoid(y)
    result = x * sig
    return result


def replacement_args(x, y):
    return (x, y)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: element-wise  out[i] = x[i] * sigmoid(y[i])
# Fixed BLOCK — no autotune overhead, good for all tensor sizes.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_sigmoid_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    IS_FP16: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid     = tl.program_id(0)
    offset  = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offset < n_elements

    x   = tl.load(x_ptr   + offset, mask=mask, other=0.0).to(tl.float32)
    y   = tl.load(y_ptr   + offset, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(y)
    out = x * sig

    if IS_FP16:
        tl.store(out_ptr + offset, out.to(tl.float16),   mask=mask)
    elif IS_BFLOAT16:
        tl.store(out_ptr + offset, out.to(tl.bfloat16),  mask=mask)
    else:
        tl.store(out_ptr + offset, out,                   mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_sigmoid_mul(x, y):
    out        = torch.empty_like(x)
    n_elements = x.numel()
    is_fp16    = (x.dtype == torch.float16)
    is_bfloat16 = (x.dtype == torch.bfloat16)

    BLOCK = 4096
    grid  = (triton.cdiv(n_elements, BLOCK),)
    _fused_sigmoid_mul_kernel[grid](
        x, y, out, n_elements, is_fp16, is_bfloat16,
        BLOCK=BLOCK, num_warps=8,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# replacement_func  (zero-argument, returns callable)
# ─────────────────────────────────────────────────────────────────────────────
def replacement_func():
    return fused_sigmoid_mul