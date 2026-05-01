"""
Shared Triton kernels for the fused roll + LayerNorm + residual-add pattern.
Both FuseRollLayerNormAdd_32_768 and FuseRollLayerNormAdd_64_384 import
`roll_ln_add_dispatch` from here so they return the SAME function object and
avoid the replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Kernel: H=32, W=32, C=768  (BLOCK_C=1024)
# One program per token; default num_warps=4.
# --------------------------------------------------------------------------
@triton.jit
def _roll_ln_add_768(
    in3_ptr, w_ptr, b_ptr, res_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid   = tl.program_id(0)
    h     = pid // W
    w     = pid % W
    src_h = (h - 4 + H) % H
    src_w = (w - 4 + W) % W
    src_base = (src_h * W + src_w) * C

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    # Issue all loads upfront so they can be in-flight during computation
    x      = tl.load(in3_ptr + src_base + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias_v = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    res    = tl.load(res_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute LayerNorm statistics
    mean    = tl.sum(x, axis=0) / C
    diff    = (x - mean) * mask.to(tl.float32)
    var     = tl.sum(diff * diff, axis=0) / C
    inv_std = tl.rsqrt(var + 1e-5)
    x_norm  = diff * inv_std

    # Scale + shift + residual add
    result = x_norm * weight + bias_v + res

    if IS_BF16:
        tl.store(out_ptr + pid * C + offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + pid * C + offs, result.to(tl.float16), mask=mask)


# --------------------------------------------------------------------------
# Kernel: H=64, W=64, C=384  (BLOCK_C=512)
# One program per token; default num_warps=4.
# --------------------------------------------------------------------------
@triton.jit
def _roll_ln_add_384(
    in3_ptr, w_ptr, b_ptr, res_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid   = tl.program_id(0)
    h     = pid // W
    w     = pid % W
    src_h = (h - 4 + H) % H
    src_w = (w - 4 + W) % W
    src_base = (src_h * W + src_w) * C

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    # Issue all loads upfront so they can be in-flight during computation
    x      = tl.load(in3_ptr + src_base + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias_v = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    res    = tl.load(res_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute LayerNorm statistics
    mean    = tl.sum(x, axis=0) / C
    diff    = (x - mean) * mask.to(tl.float32)
    var     = tl.sum(diff * diff, axis=0) / C
    inv_std = tl.rsqrt(var + 1e-5)
    x_norm  = diff * inv_std

    # Scale + shift + residual add
    result = x_norm * weight + bias_v + res

    if IS_BF16:
        tl.store(out_ptr + pid * C + offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + pid * C + offs, result.to(tl.float16), mask=mask)


# --------------------------------------------------------------------------
# Single shared dispatch wrapper (route string selects the kernel variant)
# --------------------------------------------------------------------------
@torch.fx.wrap
def roll_ln_add_dispatch(in_0, in_1, in_2, in_3, route):
    """
    in_0 : LayerNorm bias   [C]
    in_1 : LayerNorm weight [C]
    in_2 : residual         [1, N, C]
    in_3 : raw input        (may be non-contiguous)
    route: "route_768" or "route_384"
    """
    in3_c   = in_3.contiguous()
    out     = torch.empty_like(in_2)
    IS_BF16 = in_2.dtype == torch.bfloat16

    if route == "route_768":
        _roll_ln_add_768[(1024,)](
            in3_c, in_1, in_0, in_2, out,
            H=32, W=32, C=768,
            BLOCK_C=1024,
            IS_BF16=IS_BF16,
        )
    else:  # route_384
        _roll_ln_add_384[(4096,)](
            in3_c, in_1, in_0, in_2, out,
            H=64, W=64, C=384,
            BLOCK_C=512,
            IS_BF16=IS_BF16,
        )

    return out