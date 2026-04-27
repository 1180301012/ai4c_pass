"""
Shared Triton kernels and dispatch function for all passes.
Importing this module ensures both pass files return the EXACT same
fused_dispatch object, satisfying the replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: x[i] * scale + bias  (scale_add route)
# ---------------------------------------------------------------------------
@triton.jit
def _scale_add_k(
    bias_ptr, scale_ptr, x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N
    x     = tl.load(x_ptr     + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    bias  = tl.load(bias_ptr)
    tl.store(out_ptr + offsets, x * scale + bias, mask=mask)


# ---------------------------------------------------------------------------
# Kernel B: max_pool2d(2,stride=1) + cat  (maxpool_cat route)
# ---------------------------------------------------------------------------
@triton.jit
def _maxpool_cat_k(
    in3_ptr, in4_ptr, out_ptr,
    B, C, H, W, H1, W1,
    BLOCK_SIZE: tl.constexpr,
):
    b      = tl.program_id(0)
    c      = tl.program_id(1)
    hw_pid = tl.program_id(2)
    HW = H * W
    hw_offsets = hw_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw_mask    = hw_offsets < HW
    oh = hw_offsets // W
    ow = hw_offsets % W
    base3 = b * (C * H1 * W1) + c * (H1 * W1)
    v00 = tl.load(in3_ptr + base3 + oh * W1 + ow,             mask=hw_mask, other=float('-inf'))
    v01 = tl.load(in3_ptr + base3 + oh * W1 + (ow + 1),       mask=hw_mask, other=float('-inf'))
    v10 = tl.load(in3_ptr + base3 + (oh + 1) * W1 + ow,       mask=hw_mask, other=float('-inf'))
    v11 = tl.load(in3_ptr + base3 + (oh + 1) * W1 + (ow + 1), mask=hw_mask, other=float('-inf'))
    max_val = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(out_ptr + b * (2*C*HW) + c * HW + hw_offsets, max_val, mask=hw_mask)
    val4 = tl.load(in4_ptr + b*(C*HW) + c*HW + hw_offsets, mask=hw_mask, other=0.0)
    tl.store(out_ptr + b * (2*C*HW) + (C+c) * HW + hw_offsets, val4, mask=hw_mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper — returned by replacement_func() in EVERY pass.
# route="scale_add"   : a=bias[1], b=scale[1], c=relu_out[B,C,H,W]
# route="maxpool_cat" : a=in3[B,C,H1,W1], b=in4[B,C,H,W], c=dummy(=a)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_dispatch(a, b, c, route):
    if route == "scale_add":
        N   = c.numel()
        out = torch.empty_like(c)
        _scale_add_k[
            (triton.cdiv(N, 2048),)
        ](a, b, c, out, N, BLOCK_SIZE=2048)
        return out
    elif route == "maxpool_cat":
        B, C, H1, W1 = a.shape
        H, W = H1 - 1, W1 - 1
        out  = torch.empty((B, 2*C, H, W), dtype=a.dtype, device=a.device)
        _maxpool_cat_k[
            (B, C, triton.cdiv(H*W, 1024))
        ](a, b, out, B, C, H, W, H1, W1, BLOCK_SIZE=1024)
        return out
    return c   # should never be reached