"""
Shared kernel module used by FuseBilinearSigmoidMul and FuseSigmoidMulBilinearAdd.
Both passes import dispatch_sigmoid_mul from here, so replacement_func()
returns the SAME Python object in both passes, staying within the
output_pass_replacement_func_limit.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_mul_fp32_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise fused kernel: out = sigmoid(x) * y  (fp32 path)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.sigmoid(x) * y, mask=mask)


@triton.jit
def _sigmoid_mul_fp16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise fused kernel: out = sigmoid(x) * y  (fp16 path)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.sigmoid(x.to(tl.float32)) * y.to(tl.float32), mask=mask)


@triton.jit
def _combined_sigmoid_mul_kernel(
    x1_ptr, y1_ptr, out1_ptr,
    x2_ptr, y2_ptr, out2_ptr,
    N1, N2,
    BLOCK_SIZE: tl.constexpr,
):
    """Single kernel that processes BOTH sigmoid+mul pairs simultaneously.
    Grid size: ceil((N1 + N2) / BLOCK_SIZE).
    Region 1  (pid * BS < N1): processes pair (x1, y1) → out1
    Region 2  (pid * BS >= N1): processes pair (x2, y2) → out2
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # --- pair 1 ---
    mask1 = offs < N1
    x1 = tl.load(x1_ptr + offs, mask=mask1, other=0.0)
    y1 = tl.load(y1_ptr + offs, mask=mask1, other=0.0)
    out_f32_1 = tl.sigmoid(x1.to(tl.float32)) * y1.to(tl.float32)
    tl.store(out1_ptr + offs, out_f32_1, mask=mask1)

    # --- pair 2 (offs offset by N1 to avoid overlap) ---
    rel = offs - N1
    mask2 = (offs >= N1) & (rel < N2)
    x2 = tl.load(x2_ptr + rel, mask=mask2, other=0.0)
    y2 = tl.load(y2_ptr + rel, mask=mask2, other=0.0)
    out_f32_2 = tl.sigmoid(x2.to(tl.float32)) * y2.to(tl.float32)
    tl.store(out2_ptr + rel, out_f32_2, mask=mask2)


@torch.fx.wrap
def dispatch_sigmoid_mul(x, y):
    """Single sigmoid*mul kernel dispatch (for single-pair use cases)."""
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    if x.dtype == torch.float16:
        _sigmoid_mul_fp16_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    else:
        _sigmoid_mul_fp32_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


@torch.fx.wrap
def combined_sigmoid_mul(a, b, c, d):
    """Combined dispatch for BOTH sigmoid*mul pairs in one kernel launch.
    a = in_4 (64x64 sigmoid input), b = in_3 (64x64 multiplier)
    c = conv2d_out (16x16 sigmoid input), d = in_2 (16x16 multiplier)
    Returns (out1, out2) matching pattern's (tmp_5, tmp_7).
    """
    N1 = a.numel()
    N2 = c.numel()
    out1 = torch.empty_like(a)
    out2 = torch.empty_like(c)
    BLOCK_SIZE = 1024
    total = N1 + N2
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    _combined_sigmoid_mul_kernel[grid](
        a, b, out1,
        c, d, out2,
        N1, N2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (out1, out2)