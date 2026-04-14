"""
Shared Triton kernel for the fused (scale × norm, clamp, div, mul-weight) pass.

Pattern matched (per scale variant):
    tmp_4 = tmp_3 * scale_const        # scale the precomputed L2-norm
    tmp_5 = tmp_4.clamp(min=1e-05)     # floor at eps
    tmp_6 = tmp_2 / tmp_5              # normalise every feature element
    tmp_7 = tmp_6 * in_0               # apply scalar weight

Inputs to the kernel:
  in_0   – scalar weight tensor, shape [1]
  tmp_3  – L2-norm result,   shape [B, C, 1]  (keepdim=True)
  tmp_2  – flattened input,  shape [B, C, FEAT]

One CTA per (b, c) pair; processes FEAT elements in a single pass.
Supports float16, bfloat16, float32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_clamp_div_mul_kernel(
    in0_ptr,          # [1]         scalar weight
    norm_ptr,         # [N_ROWS]    L2-norm per row (contiguous from [B,C,1])
    x_ptr,            # [N_ROWS, FEAT]
    out_ptr,          # [N_ROWS, FEAT]
    scale_const,
    N_ROWS,
    FEAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row * FEAT
    offsets = tl.arange(0, BLOCK)
    mask = offsets < FEAT

    # Compute denominator in fp32 for stable arithmetic on all dtypes
    norm_raw = tl.load(norm_ptr + row)
    norm_f32 = norm_raw.to(tl.float32)
    denom = tl.maximum(norm_f32 * scale_const, 1e-5)

    # Load input row and weight, compute and store in native dtype
    x_raw = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0)
    x_f32 = x_raw.to(tl.float32)

    g_f32 = tl.load(in0_ptr).to(tl.float32)
    out_f32 = x_f32 / denom * g_f32

    tl.store(out_ptr + row_base + offsets, out_f32.to(x_raw.dtype), mask=mask)


@torch.fx.wrap
def norm_dispatch(in_0, tmp_3, tmp_2, route_str):
    """
    Dispatch wrapper shared across all scale-variant passes.

    in_0     : scalar weight [1]
    tmp_3    : L2 norm [B, C, 1]   contiguous – stride-1 along the row axis
    tmp_2    : flattened input [B, C, FEAT]  contiguous
    route_str: "scale_144" or "scale_072"

    Arithmetic is in fp32 for fast scalar division on all dtypes.
    No reshape() / view() calls – tensors are already contiguous.
    """
    if route_str == "scale_144":
        sc    = 0.14433756729740643
        BLOCK = 64    # next_power_of_2(48)
    else:             # "scale_072"
        sc    = 0.07216878364870322
        BLOCK = 256   # next_power_of_2(192)

    B    = tmp_2.shape[0]
    C    = tmp_2.shape[1]
    FEAT = tmp_2.shape[2]
    N_ROWS = B * C

    # Allocate output – same shape/dtype/device as tmp_2
    out = torch.empty((B, C, FEAT), dtype=tmp_2.dtype, device=tmp_2.device)

    nw = 2 if BLOCK <= 64 else (4 if BLOCK <= 256 else 8)
    _scale_clamp_div_mul_kernel[(N_ROWS,)](
        in_0, tmp_3, tmp_2, out,
        sc, N_ROWS, FEAT, BLOCK,
        num_warps=nw,
    )
    return out