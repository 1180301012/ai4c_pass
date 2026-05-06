"""
Shared Triton kernels for fusing batch_norm (inference) + SiLU + spatial mean.

Key optimizations:
  - BN: y = weight*(x - running_mean)/sqrt(running_var+eps) + bias  (all fp32)
  - SiLU:  x * sigmoid(x)
  - mean:  computed via atomic float32 accumulation over H*W tiles

The output mean is then stored in a contiguous (B, C, 1, 1) tensor with the
output tensor's dtype (float32/float16/bfloat16) via a second lightweight kernel.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused BN -> SiLU -> atomic mean accumulation
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW', 'C'],
)
@triton.jit
def _fused_bn_silu_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_out_ptr,
    HW,
    C,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Each program handles a (batch, channel) pair and a spatial tile."""
    pid_bc = tl.program_id(0)    # batch index * C + channel index
    pid_hw = tl.program_id(1)    # spatial tile index

    b_idx = pid_bc // C
    c_idx = pid_bc  % C

    # ---- BN parameters for this channel ----
    rm   = tl.load(running_mean_ptr + c_idx).to(tl.float32)
    rv   = tl.load(running_var_ptr  + c_idx).to(tl.float32)
    wv   = tl.load(weight_ptr       + c_idx).to(tl.float32)
    bv   = tl.load(bias_ptr         + c_idx).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(rv + eps)
    a    = wv * inv_std
    bias_val = bv - a * rm

    # ---- spatial tile ----
    hw_start   = pid_hw * BLOCK_HW
    offsets    = hw_start + tl.arange(0, BLOCK_HW)
    mask       = offsets < HW

    in_base    = (b_idx * C + c_idx) * HW
    x_vals     = tl.load(x_ptr + in_base + offsets, mask=mask, other=0.0).to(tl.float32)

    # BN
    y = a * x_vals + bias_val

    # SiLU
    silu_y = y * tl.sigmoid(y)

    # Store BN+SiLU output (cast to output dtype)
    tl.store(out_ptr + in_base + offsets, silu_y.to(x_vals.dtype), mask=mask)

    # Compute partial sum for mean (only valid elements)
    silu_y_safe = tl.where(mask, silu_y, 0.0)
    partial_sum = tl.sum(silu_y_safe, axis=0)

    # Accumulate into float32 mean via atomic add
    tl.atomic_add(mean_out_ptr + pid_bc, partial_sum)


# ---------------------------------------------------------------------------
# Kernel 2: cast float32 mean_out to the output dtype (only needed for fp16/bf16)
# ---------------------------------------------------------------------------
@triton.jit
def _cast_mean_to_out_kernel(
    mean_fp32_ptr,
    mean_out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < n_elements
    val   = tl.load(mean_fp32_ptr + offs, mask=mask, other=0.0)
    tl.store(mean_out_ptr + offs, val.to(mean_out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Python-level interface
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_silu_mean(running_mean, running_var, bias, weight, x):
    """
    Fused inference-BN + SiLU + spatial mean.

    Args:
      running_mean, running_var, bias, weight : BN parameters (shape [C])
      x                                        : input (shape [B, C, H, W])

    Returns:
      out           : BN+SiLU output, same shape / dtype as x          [B, C, H, W]
      mean_out      : spatial mean,     shape [B, C, 1, 1], same dtype as x
    """
    B, C, H, W = x.shape
    HW = H * W
    n_bc = B * C

    # Clean floats for BN computation
    x_f    = x.to(tl.float32)   # Triton sees x_f.dtype = float32

    # Output buffers
    out         = torch.empty_like(x)
    mean_fp32   = torch.empty((n_bc,), dtype=torch.float32, device=x.device)
    mean_out    = torch.empty((B, C, 1, 1), dtype=x.dtype,     device=x.device)

    # -----------------------------------------------------------------
    # Grid for fused BN+SiLU kernel: one program per (b,c) and per HW tile
    # -----------------------------------------------------------------
    grid = lambda meta: (n_bc, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_bn_silu_mean_kernel[grid](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        mean_out_ptr=mean_fp32,
        HW=HW,
        C=C,
        eps=1e-5,
    )

    # -----------------------------------------------------------------
    # Cast float32 mean_out to the original dtype (only if needed)
    # -----------------------------------------------------------------
    cast_required = (x.dtype != torch.float32)
    if cast_required:
        BLOCK_cast = 256
        cast_grid  = (triton.cdiv(n_bc, BLOCK_cast),)
        _cast_mean_to_out_kernel[cast_grid](
            mean_fp32_ptr=mean_fp32,
            mean_out_ptr=mean_out,
            n_elements=n_bc,
            BLOCK=BLOCK_cast,
        )

    return out, mean_out