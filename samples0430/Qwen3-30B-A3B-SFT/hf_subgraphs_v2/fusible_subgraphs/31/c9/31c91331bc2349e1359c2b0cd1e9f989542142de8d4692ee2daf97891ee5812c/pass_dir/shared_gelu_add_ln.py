"""
Shared Triton kernel for fused GELU + residual add + LayerNorm.

The computation pattern is:
  1. gelu(in_2)          -- in_2 shape: [1, C, H, W]
  2. flatten(2)          -- -> [1, C, H*W]
  3. transpose(1, 2)     -- -> [1, H*W, C]
  4. contiguous()        -- materialize
  5. in_3 + result       -- residual add; in_3 shape: [1, H*W, C]
  6. permute(0,2,1)      -- -> [1, C, H*W] (same values as step 4, different strides)
  7. view(1,C,H,W)       -- no-op on data
  8. view(1,C,-1)        -- no-op on data
  9. permute(0,2,1)      -- -> [1, H*W, C] (non-contiguous, same data as step 4)
 10. layer_norm(tmp_10, (C,), in_1, in_0, 1e-6)
 11. view(1,H,W,C)

Fused kernel:
  - For each (batch, spatial) pair:
      out[i,j] = layer_norm(gelu(in_2[j,i]) + in_3[i,j])
  where i indexes the spatial dimension (N positions) and j indexes channels (C).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32},  num_warps=1),
        triton.Config({'BLOCK_C': 32},  num_warps=2),
        triton.Config({'BLOCK_C': 32},  num_warps=4),
        triton.Config({'BLOCK_C': 64},  num_warps=2),
        triton.Config({'BLOCK_C': 64},  num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 512}, num_warps=8),
    ],
    key=['C', 'N'],
)
@triton.jit
def fused_gelu_add_layernorm_kernel(
    in2_ptr,          # [1, C, H, W]  -- treated as [C, N] contiguous
    in3_ptr,          # [1, N, C]     -- contiguous
    ln_w_ptr,         # [C]
    ln_b_ptr,         # [C]
    out0_ptr,         # [1, N, C]  -- tmp_10
    out1_ptr,         # [1, N, C]  -- tmp_12 (after layernorm)
    C,                # number of channels (layer-norm dim)
    N,                # H * W (spatial size)
    eps,              # layer-norm epsilon
    BLOCK_C: tl.constexpr,
):
    """One program per (batch, spatial) position."""
    pid = tl.program_id(0)          # ranges over [0, N*B)
    b = pid // N
    i = pid % N

    c_offs = tl.arange(0, BLOCK_C)
    mask   = c_offs < C

    # ---- Load in_3: layout [B, N, C] ----
    # in3_ptr[b * N * C + i * C + c_offs]  -- contiguous along c
    in3 = tl.load(in3_ptr + (b * N + i) * C + c_offs, mask=mask, other=0.0)

    # ---- Load in_2: layout [B, C, N] ----
    # in2_ptr[b * C * N + c * N + i]       -- stride-N along c
    in2 = tl.load(in2_ptr + (b * C + c_offs) * N + i, mask=mask, other=0.0)

    # ---- GELU (exact, erf-based) ----
    # Upcast to fp32 for numerical stability
    in2_f32 = in2.to(tl.float32)
    gelu_out = 0.5 * in2_f32 * (1.0 + tl.math.erf(in2_f32 * 0.7071067811865476))

    # ---- Residual add ----
    out_f32 = gelu_out + in3.to(tl.float32)

    # ---- Store tmp_10 (shape [1, N, C], same as in3) ----
    tl.store(out0_ptr + (b * N + i) * C + c_offs, out_f32.to(in3.dtype), mask=mask)

    # ---- LayerNorm over C dimension ----
    # Mean
    mean = tl.sum(out_f32, axis=0) / C
    diff = out_f32 - mean
    # Variance
    var  = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + eps)
    norm = diff * inv_std

    # Scale + shift
    ln_w = tl.load(ln_w_ptr + c_offs, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + c_offs, mask=mask, other=0.0).to(tl.float32)
    out_ln_f32 = norm * ln_w + ln_b

    # ---- Store tmp_12 (same layout [1, N, C]) ----
    tl.store(out1_ptr + (b * N + i) * C + c_offs, out_ln_f32.to(in3.dtype), mask=mask)


@torch.fx.wrap
def run_fused_gelu_add_layernorm(in_0, in_1, in_2, in_3):
    """
    in_0 : bias  [C]        (layer-norm bias)
    in_1 : weight [C]       (layer-norm weight)
    in_2 : [1, C, H, W]     (input to GELU)
    in_3 : [1, N, C]        (residual add input)

    Returns (tmp_10, tmp_12) matching the original model output.
    """
    C = in_2.shape[1]
    N = in_2.shape[2] * in_2.shape[3]   # H * W

    out0 = torch.empty_like(in_3)        # tmp_10
    out1 = torch.empty_like(in_3)        # tmp_12

    B = 1                                # batch size (always 1 in these graphs)
    grid = (B * N,)

    fused_gelu_add_layernorm_kernel[grid](
        in_2, in_3, in_1, in_0,   # in2, in3, ln_weight, ln_bias
        out0, out1,
        C, N, 1e-6,
    )

    return (out0, out1)