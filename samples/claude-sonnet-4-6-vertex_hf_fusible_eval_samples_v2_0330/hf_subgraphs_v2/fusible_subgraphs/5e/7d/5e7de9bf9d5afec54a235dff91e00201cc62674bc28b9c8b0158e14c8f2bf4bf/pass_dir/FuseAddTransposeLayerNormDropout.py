"""
Fused pass: add + transpose(1,2) + layer_norm((768,), weight, bias, 1e-5) + dropout(training=False).

Model graph context (B=1, C=768, L=124):
  tmp_6  : [1, 768, 124]  contiguous         (strides 95232, 124, 1)
  tmp_7  : [1, 768, 124]  non-contiguous     (strides 96000, 125, 1 — slice of [1,768,125])

TILE_L optimisation (no .contiguous() needed):
  Both inputs have sx_l = sy_l = 1: consecutive L values for the same channel are
  ALWAYS adjacent in memory regardless of contiguity.
  One program handles TILE_L consecutive time-steps:
    iteration i=0: loads x[b,c,l_start+0] → brings cache line (32 elements) into L1
    iteration i=1: loads x[b,c,l_start+1] → L1 HIT (same cache line)
    iteration i=2: loads x[b,c,l_start+2] → L1 HIT
    ...
  TILE_L× fewer L2 transactions; weight/bias amortised over TILE_L steps.
  Grid adapts via lambda to the autotuned TILE_L.
  No .contiguous() copy → avoids the ~0.3 µs GPU copy overhead per call.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match in the computation graph
# ---------------------------------------------------------------------------

def pattern(tmp_6, tmp_7, in_1, in_0):
    """Matches: add → transpose(1,2) → layer_norm → dropout(inference)."""
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return tmp_11


def replacement_args(tmp_6, tmp_7, in_1, in_0):
    return (tmp_6, tmp_7, in_1, in_0)


# ---------------------------------------------------------------------------
# Triton kernel with TILE_L (L1 cache reuse across time-steps)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 1}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 2}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 4}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 8}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 1}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 2}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 4}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_C': 1024, 'TILE_L': 8}, num_warps=8,  num_stages=1),
    ],
    key=['C', 'L'],
)
@triton.jit
def _fused_add_trans_ln_kernel(
    x_ptr, y_ptr,          # [B, C, L]  arbitrary strides; sx_l = sy_l = 1
    out_ptr,               # [B, L, C]  contiguous
    weight_ptr, bias_ptr,  # [C]
    B, C, L,
    sx_b, sx_c, sx_l,     # strides of x
    sy_b, sy_c, sy_l,     # strides of y
    BLOCK_C: tl.constexpr,
    TILE_L: tl.constexpr,  # consecutive time-steps per program
):
    pid     = tl.program_id(0)
    n_per_b = tl.cdiv(L, TILE_L)
    b       = pid // n_per_b
    l_tile  = pid  % n_per_b
    l_start = l_tile * TILE_L

    c_idx  = tl.arange(0, BLOCK_C)
    c_mask = c_idx < C

    # Weight / bias loaded ONCE per program (reused across TILE_L time-steps)
    w  = tl.load(weight_ptr + c_idx, mask=c_mask, other=1.0).to(tl.float32)
    bp = tl.load(bias_ptr   + c_idx, mask=c_mask, other=0.0).to(tl.float32)

    # Compile-time unroll over TILE_L time-steps
    for i in tl.static_range(TILE_L):
        l = l_start + i
        eff_mask = c_mask & (l < L)

        x_base   = b * sx_b + l * sx_l
        y_base   = b * sy_b + l * sy_l
        out_base = (b * L + l) * C

        # sx_l = sy_l = 1 → consecutive l for same c are adjacent in memory
        # → iterations i>0 hit the same L1 cache line loaded by i=0
        x_raw = tl.load(x_ptr + x_base + c_idx * sx_c, mask=eff_mask, other=0.0)
        y_raw = tl.load(y_ptr + y_base + c_idx * sy_c, mask=eff_mask, other=0.0)

        z = x_raw.to(tl.float32) + y_raw.to(tl.float32)

        mean    = tl.sum(z, axis=0) / C
        diff    = tl.where(c_mask, z - mean, 0.0)
        var     = tl.sum(diff * diff, axis=0) / C
        inv_std = 1.0 / tl.sqrt(var + 1e-5)
        z_norm  = tl.where(c_mask, diff * inv_std, 0.0)

        out = z_norm * w + bp
        tl.store(out_ptr + out_base + c_idx, out.to(x_raw.dtype), mask=eff_mask)


# ---------------------------------------------------------------------------
# Python wrapper (no .contiguous() – avoids the ~0.3 µs GPU copy overhead)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_transpose_layernorm(tmp_6, tmp_7, in_1, in_0):
    """
    Fused replacement:
        add + transpose(1,2) + layer_norm((768,), weight, bias, 1e-5) + dropout(inference)

    Inputs are passed directly without .contiguous(): both have sl=1 so the
    TILE_L cache-reuse optimisation applies regardless of contiguity.
    """
    B = tmp_6.shape[0]
    C = tmp_6.shape[1]
    L = tmp_6.shape[2]
    out = torch.empty((B, L, C), dtype=tmp_6.dtype, device=tmp_6.device)

    # Grid size adapts to the TILE_L chosen by autotune
    grid = lambda meta: (B * triton.cdiv(L, meta['TILE_L']),)

    _fused_add_trans_ln_kernel[grid](
        tmp_6, tmp_7,
        out,
        in_1, in_0,
        B, C, L,
        tmp_6.stride(0), tmp_6.stride(1), tmp_6.stride(2),
        tmp_7.stride(0), tmp_7.stride(1), tmp_7.stride(2),
    )

    return out


# ---------------------------------------------------------------------------
# Required by the AI4C pass framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_add_transpose_layernorm