"""
Optimization Pass: Fuse hardsigmoid + elementwise_mul + adaptive_avg_pool2d + flatten + dropout

Pattern matched:
    conv_out: [B, C, 1, 1]
    in_2:     [B, C, H, W]
    hs   = hardsigmoid(conv_out)          → [B, C, 1, 1]
    prod = in_2 * hs                       → [B, C, H, W]  (broadcast)
    pool = adaptive_avg_pool2d(prod, 1)    → [B, C, 1, 1]
    flat = pool.flatten(1, -1)             → [B, C]
    out  = dropout(flat, 0.0, False, False)→ [B, C]  (identity)

Key insight:
    out[b,c] = hardsigmoid(conv_out[b,c,0,0]) * mean(in_2[b,c,:,:])

Multi-channel kernel design:
    Grid: (B * (C // TILE_C),)    — TILE_C channels processed per program
    TILE_C is autotuned to reduce wave-count (fewer, larger programs ⟹
    far less GPU scheduler overhead than one program per (b,c) pair).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(conv_out, in_2):
    hs   = torch.nn.functional.hardsigmoid(conv_out, False)
    prod = in_2 * hs
    pool = torch.nn.functional.adaptive_avg_pool2d(prod, 1)
    flat = pool.flatten(1, -1)
    out  = torch.nn.functional.dropout(flat, 0.0, False, False)
    return out


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


# ---------------------------------------------------------------------------
# Triton kernel  (multi-channel: TILE_C channels per program)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'TILE_C':   1, 'num_warps': 1}),
        triton.Config({'TILE_C':   1, 'num_warps': 2}),
        triton.Config({'TILE_C':   4, 'num_warps': 2}),
        triton.Config({'TILE_C':   4, 'num_warps': 4}),
        triton.Config({'TILE_C':   8, 'num_warps': 2}),
        triton.Config({'TILE_C':   8, 'num_warps': 4}),
        triton.Config({'TILE_C':  16, 'num_warps': 2}),
        triton.Config({'TILE_C':  16, 'num_warps': 4}),
        triton.Config({'TILE_C':  32, 'num_warps': 4}),
        triton.Config({'TILE_C':  32, 'num_warps': 8}),
        triton.Config({'TILE_C':  64, 'num_warps': 4}),
        triton.Config({'TILE_C':  64, 'num_warps': 8}),
        triton.Config({'TILE_C': 128, 'num_warps': 8}),
    ],
    key=['B', 'C', 'HW'],   # include B so configs adapt to batch size
)
@triton.jit
def _hardsigmoid_mul_avgpool_kernel(
    conv_ptr,               # [B*C]    – conv output (1×1 spatial, flattened)
    x_ptr,                  # [B*C*HW] – in_2 flattened
    out_ptr,                # [B*C]    – output in orig dtype
    B,                      # int: batch size (autotune key — not used in body)
    C,                      # int: number of channels
    HW,                     # int: H * W
    inv_HW,                 # float: 1.0 / HW
    TILE_C:   tl.constexpr, # channels processed per program (autotuned)
    BLOCK_HW: tl.constexpr, # >= HW; power-of-2 tile for the HW reduction
):
    """
    Each Triton program processes TILE_C channels for one batch index.
    Grid size: B * (C // TILE_C)
    """
    prog_idx    = tl.program_id(0)
    num_c_tiles = C // TILE_C
    b           = prog_idx // num_c_tiles
    c_base      = (prog_idx % num_c_tiles) * TILE_C

    # Unrolled by the compiler because TILE_C is constexpr
    for ci in range(TILE_C):
        bc = b * C + c_base + ci

        # ---- hardsigmoid(conv_out[b, c]) ----
        conv_val = tl.load(conv_ptr + bc).to(tl.float32)
        hs = tl.minimum(tl.maximum(conv_val + 3.0, 0.0), 6.0) * (1.0 / 6.0)

        # ---- sum in_2[b, c, :, :] over HW (single masked load) ----
        base    = bc * HW
        offsets = tl.arange(0, BLOCK_HW)
        mask    = offsets < HW
        x_vals  = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

        # result = hardsigmoid * mean(in_2[b,c])
        # tl.store auto-converts fp32 → out_ptr's native dtype (fp16/bf16/fp32)
        result = tl.sum(x_vals, axis=0) * hs * inv_HW
        tl.store(out_ptr + bc, result)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Wrapper  (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def hardsigmoid_mul_avgpool_flatten(conv_out, in_2):
    """
    Fused replacement for:
        hardsigmoid(conv_out) * in_2  →  avg_pool  →  flatten  →  dropout

    conv_out : [B, C, 1, 1]
    in_2     : [B, C, H, W]
    returns  : [B, C]  (same dtype as in_2)

    Uses a multi-channel Triton kernel (TILE_C channels per program) so the
    grid is B*(C//TILE_C), far fewer programs than the naive B*C approach.
    This dramatically cuts GPU scheduling / wave-dispatch overhead.
    """
    B, C, H, W = in_2.shape
    HW         = H * W
    BLOCK_HW   = _next_power_of_2(HW)
    inv_HW     = 1.0 / HW
    orig_dtype = in_2.dtype

    conv_flat = conv_out.contiguous().reshape(B * C)
    in2_flat  = in_2.contiguous().reshape(B * C * HW)

    # Output in orig_dtype: Triton auto-converts fp32 result on store,
    # saving a separate dtype-conversion kernel launch.
    out = torch.empty(B * C, dtype=orig_dtype, device=in_2.device)

    # Grid is a lambda so it can adapt to the autotuned TILE_C value.
    _hardsigmoid_mul_avgpool_kernel[
        lambda meta: (B * (C // meta['TILE_C']),)
    ](
        conv_flat, in2_flat, out,
        B, C, HW, inv_HW,
        BLOCK_HW=BLOCK_HW,
    )

    return out.reshape(B, C)   # free view, no copy


# ---------------------------------------------------------------------------
# Replacement entry-point
# ---------------------------------------------------------------------------

def replacement_func():
    return hardsigmoid_mul_avgpool_flatten