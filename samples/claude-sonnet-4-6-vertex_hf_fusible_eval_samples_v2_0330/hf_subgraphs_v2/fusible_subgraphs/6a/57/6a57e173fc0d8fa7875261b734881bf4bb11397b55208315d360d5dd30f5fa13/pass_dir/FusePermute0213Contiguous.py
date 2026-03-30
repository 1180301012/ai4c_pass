import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: permute(0, 2, 1, 3)  +  contiguous()
# ---------------------------------------------------------------------------

def pattern(x):
    t = x.permute(0, 2, 1, 3)
    c = t.contiguous()
    return c


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: transpose [B, G, S, H] → [B, S, G, H] (contiguous output)
#
# Strategy: iterate over output elements in flat order.  Each output element
# at flat index `out_flat` maps to coordinates (b, s, g, h) in the output
# [B, S, G, H] layout and reads from (b, g, s, h) in the input [B, G, S, H]
# layout.  For H >= warp-width the innermost H elements are contiguous in
# both input and output, giving perfectly coalesced access.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['G', 'S', 'H'],
)
@triton.jit
def _permute_0213_kernel(
    x_ptr, out_ptr,
    G, S, H,
    SGH,   # S * G * H  (output stride for dim B)
    GH,    # G * H      (output stride for dim S)
    SH,    # S * H      (input  stride for dim G)
    GSH,   # G * S * H  (input  stride for dim B)
    N,     # total elements  B * G * S * H
    BLOCK_SIZE: tl.constexpr,
):
    pid      = tl.program_id(0)
    offsets  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < N

    # ---- Decode output flat index → (b, s, g, h) -------------------------
    # Output layout: [B, S, G, H]  contiguous
    h_o = offsets % H
    g_o = (offsets // H) % G
    s_o = (offsets // GH) % S
    b_o = offsets // SGH

    # ---- Input flat index ← (b, g, s, h) in [B, G, S, H] contiguous ------
    in_idx = b_o * GSH + g_o * SH + s_o * H + h_o

    val = tl.load(x_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def permute_0213_contiguous(x):
    """Replace permute(0,2,1,3) + contiguous() with a single Triton kernel."""
    B, G, S, H = x.shape
    N = x.numel()

    out = torch.empty((B, S, G, H), dtype=x.dtype, device=x.device)

    GH  = G * H
    SH  = S * H
    SGH = S * G * H
    GSH = G * S * H

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _permute_0213_kernel[grid](
        x, out,
        G, S, H,
        SGH, GH, SH, GSH,
        N,
    )

    return out


# ---------------------------------------------------------------------------
# Required by the evaluation framework
# ---------------------------------------------------------------------------

def replacement_func():
    return permute_0213_contiguous