import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: permute(0,2,1,3) followed by contiguous()
# Matches all graph variants regardless of tensor shape.
# The view() that follows is shape-specific but is O(1) metadata op, so
# we leave it outside the pattern – the framework will execute it on our
# contiguous output automatically.
# ---------------------------------------------------------------------------

def pattern(x):
    tmp = x.permute(0, 2, 1, 3)
    out = tmp.contiguous()
    return out


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel v2: 2D grid over (B*G, S-tiles).
#
# Strategy: read input [B, G, S, D] tile-by-tile along (b, g) then blocks
# of S rows.  For a fixed (b, g), the D elements per S-row are contiguous
# in input, giving coalesced loads.  Output [B, S, G, D] writes are also
# nearly coalesced for reasonable BLOCK_D.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- configs for D=8 (groups=4, head_dim=8) ----------------------
        triton.Config({'BLOCK_S':  16, 'BLOCK_D': 8}, num_warps=1),
        triton.Config({'BLOCK_S':  32, 'BLOCK_D': 8}, num_warps=2),
        triton.Config({'BLOCK_S':  64, 'BLOCK_D': 8}, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_D': 8}, num_warps=4),
        # ---- configs for D=64 (groups=12, head_dim=64) -------------------
        triton.Config({'BLOCK_S':  2, 'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_S':  4, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_S':  8, 'BLOCK_D': 64}, num_warps=4),
    ],
    key=['D'],   # Key only on D; avoids expensive re-tuning for each (G,S,D) combo
)
@triton.jit
def _permute_0213_2d_kernel(
    src_ptr,
    dst_ptr,
    G,
    S,
    D,
    SD,   # S * D
    GD,   # G * D
    GSD,  # G * S * D  (= S * G * D, same value)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (B*G,  cdiv(S, BLOCK_S),  cdiv(D, BLOCK_D))
    Each CTA transposes a BLOCK_S × BLOCK_D tile for a single (b, g) pair.

    Input [B, G, S, D]:   src[b, g, s, d] = b*GSD + g*SD + s*D + d
    Output [B, S, G, D]:  dst[b, s, g, d] = b*GSD + s*GD + g*D + d
    """
    pid_bg = tl.program_id(0)    # linear (b, g) index
    pid_s  = tl.program_id(1)    # S-tile index
    pid_d  = tl.program_id(2)    # D-tile index

    g = pid_bg % G
    b = pid_bg // G

    s_start = pid_s * BLOCK_S
    d_start = pid_d * BLOCK_D

    s_ids = s_start + tl.arange(0, BLOCK_S)   # (BLOCK_S,)
    d_ids = d_start + tl.arange(0, BLOCK_D)   # (BLOCK_D,)

    s_mask = s_ids < S
    d_mask = d_ids < D

    # 2-D masks / offsets  (BLOCK_S, BLOCK_D)
    mask = s_mask[:, None] & d_mask[None, :]

    # src[b, g, s_ids, d_ids]
    src_off = b * GSD + g * SD + s_ids[:, None] * D + d_ids[None, :]
    # dst[b, s_ids, g, d_ids]
    dst_off = b * GSD + s_ids[:, None] * GD + g * D + d_ids[None, :]

    vals = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_off, vals, mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_permute_0213_contiguous(x):
    B, G, S, D = x.shape
    out = torch.empty((B, S, G, D), dtype=x.dtype, device=x.device)

    def grid(meta):
        return (
            B * G,
            triton.cdiv(S, meta['BLOCK_S']),
            triton.cdiv(D, meta['BLOCK_D']),
        )

    _permute_0213_2d_kernel[grid](
        x, out,
        G, S, D,
        S * D,   # SD
        G * D,   # GD
        G * S * D,  # GSD
    )
    return out


def replacement_func():
    return fused_permute_0213_contiguous