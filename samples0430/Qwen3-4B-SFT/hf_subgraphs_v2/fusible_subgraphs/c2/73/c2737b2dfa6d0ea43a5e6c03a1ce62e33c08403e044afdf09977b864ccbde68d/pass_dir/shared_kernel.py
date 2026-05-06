"""
Shared kernel and wrapper for fused divide + last-2-dim transpose.

Pattern:
    tmp_0 = in_0 / SCALAR
    tmp_1 = tmp_0.transpose(-1, -2)

Optimisation: single Triton kernel that reads in_0 once, applies the scale,
writes directly into a contiguous out tensor with the transposed layout.

Memory access:
    Input  layout [B, H, S, D]: D is the innermost (stride-1) dimension
    → loads are perfectly coalesced over consecutive offs.
    Output layout [B, H, D, S]: S is the innermost dimension.
    Per-program writes coalesce when D >= warp_size (D=64 → full cache-line).

    bh = offs // (S * D) = offs // BHDS   (BHDS = B*H*D, scalar, one div total)
    out[bh, d, s] = bh * HDS + d * S + s
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_transpose_kernel(
    in_ptr,
    out_ptr,
    n,                      # B*H*S*D  (total elements, for mask)
    S,                      # runtime sequence length (for s_idx)
    HDS,                    # H * D * S (output stride for bh)
    BHDS,                   # B * H * D (bh = offs // BHDS)
    D: tl.constexpr,        # head-dim — compile-time const for masks & D!=0
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n          # simple comparison with a single scalar n

    # Decode [B,H,S,D] coordinates from consecutive input offset
    # D is innermost → d = offs % D, s = (offs//D) % S, bh = offs // BHDS
    d_idx = offs % D
    tmp   = offs // D
    s_idx = tmp % S
    bh    = offs // BHDS     # = offs // (B*H*D)

    # Load — perfectly coalesced (consecutive offs)
    x = tl.load(in_ptr + offs, mask=mask)
    x = x * (1.0 / 1.6817928305074292)

    # Write to [B,H,D,S] — coalesced when D is a cache-line multiple
    out_off = bh * HDS + d_idx * S + s_idx
    tl.store(out_ptr + out_off, x, mask=mask)


@torch.fx.wrap
def fused_scale_transpose_1_68(in_0):
    B, H, S, D = in_0.shape
    n    = B * H * S * D
    BHDS = B * H * D         # bh = offs // BHDS
    HDS  = H * D * S
    out  = torch.empty((B, H, D, S), dtype=in_0.dtype, device=in_0.device)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _scale_transpose_kernel[grid](
        in_0, out, n, S, HDS, BHDS, D, BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_scale_transpose_1_68