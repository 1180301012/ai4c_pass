"""
Shared Triton kernel and dispatch wrapper for:
  embedding + permute([2,0,1]) + unsqueeze(0) + expand((B,-1,S,S)) + contiguous

Fused output: output[b, d, i, j] = weight[indices[i, j], d]

Flat 1D design: grid = (ceil(B*D*S*S / BLOCK_SIZE),)
  - Output viewed as flat [B*D*S*S]; stores are coalesced
  - All of D, S, B are tl.constexpr so divisions use compiler magic-multiply
  - Tiny weight table (32*D <= 768 bytes) fits entirely in L1 cache
  - BLOCK_SIZE=256 gives a reasonable grid for all sizes
"""
import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 256


@triton.jit
def fused_emb_flat_kernel(
    indices_ptr,              # int64 [S, S]
    weight_ptr,               # float [N, D]
    output_ptr,               # float [B*D*S*S]
    D: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid: (ceil(B*D*S*S / BLOCK_SIZE),)
    flat_index = b*(D*S*S) + d*(S*S) + ij
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    NiJ   = S * S
    TOTAL = B * D * NiJ
    mask  = offs < TOTAL

    b         = offs // (D * NiJ)
    remainder = offs - b * (D * NiJ)
    d         = remainder // NiJ
    ij        = remainder - d * NiJ

    idx = tl.load(indices_ptr + ij, mask=mask, other=0).to(tl.int32)
    w   = tl.load(weight_ptr  + idx * D + d, mask=mask, other=0.0)
    tl.store(output_ptr + offs, w, mask=mask)


@torch.fx.wrap
def fused_emb_dispatch(indices, weight, route):
    """
    Shared dispatch wrapper used by all three pass files.
    Pre-computes grid sizes to minimise Python overhead per call.
    """
    D = weight.shape[1]
    if route == "1_45":
        # B=1, S=45, D=4: TOTAL=8100, grid=(32,)
        out = torch.empty((1, D, 45, 45), dtype=weight.dtype, device=weight.device)
        fused_emb_flat_kernel[(32,)](
            indices, weight, out, D=D, S=45, B=1, BLOCK_SIZE=_BLOCK_SIZE)
        return out
    elif route == "1_11":
        # B=1, S=11, D=12: TOTAL=1452, grid=(6,)
        out = torch.empty((1, D, 11, 11), dtype=weight.dtype, device=weight.device)
        fused_emb_flat_kernel[(6,)](
            indices, weight, out, D=D, S=11, B=1, BLOCK_SIZE=_BLOCK_SIZE)
        return out
    else:  # "2_7"
        # B=2, S=7, D=12: TOTAL=1176, grid=(5,)
        out = torch.empty((2, D, 7, 7), dtype=weight.dtype, device=weight.device)
        fused_emb_flat_kernel[(5,)](
            indices, weight, out, D=D, S=7, B=2, BLOCK_SIZE=_BLOCK_SIZE)
        return out