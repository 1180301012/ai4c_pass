"""
Fused pass: to(cuda,index=0) + embedding + permute([2,0,1]) + unsqueeze(0)
Replaces the subgraph and returns [1, D, S, S] contiguous.
The remaining expand((1,-1,S,S)) + contiguous() in the graph become no-ops.
Matches 4/5 target graphs (all that use device(type='cuda', index=0)).
"""
import torch
from torch import device
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
# Matches the common prefix of 4 graphs (S=45 × 2, S=11 × 2).
# NOTE: does NOT include expand or contiguous — those run as no-ops on our output.
def pattern(in_0, in_1):
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Flat 1D kernel: output is stored in [D, SS] layout where flat index = d*SS + ij.
# Works for any (D, S): D and SS are passed as runtime values; only BLOCK_SIZE is constexpr.
@triton.jit
def _fused_emb_permute_kernel(
    weight_ptr,    # [V, D] contiguous, any dtype
    indices_ptr,   # [SS]   contiguous int64  (already on CUDA)
    output_ptr,    # [D, SS] contiguous — flat: offs = d*SS + ij
    D,             # embedding dim (runtime)
    SS,            # S*S (runtime)
    N,             # D*SS (runtime)
    weight_stride, # stride(0) of weight = D
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Decompose flat index: d = offs // SS, ij = offs % SS
    d  = offs // SS   # embedding dimension (0..D-1)
    ij = offs % SS    # position in S×S grid

    # Load index (sequential within block → coalesced)
    idx = tl.load(indices_ptr + ij, mask=mask, other=0)

    # Gather from weight: weight[idx, d] — small table (32×D), fully cached in L1
    val = tl.load(weight_ptr + idx * weight_stride + d, mask=mask, other=0.0)

    # Store directly to transposed output layout (offs = d*SS + ij already)
    tl.store(output_ptr + offs, val, mask=mask)


# ── Wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_embedding_permute(weight, indices):
    """
    weight  : [V, D]   cuda
    indices : [S, S]   cpu, int64
    returns : [1, D, S, S] cuda, contiguous — expand+contiguous that follow are no-ops
    """
    cuda_dev = weight.device if weight.is_cuda else 'cuda:0'
    if not weight.is_cuda:
        weight = weight.to(cuda_dev)
    indices_cuda = indices.to(cuda_dev)

    V, D = weight.shape
    S  = indices.shape[0]   # works for S=45 and S=11
    SS = S * S
    N  = D * SS

    # [D, SS] layout: output[d, ij] at flat index d*SS + ij
    output_flat = torch.empty((D, SS), dtype=weight.dtype, device=cuda_dev)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _fused_emb_permute_kernel[grid](
        weight, indices_cuda, output_flat,
        D, SS, N,
        weight.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Zero-copy reshape to [1, D, S, S] — expand+contiguous that follow are no-ops
    return output_flat.view(1, D, S, S)


# ── Registration ──────────────────────────────────────────────────────────────
def replacement_func():
    return fused_embedding_permute