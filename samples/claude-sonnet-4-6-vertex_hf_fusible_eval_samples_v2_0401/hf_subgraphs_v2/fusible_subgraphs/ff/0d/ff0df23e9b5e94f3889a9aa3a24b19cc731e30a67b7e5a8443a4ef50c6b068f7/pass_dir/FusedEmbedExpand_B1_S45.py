"""
Fused pass: embedding + permute([2,0,1]) + unsqueeze(0) + expand((1,-1,45,45)) + contiguous
Targets tiny MPNet models with S=45, B=1, D=4
Output: [1, D, 45, 45] where output[0, d, i, j] = weight[indices[i,j], d]
"""
import torch
from torch import device
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Flat 1D kernel: output[d, ij] stored at flat index offs = d*SS + ij.
# Grid = (ceil(D*SS / BLOCK_SIZE),) — gives ~32 CTAs with good occupancy.
# Index loads are coalesced; weight (256 bytes total) fits entirely in L1.
@triton.jit
def _fused_emb_expand_B1_kernel(
    weight_ptr,    # [V, D] contiguous float16/bf16
    indices_ptr,   # [SS]   contiguous int64  (already on CUDA)
    output_ptr,    # [D, SS] contiguous  (flat index = d*SS + ij = offs)
    D,             # embedding dim (4)
    SS,            # S*S = 45*45 = 2025
    N,             # D*SS = total output elements
    weight_stride, # stride(0) of weight = D
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Decompose flat index: d = offs // SS, ij = offs % SS
    d  = offs // SS    # embedding dimension index
    ij = offs % SS     # position in S×S grid (consecutive within block)

    # Index load (coalesced when ij is consecutive)
    idx = tl.load(indices_ptr + ij, mask=mask, other=0)

    # Weight gather: weight[idx, d] — at most 32 distinct rows, all fit in L1
    val = tl.load(weight_ptr + idx * weight_stride + d, mask=mask, other=0.0)

    # Sequential output store (offs is already the correct flat address)
    tl.store(output_ptr + offs, val, mask=mask)


# ── Wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_embedding_expand_B1_S45(weight, indices):
    cuda_dev = weight.device if weight.is_cuda else 'cuda:0'
    if not weight.is_cuda:
        weight = weight.to(cuda_dev)
    indices_cuda = indices.to(cuda_dev)

    V, D = weight.shape
    S  = 45
    SS = S * S          # 2025
    N  = D * SS         # 8100

    # Allocate as [D, SS] so output flat index = d*SS + ij = offs
    output_flat = torch.empty((D, SS), dtype=weight.dtype, device=cuda_dev)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _fused_emb_expand_B1_kernel[grid](
        weight, indices_cuda, output_flat,
        D, SS, N,
        weight.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape to [1, D, S, S] — zero-copy view
    return output_flat.view(1, D, S, S)


# ── Registration ──────────────────────────────────────────────────────────────
def replacement_func():
    return fused_embedding_expand_B1_S45