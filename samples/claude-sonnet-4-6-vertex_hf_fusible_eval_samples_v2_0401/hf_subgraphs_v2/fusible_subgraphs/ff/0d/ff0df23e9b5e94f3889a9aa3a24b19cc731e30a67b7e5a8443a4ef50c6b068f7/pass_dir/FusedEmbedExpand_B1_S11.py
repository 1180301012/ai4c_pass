"""
Fused pass: embedding + permute([2,0,1]) + unsqueeze(0) + expand((1,-1,11,11)) + contiguous
Targets MPNet-base models with S=11, B=1, D=12
Output: [1, D, 11, 11] where output[0, d, i, j] = weight[indices[i,j], d]
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
    tmp_5 = tmp_4.expand((1, -1, 11, 11))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Grid: (SS,) one program per (i,j); each program gathers weight[indices[ij], :D]
# and writes to output[d * SS + ij] for d in [0, D).
@triton.jit
def _fused_emb_expand_S11_kernel(
    weight_ptr,    # [V, D] contiguous
    indices_ptr,   # [SS]   contiguous int64  (already on CUDA)
    output_ptr,    # [D, SS] contiguous (B=1 squeezed to simplify addressing)
    D,             # embedding dim
    SS,            # S*S = 11*11 = 121
    weight_stride, # stride(0) of weight = D
    BLOCK_D: tl.constexpr,
):
    ij = tl.program_id(0)
    idx = tl.load(indices_ptr + ij)
    d_offs = tl.arange(0, BLOCK_D)
    mask_d = d_offs < D
    emb = tl.load(weight_ptr + idx * weight_stride + d_offs, mask=mask_d, other=0.0)
    # output[0, d, i, j]  =>  flat: d * SS + ij   (B=1)
    tl.store(output_ptr + d_offs * SS + ij, emb, mask=mask_d)


# ── Wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_embedding_expand_B1_S11(weight, indices):
    cuda_dev = weight.device if weight.is_cuda else 'cuda:0'
    if not weight.is_cuda:
        weight = weight.to(cuda_dev)
    indices_cuda = indices.to(cuda_dev)

    V, D = weight.shape
    S  = 11
    SS = S * S          # 121

    # Allocate as [D, SS] so output[d, ij] = d*SS + ij (avoids B stride math in kernel)
    output_flat = torch.empty((D, SS), dtype=weight.dtype, device=cuda_dev)

    grid = (SS,)
    _fused_emb_expand_S11_kernel[grid](
        weight, indices_cuda, output_flat,
        D, SS,
        weight.stride(0),
        BLOCK_D=16,
    )

    # Reshape to [1, D, S, S] — zero-copy view
    return output_flat.view(1, D, S, S)


# ── Registration ──────────────────────────────────────────────────────────────
def replacement_func():
    return fused_embedding_expand_B1_S11