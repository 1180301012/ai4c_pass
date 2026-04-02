import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1, in_4):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    return tmp_13


def replacement_args(in_0, in_1, in_4):
    return (in_0, in_1, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 16},   num_warps=1, num_stages=1),
        triton.Config({'BLOCK_D': 32},   num_warps=1, num_stages=2),
        triton.Config({'BLOCK_D': 64},   num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 128},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_D': 1024}, num_warps=16, num_stages=3),
    ],
    key=['D'],
)
@triton.jit
def fused_embedding_add_kernel(
    in_0_ptr,   # [B*N, D] input embeddings
    in_1_ptr,   # [V, D] position embedding table
    in_4_ptr,   # [N] cache positions (int64)
    out_ptr,    # [B*N, D] output
    D,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    pos_idx = tl.load(in_4_ptr + pid)
    emb_idx = pos_idx + 2

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    row_base_0 = pid * D
    row_base_1 = emb_idx * D

    x0 = tl.load(in_0_ptr + row_base_0 + d_offsets, mask=d_mask, other=0.0)
    x1 = tl.load(in_1_ptr + row_base_1 + d_offsets, mask=d_mask, other=0.0)

    tl.store(out_ptr + row_base_0 + d_offsets, x0 + x1, mask=d_mask)


@torch.fx.wrap
def fused_embedding_add(in_0, in_1, in_4):
    B, N, D = in_0.shape
    BN = B * N
    out = torch.empty_like(in_0)
    grid = (BN,)
    fused_embedding_add_kernel[grid](
        in_0.view(BN, D),
        in_1,
        in_4,
        out.view(BN, D),
        D=D,
    )
    return out


def replacement_func():
    return fused_embedding_add