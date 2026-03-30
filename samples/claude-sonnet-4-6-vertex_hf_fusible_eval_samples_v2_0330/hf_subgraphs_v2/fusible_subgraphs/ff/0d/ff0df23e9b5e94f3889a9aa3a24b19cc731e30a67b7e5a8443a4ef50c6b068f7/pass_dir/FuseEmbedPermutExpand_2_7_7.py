"""
Fused pass: embedding + permute([2,0,1]) + unsqueeze(0) + expand((2,-1,7,7)) + contiguous
Handles float32 all-mpnet-base-v2 model.

Output formula: output[b, d, i, j] = weight[indices[i, j], d]
where b=2 (batch), d in [0, embed_dim), i,j in [0, 7)
"""
import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _fused_embed_kernel_2_7_7(
    weight_ptr,    # [num_embed, embed_dim]  float32
    indices_ptr,   # [7, 7]  int64, on CUDA
    output_ptr,    # [2, embed_dim, 7, 7]  float32
    embed_dim,     # runtime: embedding feature size
    SEQ_H: tl.constexpr,  # 7
    SEQ_W: tl.constexpr,  # 7
    BATCH: tl.constexpr,  # 2
    BLOCK: tl.constexpr,
):
    total_elements = BATCH * embed_dim * SEQ_H * SEQ_W
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    # Decode flat index → (b, d, i, j) in [BATCH, embed_dim, SEQ_H, SEQ_W] layout
    j = offsets % SEQ_W
    tmp = offsets // SEQ_W
    i = tmp % SEQ_H
    tmp = tmp // SEQ_H
    d = tmp % embed_dim
    # b = tmp // embed_dim  (output same for all batches, computed automatically by tiling)

    # Gather embedding index — same for all batches
    idx = tl.load(indices_ptr + i * SEQ_W + j, mask=mask, other=0)

    # Load embedding value: weight[idx, d]
    val = tl.load(weight_ptr + idx * embed_dim + d, mask=mask, other=0.0)

    # Write directly to output
    tl.store(output_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def _fused_embed_wrapper_2_7_7(in_0, in_1):
    """
    in_0: weight [num_embed, embed_dim]  (may be on CPU or CUDA, float32)
    in_1: indices [7, 7]  int64  (on CPU)
    returns: [2, embed_dim, 7, 7]  same dtype as in_0
    """
    in_0_cuda = in_0.cuda() if not in_0.is_cuda else in_0
    in_1_cuda = in_1.cuda() if not in_1.is_cuda else in_1

    BATCH = 2
    SEQ_H = 7
    SEQ_W = 7
    embed_dim = in_0_cuda.shape[1]
    total_elements = BATCH * embed_dim * SEQ_H * SEQ_W

    output = torch.empty((BATCH, embed_dim, SEQ_H, SEQ_W),
                         dtype=in_0_cuda.dtype, device='cuda')

    BLOCK = 256
    grid = ((total_elements + BLOCK - 1) // BLOCK,)

    _fused_embed_kernel_2_7_7[grid](
        in_0_cuda, in_1_cuda, output,
        embed_dim,
        SEQ_H=SEQ_H, SEQ_W=SEQ_W, BATCH=BATCH,
        BLOCK=BLOCK,
    )
    return output


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_1 = in_1.to(device(type='cuda'))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((2, -1, 7, 7))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _fused_embed_wrapper_2_7_7