import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern to match relative position embedding computation with N=512"""
    # The arange is computed inside the model, so it needs to be part of the pattern
    tmp_1 = torch.arange(512, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = in_1 - tmp_2
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    tmp_6 = torch.nn.functional.embedding(tmp_5, in_0, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_relative_position_embedding_kernel_512(
    position_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    N,
    embedding_dim,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Grid: (N, N) - each program handles one (i, j) position
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    # Load position_id for row i
    pos_id = tl.load(position_ids_ptr + i)
    
    # Compute embedding index: position_id - j + 2047
    idx = pos_id - j + 2047
    
    # Load entire embedding vector (vectorized)
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < embedding_dim
    
    embedding_vec = tl.load(
        embedding_weight_ptr + idx * embedding_dim + d_offsets,
        mask=d_mask,
        other=0.0
    )
    
    # Compute output offset and store
    out_base = i * N * embedding_dim + j * embedding_dim
    tl.store(output_ptr + out_base + d_offsets, embedding_vec, mask=d_mask)


@torch.fx.wrap
def fused_relative_position_embedding_512(embedding_weight, position_ids):
    N = position_ids.shape[0]
    embedding_dim = embedding_weight.shape[1]
    
    # Create output tensor
    output = torch.empty((N, N, embedding_dim), dtype=torch.float32, device=embedding_weight.device)
    
    # Use simple kernel with grid (N, N)
    BLOCK_SIZE_D = 64  # embedding_dim is 64
    grid = (N, N)
    
    fused_relative_position_embedding_kernel_512[grid](
        position_ids,
        embedding_weight,
        output,
        N,
        embedding_dim,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return output


def replacement_func():
    return fused_relative_position_embedding_512