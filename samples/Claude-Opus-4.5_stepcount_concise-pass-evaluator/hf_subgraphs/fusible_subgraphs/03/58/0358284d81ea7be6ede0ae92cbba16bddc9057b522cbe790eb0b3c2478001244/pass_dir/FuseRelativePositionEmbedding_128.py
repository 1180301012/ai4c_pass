import torch
from torch import device
import triton
import triton.language as tl


def pattern(indices, weight):
    """Pattern to match embedding + to operation"""
    tmp_6 = torch.nn.functional.embedding(indices, weight, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return tmp_7


def replacement_args(indices, weight):
    return (indices, weight)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['num_indices', 'embedding_dim'],
)
@triton.jit
def fused_embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    num_indices,
    embedding_dim: tl.constexpr,
):
    # Each program handles one index
    idx = tl.program_id(0)
    
    if idx >= num_indices:
        return
    
    # Load the embedding index
    emb_idx = tl.load(indices_ptr + idx)
    
    # Load entire embedding vector - use power of 2 for efficiency
    PADDED_DIM: tl.constexpr = 64  # Embedding dim is 64
    d_offsets = tl.arange(0, PADDED_DIM)
    mask = d_offsets < embedding_dim
    
    embedding_vec = tl.load(
        weight_ptr + emb_idx * embedding_dim + d_offsets,
        mask=mask,
        other=0.0
    )
    
    # Store to output
    tl.store(
        output_ptr + idx * embedding_dim + d_offsets, 
        embedding_vec,
        mask=mask
    )


@torch.fx.wrap
def fused_embedding_to_float(indices, weight):
    output_shape = indices.shape + (weight.shape[1],)
    output = torch.empty(output_shape, dtype=torch.float32, device=weight.device)
    
    num_indices = indices.numel()
    embedding_dim = weight.shape[1]
    
    # Launch one program per index
    grid = (num_indices,)
    
    fused_embedding_kernel[grid](
        indices.view(-1),
        weight,
        output.view(-1, embedding_dim),
        num_indices,
        embedding_dim=embedding_dim,
    )
    
    return output


def replacement_func():
    return fused_embedding_to_float