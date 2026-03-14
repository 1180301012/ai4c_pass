import torch
import triton
import triton.language as tl
from torch import device

def pattern(in_0, in_1):
    """Match the outer product + cat + cos/sin + reshape computation"""
    tmp_3 = torch.outer(in_0, in_1)
    tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
    tmp_5 = tmp_4.to(device(type='cuda', index=0))
    tmp_6 = tmp_5.cos()
    tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_5.sin()
    tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
    return (tmp_7, tmp_9)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def rope_cos_sin_kernel(
    seq_idx_ptr,
    inv_freq_ptr,
    cos_out_ptr,
    sin_out_ptr,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    """Compute cos and sin for RoPE embeddings from outer product"""
    pid_seq = tl.program_id(0)
    pid_dim = tl.program_id(1)
    
    seq_start = pid_seq * BLOCK_SIZE_SEQ
    dim_start = pid_dim * BLOCK_SIZE_DIM
    
    seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_SEQ)
    dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_DIM)
    
    seq_mask = seq_offsets < seq_len
    dim_mask = dim_offsets < dim
    
    # Load seq_idx and inv_freq values
    seq_vals = tl.load(seq_idx_ptr + seq_offsets, mask=seq_mask, other=0.0)
    inv_freq = tl.load(inv_freq_ptr + dim_offsets, mask=dim_mask, other=0.0)
    
    # Compute outer product: seq_vals[:, None] * inv_freq[None, :]
    seq_expanded = seq_vals[:, None]
    inv_freq_expanded = inv_freq[None, :]
    
    outer = seq_expanded * inv_freq_expanded
    
    # Compute indices for the concatenated result (cat doubles the last dimension)
    for i in range(BLOCK_SIZE_SEQ):
        for j in range(BLOCK_SIZE_DIM):
            if seq_offsets[i] < seq_len and dim_offsets[j] < dim:
                seq_idx_val = seq_offsets[i]
                dim_idx = dim_offsets[j]
                val = outer[i, j]
                
                # Store for first half (original outer product)
                out_idx = seq_idx_val * (dim * 2) + dim_idx
                cos_val = tl.cos(val)
                sin_val = tl.sin(val)
                tl.store(cos_out_ptr + out_idx, cos_val)
                tl.store(sin_out_ptr + out_idx, sin_val)
                
                # Store for second half (duplicate)
                out_idx_dup = seq_idx_val * (dim * 2) + dim + dim_idx
                tl.store(cos_out_ptr + out_idx_dup, cos_val)
                tl.store(sin_out_ptr + out_idx_dup, sin_val)

@triton.jit
def rope_multiply_kernel(
    query_ptr,
    cos_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Multiply query with cosine embeddings"""
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = batch_size * num_heads * seq_len * head_dim
    mask = offsets < total_elements
    
    # Load query and cos
    query = tl.load(query_ptr + offsets, mask=mask, other=0.0)
    cos = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    out = query * cos
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_rope_cos_sin(seq_idx, inv_freq):
    """Fused RoPE cosine/sine computation"""
    # Get dimensions
    seq_len = seq_idx.shape[0]
    dim = inv_freq.shape[0]
    
    # Allocate outputs
    cos_out = torch.empty((seq_len, dim * 2), dtype=inv_freq.dtype, device=inv_freq.device)
    sin_out = torch.empty((seq_len, dim * 2), dtype=inv_freq.dtype, device=inv_freq.device)
    
    # Launch kernel to compute cos and sin
    BLOCK_SIZE_SEQ = 64
    BLOCK_SIZE_DIM = 32
    grid_seq = (seq_len + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ
    grid_dim = (dim + BLOCK_SIZE_DIM - 1) // BLOCK_SIZE_DIM
    
    rope_cos_sin_kernel[(grid_seq, grid_dim)](
        seq_idx,
        inv_freq,
        cos_out,
        sin_out,
        seq_len=seq_len,
        dim=dim,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )
    
    # Add dimensions: [seq_len, dim*2] -> [1, 1, seq_len, dim*2]
    tmp_7 = cos_out[None, None, :, :]
    tmp_9 = sin_out[None, None, :, :]
    
    return (tmp_7, tmp_9)

def replacement_func():
    return fused_rope_cos_sin