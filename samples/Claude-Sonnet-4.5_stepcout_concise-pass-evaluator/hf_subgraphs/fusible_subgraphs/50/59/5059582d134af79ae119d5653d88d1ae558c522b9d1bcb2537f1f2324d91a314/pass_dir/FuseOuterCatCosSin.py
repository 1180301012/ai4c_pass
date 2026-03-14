import torch
import triton
import triton.language as tl
from torch import device

def pattern(in_0, in_1):
    """Match outer product + cat + to pattern"""
    tmp_3 = torch.outer(in_0, in_1)
    tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
    tmp_5 = tmp_4.to(device(type='cuda', index=0))
    return tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_outer_cat_kernel(
    seq_idx_ptr,
    inv_freq_ptr,
    out_ptr,
    seq_len,
    dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused outer product and concatenation using 1D indexing"""
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute 2D indices from 1D offset (for output which is seq_len x (dim*2))
    row = offsets // (dim * 2)
    col = offsets % (dim * 2)
    
    # For the first half (col < dim), we compute outer product
    # For the second half (col >= dim), we duplicate the first half
    actual_col = tl.where(col < dim, col, col - dim)
    
    # Load seq and inv_freq values
    seq_val = tl.load(seq_idx_ptr + row, mask=mask, other=0.0)
    inv_freq_val = tl.load(inv_freq_ptr + actual_col, mask=mask, other=0.0)
    
    # Compute outer product
    result = seq_val * inv_freq_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_outer_cat(seq_idx, inv_freq):
    """Fused outer product and cat"""
    seq_len = seq_idx.shape[0]
    dim = inv_freq.shape[0]
    
    # Allocate output
    out = torch.empty((seq_len, dim * 2), dtype=inv_freq.dtype, device=inv_freq.device)
    
    # Total number of elements in output
    n_elements = seq_len * dim * 2
    
    # Use a fixed block size optimized for small tensors
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_outer_cat_kernel[grid](
        seq_idx,
        inv_freq,
        out,
        seq_len,
        dim,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_outer_cat