import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if not mask.any():
        return
    
    # Get 2D coordinates from 1D offset
    batch_idx = offsets // dim_size
    dim_idx = offsets % dim_size
    
    # Only process if we're within batch bounds
    batch_mask = batch_idx < (n_elements // dim_size)
    if not (mask & batch_mask).any():
        return
    
    # Load current element
    current_x = tl.load(x_ptr + offsets, mask=mask & batch_mask, other=0.0)
    
    # For each batch, calculate L2 norm and normalize
    if dim_idx == 0:
        # Start of batch: compute L2 norm for entire batch along dim=1
        batch_offset = batch_idx * dim_size
        batch_range = tl.arange(0, dim_size)
        batch_valid = batch_range < dim_size
        
        # Load entire batch and compute sum of squares
        batch_data = tl.load(x_ptr + batch_offset, mask=batch_valid, other=0.0)
        sum_sq = tl.sum(batch_data * batch_data)
        norm = tl.sqrt(sum_sq + 1e-6)  # Add epsilon for numerical stability
        
        # Normalize all elements in this batch
        for i in range(dim_size):
            if mask[batch_offset + i] and batch_idx < (n_elements // dim_size):
                element_x = tl.load(x_ptr + batch_offset + i, mask=True, other=0.0)
                normalized_val = element_x / norm
                tl.store(out_ptr + batch_offset + i, normalized_val)
    
    # Also process the current element directly (fallback for cases where batch processing doesn't cover all)
    if mask & batch_mask:
        batch_offset = batch_idx * dim_size
        batch_range = tl.arange(0, dim_size)
        batch_valid = batch_range < dim_size
        
        # Load entire batch and compute sum of squares
        batch_data = tl.load(x_ptr + batch_offset, mask=batch_valid, other=0.0)
        sum_sq = tl.sum(batch_data * batch_data)
        norm = tl.sqrt(sum_sq + 1e-6)
        
        # Store normalized value for current element
        normalized_val = current_x / norm
        tl.store(out_ptr + offsets, normalized_val, mask=mask & batch_mask)

@torch.fx.wrap
def optimized_l2_normalize(x):
    """Optimized L2 normalization using Triton kernel"""
    if x.dim() != 2:
        raise ValueError("This kernel is optimized for 2D tensors")
    
    n_elements = x.numel()
    dim_size = x.shape[1]  # Normalize along dimension 1
    
    # Use optimal block size based on hidden dimension
    BLOCK_SIZE = 1024  # This works well for 768 hidden dimension
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch the kernel
    l2_normalize_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        dim_size=dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_l2_normalize