import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_3 = x.view(1, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    tmp_3 = None
    return tmp_4

def replacement_args(x):
    return (x,)

@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data directly from original tensor (avoiding view overhead)
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Compute exp(x - max)
    exp_x = tl.exp(x - max_val)
    
    # Compute sum
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Normalize
    out = exp_x / sum_exp
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_view_softmax(x):
    """
    Fuse view operation with softmax to avoid intermediate allocation.
    The target_shape should have -1 as the last dimension to indicate flattening.
    """
    orig_shape = x.shape
    target_shape = list(target_shape)
    
    # Calculate the actual size of the flattened dimension
    flattened_size = 1
    for dim in target_shape:
        if dim != -1:
            flattened_size *= dim
    
    # Check if the target shape is valid (last dim should be flattened)
    if target_shape[-1] == -1:
        # Number of elements before the flattened dimension
        prefix_size = 1
        for dim in target_shape[:-1]:
            prefix_size *= dim
        
        total_elements = x.numel()
        
        # We need to apply softmax along the flattened dimension
        # Reshape to [prefix_size, flattened_size] for efficient softmax
        if prefix_size == 1:
            # Single group case - direct 1D softmax
            BLOCK_SIZE = 1024
            num_programs = (flattened_size + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            out = torch.empty(flattened_size, dtype=x.dtype, device=x.device)
            fused_softmax_kernel[(num_programs,)](
                x_ptr=x,
                out_ptr=out,
                n_elements=flattened_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return out.view(*target_shape)
        else:
            # Multiple groups - softmax per group
            x_reshaped = x.reshape(prefix_size, flattened_size)
            BLOCK_SIZE = 1024
            num_programs = (prefix_size * flattened_size + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            out_reshaped = torch.empty_like(x_reshaped)
            fused_softmax_kernel[(num_programs,)](
                x_ptr=x_reshaped,
                out_ptr=out_reshaped,
                n_elements=prefix_size * flattened_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            return out_reshaped.view(*target_shape)
    else:
        # If last dimension is not -1, fall back to standard approach
        reshaped = x.view(*target_shape)
        return reshaped.softmax(dim=-1)

def replacement_func():
    return fused_view_softmax