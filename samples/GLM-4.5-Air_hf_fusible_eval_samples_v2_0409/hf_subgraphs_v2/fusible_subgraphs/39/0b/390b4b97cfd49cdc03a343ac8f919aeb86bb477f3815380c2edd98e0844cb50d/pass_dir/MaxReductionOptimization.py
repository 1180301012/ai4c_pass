import torch
import triton
import triton.language as tl

def pattern(x):
    max1 = x.max(0, keepdim=False)
    result1 = max1[0]
    max2 = result1.max(-1, keepdim=True)
    result2 = max2[0]
    return result2

def replacement_args(x):
    return (x,)

@triton.jit
def max_reduction_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program reduces a block along dimension 0
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input block
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max in this block
    block_max = tl.max(x)
    
    # Store to output
    tl.store(out_ptr + pid, block_max)

@torch.fx.wrap
def optimized_max_reduction(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # First reduction: along dimension 0
    if x.ndim > 1:
        # Reshape to 2D for easier processing
        x_flat = x.flatten()
        num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Temporary buffer for block maxima
        block_maxima = torch.empty(num_blocks, dtype=x.dtype, device=x.device)
        
        # First pass: reduce along first dimension
        max_reduction_kernel[(num_blocks,)](
            x_flat,
            block_maxima,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Second pass: reduce the block maxima to get final result
        final_result = torch.max(block_maxima).item()
        
        return torch.tensor(final_result, dtype=x.dtype, device=x.device)
    else:
        # For 1D tensors, just use torch.max for simplicity
        return torch.max(x)

def replacement_func():
    return optimized_max_reduction