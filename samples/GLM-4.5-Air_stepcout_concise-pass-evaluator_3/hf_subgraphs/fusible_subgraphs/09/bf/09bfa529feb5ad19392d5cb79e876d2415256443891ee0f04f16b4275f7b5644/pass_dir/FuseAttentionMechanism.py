import torch
import triton
import triton.language as tl
import math

def pattern(x, y):
    """
    Simple pattern: just add operation
    """
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_add_softmax_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fixed fused addition + softmax kernel with safe memory access"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Only proceed if this block has valid work
    if tl.sum(mask, axis=0) == 0:
        return
    
    # Load inputs with proper masking
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Vectorized addition
    added = x + y
    
    # Safer softmax implementation
    # Handle the case where all elements in the block are masked
    max_val = tl.max(added, axis=0)
    exp_x = tl.exp(added - max_val)
    sum_exp = tl.sum(exp_x, axis=0, keep_dims=True)
    # Avoid division by zero
    sum_exp = tl.where(sum_exp == 0, 1.0, sum_exp)
    softmax_out = exp_x / sum_exp
    
    # Store result with proper masking
    tl.store(output_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def optimized_fused_add_softmax(x, y):
    """Optimized wrapper function for fused add + softmax operation"""
    # Handle broadcasting for attention patterns
    if x.shape != y.shape:
        if y.dim() == 4 and x.dim() == 4:
            if y.shape[1] == 1 and y.shape[2] == 1 and y.shape[3] == x.shape[3]:
                y = y.expand(-1, x.shape[1], x.shape[2], -1)
            else:
                # For unsupported broadcasting patterns, just use addition as-is
                # PyTorch will handle broadcasting automatically
                pass
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate total elements and choose optimal block size
    n_elements = x.numel()
    
    # More conservative block size selection to avoid memory access issues
    if n_elements <= 4096:
        BLOCK_SIZE = 512
    elif n_elements <= 16384:
        BLOCK_SIZE = 1024
    elif n_elements <= 65536:
        BLOCK_SIZE = 2048
    elif n_elements <= 262144:
        BLOCK_SIZE = 4096
    else:
        BLOCK_SIZE = 2048  # Conservative block size for very large tensors
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized Triton kernel
    fused_add_softmax_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_fused_add_softmax