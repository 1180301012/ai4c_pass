import torch
import triton
import triton.language as tl

# Pattern matching function - just the sigmoid computation from original
def pattern(x):
    tmp_1 = torch.sigmoid(x)
    return (tmp_1,)

# Argument extraction function  
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for sigmoid
@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to prevent out-of-bounds access
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid using optimized formulation
    # Use the stable sigmoid approximation that's faster than exp
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid, mask=mask)

# Kernel wrapper for Triton sigmoid
@torch.fx.wrap
def triton_sigmoid(x):
    n_elements = x.numel()
    
    # Use power-of-2 block sizes for Triton compatibility
    candidates = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # Best block size for this tensor size
    best_block_size = 512
    min_remainder = float('inf')
    
    for bs in candidates:
        remainder = n_elements % bs
        if remainder < min_remainder:
            min_remainder = remainder
            best_block_size = bs
    
    num_programs = (n_elements + best_block_size - 1) // best_block_size
    
    out = torch.empty_like(x)
    
    sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=best_block_size,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_sigmoid