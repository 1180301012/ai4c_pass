import torch
import triton
import triton.language as tl

# Pattern matching function - simple square to test baseline
def pattern(x):
    return torch.square(x)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized fused kernel - combine all 3 operations in one kernel
@triton.jit
def fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    inv_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to avoid out-of-bounds access
    
    # Load input data once - single memory access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations in one computation:
    # 1. Division by constant (optimized as multiplication)
    tmp_0 = x * inv_scale
    
    # 2. ReLU activation (element-wise maximum)
    tmp_1 = tl.maximum(tmp_0, 0.0)
    
    # 3. Square operation 
    out = tmp_1 * tmp_1
    
    # Store result - single memory access
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_div_relu_square(x):
    N = x.numel()
    
    # Optimal block size for GPU occupancy
    BLOCK_SIZE = 512  # Smaller block size for better occupancy
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch optimized square kernel with adaptive grid
    if num_programs <= 256:  # Use 2D grid for better utilization
        fused_div_relu_square_kernel[(num_programs, 4)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
            inv_scale=1.0,  # Remove unused parameter for now
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        fused_div_relu_square_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
            inv_scale=1.0,  # Remove unused parameter for now
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_div_relu_square