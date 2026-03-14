import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation from model.py
def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized softmax kernel with better configuration
@triton.jit
def simple_softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for this block
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Apply scalar multiplication
    scaled_x = x * scale
    
    # Subtract max for numerical stability
    max_x = tl.max(scaled_x)
    stable_x = scaled_x - max_x
    
    # Compute exp and sum
    exp_x = tl.exp(stable_x)
    sum_exp = tl.sum(exp_x)
    
    # Compute softmax
    softmax = exp_x / sum_exp
    
    # Store result
    tl.store(out_ptr + offsets, softmax, mask=mask)

@torch.fx.wrap
def optimized_forward(in_0):
    # Get input tensor properties
    batch_size, channels, height, width = in_0.shape
    total_elements = batch_size * channels * height * width
    
    # Choose block size based on input size to optimize for different workloads
    if batch_size >= 16:
        # Larger block size for larger batches
        BLOCK_SIZE = 1024
        num_warps = 4
        num_stages = 2
    elif batch_size >= 4:
        # Medium block size
        BLOCK_SIZE = 512
        num_warps = 2
        num_stages = 1
    else:
        # Smaller block size for smaller batches to reduce overhead
        BLOCK_SIZE = 256
        num_warps = 1
        num_stages = 1
    
    # Calculate grid size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (intermediate result before transpose)
    out_intermediate = torch.empty_like(in_0)
    
    # Scale factor as Python scalar
    scale = 0.1767766952966369
    
    # Launch kernel with configuration
    simple_softmax_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=out_intermediate,
        n_elements=total_elements,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply transpose
    result = out_intermediate.transpose(-2, -1)
    return result

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return optimized_forward