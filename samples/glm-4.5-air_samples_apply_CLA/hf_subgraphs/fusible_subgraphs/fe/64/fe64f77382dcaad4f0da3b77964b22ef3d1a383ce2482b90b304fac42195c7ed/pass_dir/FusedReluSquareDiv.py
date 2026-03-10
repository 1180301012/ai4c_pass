import torch
import triton
import triton.language as tl

# Pattern matching function - match the exact computation sequence
def pattern(x):
    # Match the exact computation from the original model
    tmp_0 = x / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    # Return the observable value (what the original function returns)
    return tmp_2

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized fused kernel using Triton
@triton.jit
def fused_relu_square_div_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.range(BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: relu(x / scale) ** 2
    x_div_scale = x / scale
    apply_relu = tl.maximum(x_div_scale, 0.0)
    result = apply_relu * apply_relu
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper with automatic block size tuning
@torch.fx.wrap
def fused_relu_square_div(x):
    # Get tensor properties
    n_elements = x.numel()
    x = x.contiguous()  # Ensure contiguous memory layout
    
    # Determine optimal block size based on tensor size
    if n_elements < 8192:
        BLOCK_SIZE = 64
    elif n_elements < 65536:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_relu_square_div_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        scale=11.313708498984761,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Optimized fused kernel using Triton
@triton.jit
def fused_relu_square_div_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: relu(x / scale) ** 2
    x_div_scale = x / scale
    apply_relu = tl.maximum(x_div_scale, 0.0)
    result = apply_relu * apply_relu
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Replacement function - returns the fused kernel
def replacement_func():
    return fused_relu_square_div