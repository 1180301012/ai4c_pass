import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple division pattern - match division operation
    return x / y

def replacement_args(x, y):
    # Extract tensors - for division by constant, y represents the constant
    # Calculate reciprocal for optimization
    reciprocal = 1.0 / y
    return (x, reciprocal)

@triton.jit
def division_kernel(
    x_ptr,
    reciprocal,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply by reciprocal instead of dividing by constant
    out = x * reciprocal
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def division_kernel_optimized(
    x_ptr,
    reciprocal,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply by reciprocal instead of dividing by constant
    out = x * reciprocal
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_division(x, reciprocal):
    # Handle both cases where x could be 2D, 3D, or 4D
    original_shape = x.shape
    N = x.numel()
    
    # Use dynamic block size based on tensor size for better performance
    if N < 1024:
        BLOCK_SIZE = 256
    elif N < 4096:
        BLOCK_SIZE = 1024
    elif N < 16384:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch the optimized kernel with dynamic block size
    division_kernel_optimized[(num_programs,)](
        x_ptr=x,
        reciprocal=reciprocal,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_division