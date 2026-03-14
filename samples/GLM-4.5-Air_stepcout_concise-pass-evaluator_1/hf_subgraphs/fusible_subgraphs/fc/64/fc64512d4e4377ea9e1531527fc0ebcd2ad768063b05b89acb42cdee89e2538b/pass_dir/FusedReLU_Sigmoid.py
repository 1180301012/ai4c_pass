import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.sigmoid(x)

def replacement_args(x):
    return (x,)

@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized sigmoid kernel using Triton"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Sigmoid: 1/(1+exp(-x))
    # Use optimized exponential for better performance
    out = 1.0 / (1.0 + tl.exp(-x))
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

# Optimized kernel for all tensor sizes
@triton.jit
def sigmoid_kernel_optimized(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for all tensor sizes"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Single expression for performance
    tl.store(out_ptr + offsets, 1.0 / (1.0 + tl.exp(-x)), mask=mask)

@torch.fx.wrap
def triton_sigmoid(x):
    n_elements = x.numel()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Perfectly optimized for our exact tensor sizes
    # Case 1: n_elements = 256 (1, 256, 1, 1)
    # Case 2: n_elements = 8192 (32, 256, 1, 1)
    if n_elements == 256:
        # Perfect for 1 program to handle the entire tensor
        BLOCK_SIZE = 256
        num_programs = 1
    elif n_elements == 8192:
        # 8192 / 64 = 128 programs with optimal block size
        BLOCK_SIZE = 64
        num_programs = 128
    elif n_elements <= 512:
        BLOCK_SIZE = 256
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    elif n_elements <= 16384:
        BLOCK_SIZE = 128
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:  # Large tensors
        BLOCK_SIZE = 512
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized kernel
    sigmoid_kernel_optimized[(num_programs, 1, 1)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_sigmoid