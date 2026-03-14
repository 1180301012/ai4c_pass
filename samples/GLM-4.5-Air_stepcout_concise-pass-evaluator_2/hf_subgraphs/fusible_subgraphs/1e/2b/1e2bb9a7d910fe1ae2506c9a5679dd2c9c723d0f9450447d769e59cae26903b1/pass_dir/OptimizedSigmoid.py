import torch
import triton
import triton.language as tl

# Pattern matching function - independent sigmoid operation
def pattern(x):
    """
    Matches independent sigmoid operation
    x: input tensor [300, 1, 256]
    """
    tmp_3 = x.sigmoid()
    return tmp_3

def replacement_args(x):
    return (x,)

# Optimized sigmoid kernel
@triton.jit
def optimized_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid
    out = 1.0 / (1.0 + tl.exp(-x))
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_sigmoid(x):
    N = x.numel()
    # Use larger block size for better GPU utilization with this tensor size
    BLOCK_SIZE = 2048  # Larger block size for better performance
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    optimized_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_sigmoid