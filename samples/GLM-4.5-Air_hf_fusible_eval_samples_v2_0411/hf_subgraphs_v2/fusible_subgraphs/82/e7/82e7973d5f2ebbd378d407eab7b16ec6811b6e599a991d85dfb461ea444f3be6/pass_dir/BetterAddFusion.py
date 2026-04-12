import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Pattern matching for simple addition operation
    """
    return a + b

def replacement_args(x, y):
    return (x, y)

@triton.jit
def better_add_kernel(
    x_ptr,           # pointer to input x
    y_ptr,           # pointer to input y  
    out_ptr,         # pointer to output
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with optimized memory access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def better_add(x, y):
    """
    Better addition operation using Triton with optimized block sizing
    """
    # Ensure tensors are on the same device and have same shape/dtype
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    
    N = x.numel()
    
    # Use optimized block sizes based on tensor size for better performance
    if N < 1024:
        BLOCK_SIZE = 256  # Smaller block for small tensors
    elif N < 10000:
        BLOCK_SIZE = 512  # Medium block for medium tensors
    elif N < 100000:
        BLOCK_SIZE = 1024  # Large block for large tensors
    else:
        BLOCK_SIZE = 2048  # Very large block for very large tensors
    
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with optimized configuration
    better_add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return better_add