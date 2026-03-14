import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple addition pattern - this is more reliable to optimize
    return x + y,

def replacement_args(x, y):
    return (x, y)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_triton_add(x, y):
    # Handle multi-dimensional tensors
    if x.shape != y.shape:
        # Handle broadcasting for simple cases
        if len(x.shape) != len(y.shape):
            # Simple broadcasting: pad with ones on the left
            max_ndim = max(len(x.shape), len(y.shape))
            x_shape = (1,) * (max_ndim - len(x.shape)) + x.shape
            y_shape = (1,) * (max_ndim - len(y.shape)) + y.shape
            x = x.reshape(x_shape)
            y = y.reshape(y_shape)
    
    # Flatten for element-wise operations
    n_elements = x.numel()
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Determine block size
    BLOCK_SIZE = min(1024, n_elements)
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    out_flat = torch.empty_like(x_flat)
    
    # Launch kernel
    add_kernel[(num_programs,)](
        x_ptr=x_flat,
        y_ptr=y_flat,
        out_ptr=out_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_flat.reshape(x.shape)

def replacement_func():
    return simple_triton_add