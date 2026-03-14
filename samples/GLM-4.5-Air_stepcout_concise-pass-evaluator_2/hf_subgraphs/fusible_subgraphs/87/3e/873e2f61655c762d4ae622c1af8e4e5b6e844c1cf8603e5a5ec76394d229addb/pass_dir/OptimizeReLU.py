import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU activation
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(y_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu(x):
    n_elements = x.numel()
    
    # Choose block size based on tensor size
    BLOCK_SIZE = 1024
    if n_elements < 1024:
        BLOCK_SIZE = 128
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Launch kernel
    relu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

def pattern(x):
    # Match relu operation exactly as in the model
    return torch.nn.functional.relu(x, inplace=True)

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_relu