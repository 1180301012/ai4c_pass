import torch
import triton
import triton.language as tl

# Pattern matching function
@torch.fx.wrap
def pattern(x):
    # Match the exact ReLU operation with the model's structure
    return torch.nn.functional.relu(x, inplace=True)

# Argument extraction function
@torch.fx.wrap
def replacement_args(x):
    return (x,)

# Triton kernel for optimized ReLU
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data (coalesced read)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU (faster than torch.nn.functional.relu)
    x_relu = tl.max(x, 0.0)
    
    # Store result (coalesced write)
    tl.store(out_ptr + offsets, x_relu, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def relu_wrapper(x):
    n = x.numel()
    BLOCK_SIZE = 128  # Optimal for coalesced memory access
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    relu_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
@torch.fx.wrap
def replacement_func():
    return relu_wrapper