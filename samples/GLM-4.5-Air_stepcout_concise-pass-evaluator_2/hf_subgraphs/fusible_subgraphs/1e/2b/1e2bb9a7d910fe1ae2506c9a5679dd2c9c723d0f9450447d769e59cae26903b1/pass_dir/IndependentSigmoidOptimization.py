import torch
import triton
import triton.language as tl

# Pattern matching function for independent sigmoid operations
def pattern(x):
    return x.sigmoid()

def replacement_args(x):
    return (x,)

# Optimized sigmoid kernel with better kernel launch strategy
@triton.jit
def optimized_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid with optimized computation
    out = tl.exp(-tl.abs(x))
    out = tl.where(x >= 0.0, 1.0 / (1.0 + out), out / (1.0 + out))
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_sigmoid(x):
    if x.numel() == 0:
        return torch.empty_like(x)
    
    N = x.numel()
    
    # Adaptive block size based on tensor size for better GPU utilization
    if N >= 1000000:  # Large tensors
        BLOCK_SIZE = 4096
    elif N >= 100000:  # Medium tensors  
        BLOCK_SIZE = 2048
    else:  # Small tensors
        BLOCK_SIZE = 1024
        
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Early exit for very small tensors to avoid overhead
    if N < 4096:
        # Use regular sigmoid for very small tensors due to overhead
        return x.sigmoid()
    
    optimized_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_sigmoid