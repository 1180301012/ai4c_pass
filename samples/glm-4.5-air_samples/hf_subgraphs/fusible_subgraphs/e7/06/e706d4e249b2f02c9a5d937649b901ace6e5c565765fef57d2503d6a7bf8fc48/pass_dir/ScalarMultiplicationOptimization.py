import torch
import triton
import triton.language as tl

@triton.jit
def scalar_mul_kernel(
    x_ptr,
    out_ptr,
    scalar: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized scalar multiplication kernel with better memory access patterns
    """
    # Get program ID and compute memory offsets 
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create contiguous offsets for better memory coalescing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input tensor with better alignment
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply scalar multiplication
    out = x * scalar
    
    # Store results with coalesced memory access
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_scalar_multiply(x, scalar):
    """
    Optimized scalar multiplication wrapper with smart execution
    """
    # Get tensor properties
    n_elements = x.numel()
    
    # For small tensors, fall back to PyTorch (avoid kernel launch overhead)
    # Threshold: use kernel only if we have enough work to justify overhead
    OVERHEAD_THRESHOLD = 4096  # Elements
    
    if n_elements < OVERHEAD_THRESHOLD:
        # For small tensors, use PyTorch's highly optimized implementation
        return x * scalar
    
    # adaptive block sizing based on tensor size
    if n_elements < 32768:
       BLOCK_SIZE = 512
    elif n_elements < 262144:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Compute grid dimensions
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Only launch kernel if we have sufficient work
    if num_programs > 0:
        scalar_mul_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
            scalar=scalar,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def pattern(x, scalar):
    """
    Pattern to match: scalar multiplication operation
    """
    return x * scalar

def replacement_args(x, scalar):
    """
    Extract arguments for the optimized kernel
    """
    return (x, scalar)

def replacement_func():
    """
    Return the optimized scalar multiplication function
    """
    return optimized_scalar_multiply