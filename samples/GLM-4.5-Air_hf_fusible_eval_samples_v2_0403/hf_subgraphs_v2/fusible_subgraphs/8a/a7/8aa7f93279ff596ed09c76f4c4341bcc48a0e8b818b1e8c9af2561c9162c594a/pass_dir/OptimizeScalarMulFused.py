import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1):
    """Match scalar multiplication operation"""
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized scalar multiplication kernel
@triton.jit
def scalar_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance element-wise scalar multiplication kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform scalar multiplication
    out = x * scalar
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_scalar_mul(x, scalar_const=0.1767766952966369):
    """Wrapper function for scalar multiplication with Triton kernel"""
    # Get tensor properties
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimized block size
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch Triton kernel
    scalar_mul_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        scalar=scalar_const,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return triton_scalar_mul