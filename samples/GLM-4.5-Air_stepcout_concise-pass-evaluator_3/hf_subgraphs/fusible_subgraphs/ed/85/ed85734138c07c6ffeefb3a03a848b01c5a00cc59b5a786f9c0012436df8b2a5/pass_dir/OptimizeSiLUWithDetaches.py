import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match the computation pattern: SiLU + detach operations
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for SiLU activation
@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    High-performance SiLU kernel using Triton
    SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    """
    # Get program ID and create offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    # Use stable sigmoid computation
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper decorated with torch.fx.wrap
@torch.fx.wrap
def optimized_silu(in_0, in_1, in_2):
    """
    Optimized computation that replaces the original pattern
    """
    # Handle SiLU computation with Triton kernel
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor for SiLU result
    out_silu = torch.empty_like(in_0)
    
    # Launch Triton kernel
    silu_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=out_silu,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Handle detach operations (essentially free - just reference counting)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = out_silu.detach()  # Detach the result from SiLU computation
    
    # Return the same structure as the original pattern
    return (tmp_1, tmp_2, tmp_3, out_silu)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_silu