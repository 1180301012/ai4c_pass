import torch
import triton
import triton.language as tl

# Pattern matching function - exactly mirror the dropout pattern structure
def pattern(in_0):
    # Match the exact SiLU operation from the model
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    return tmp_0

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for SiLU operation
@triton.jit
def triton_silu_kernel(
    x_ptr,        # Input to SiLU
    out_ptr,      # Output
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    sig_x = tl.sigmoid(x)  # Compute sigmoid(x)
    out = x * sig_x        # Compute SiLU
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_silu(x):
    # Launch the SiLU kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create output tensor
    out = torch.empty_like(x)

    # Launch kernel
    triton_silu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_silu