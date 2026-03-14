import torch
import triton
import triton.language as tl

# Pattern matching function - match the multiplication operation
def pattern(tmp_0, in_1):
    # Match the multiplication: tmp_1 = tmp_0 * in_1
    tmp_1 = tmp_0 * in_1
    return tmp_1

# Argument extraction function
def replacement_args(tmp_0, in_1):
    return (tmp_0, in_1)

# Optimized Triton kernel for element-wise multiplication
@triton.jit
def triton_mul_kernel(
    x_ptr,        # First operand
    y_ptr,        # Second operand
    out_ptr,      # Output
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Multiplication
    out = x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_multiply(x, y):
    # Launch the multiplication kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create output tensor
    out = torch.empty_like(x)

    # Launch kernel
    triton_mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_multiply