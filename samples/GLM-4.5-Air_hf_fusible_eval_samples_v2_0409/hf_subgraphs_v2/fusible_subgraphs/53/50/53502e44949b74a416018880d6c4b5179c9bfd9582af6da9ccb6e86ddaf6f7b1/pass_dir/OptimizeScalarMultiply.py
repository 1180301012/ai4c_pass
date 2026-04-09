import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor, scale):
    """
    Match tensor multiplication by scalar
    """
    tmp_5 = input_tensor * scale
    return tmp_5

# Argument extraction function
def replacement_args(input_tensor, scale):
    return (input_tensor, scale)

# Triton kernel for efficient scalar multiplication
@triton.jit
def scalar_multiply_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Apply scalar multiplication
    out = x * scale
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_scalar_multiply(input_tensor, scale):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    scalar_multiply_kernel[(num_programs,)](
        output_ptr=output,
        input_ptr=input_tensor,
        n_elements=n_elements,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_scalar_multiply