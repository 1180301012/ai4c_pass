import torch
import triton
import triton.language as tl

# Pattern matching function for element-wise scalar multiplication
def pattern(in_1):
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized kernel for element-wise scalar multiplication
@triton.jit
def elementwise_scalar_mul_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scalar: tl.constexpr,
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
    out = x * scalar
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def elementwise_scalar_mul_kernel_wrapper(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024  # Optimized for GPU occupancy
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_tensor = torch.empty_like(input_tensor)
    
    elementwise_scalar_mul_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=N,
        scalar=0.1767766952966369,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

# Replacement function (returns kernel wrapper function)
def replacement_func():
    return elementwise_scalar_mul_kernel_wrapper