import torch
import triton
import triton.language as tl

# Pattern matching function for scalar multiplication
def pattern(input_tensor, constant):
    """Match scalar multiplication pattern: tmp = input * constant"""
    result = input_tensor * constant
    return result

# Argument extraction function
def replacement_args(input_tensor, constant):
    """Extract arguments needed for the replacement"""
    # Return arguments without conversion to handle symbolic proxies
    return (input_tensor, constant)

# Triton kernel for optimized scalar multiplication
@triton.jit
def scalar_mult_kernel(
    input_ptr,
    output_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for efficient scalar multiplication using Triton"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with bounds checking
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform scalar multiplication
    output_data = input_data * scalar
    
    # Store result
    tl.store(output_ptr + offsets, output_data, mask=mask)

# Kernel wrapper (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_scalar_multiply(input_tensor, scalar_value):
    """Perform optimized scalar multiplication using Triton"""
    # Extract actual scalar value (handles both proxies and concrete values)
    if hasattr(scalar_value, 'item'):
        # This is a tensor with a single value
        scalar = scalar_value.item()
    else:
        # This is already a scalar value
        scalar = float(scalar_value)
    
    # Get tensor properties
    n_elements = input_tensor.numel()
    
    # Optimized block size for float32 operations on modern GPUs
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch the kernel
    scalar_mult_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        scalar=scalar,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (no arguments, returns function reference)
def replacement_func():
    """Return the optimized scalar multiplication function"""
    return optimized_scalar_multiply