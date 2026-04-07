import torch
import triton
import triton.language as tl

# Define the divisor constant from the computation
DIVISOR = 11.313708498984761

def pattern(in_0):
    """
    Simple test pattern: just division by constant
    """
    return in_0 / 11.313708498984761

def replacement_args(in_0):
    """Extract input tensor argument for the fused kernel"""
    return (in_0,)

@triton.heuristics({
    'BLOCK_SIZE': lambda kwargs: kwargs['n_elements']
})
@triton.jit
def division_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized division kernel with autotuning support
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply division with high precision
    output = x / divisor
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def optimized_division_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized division kernel with optimal block size, memory coalescing, and warp tuning
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with optimized memory access
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply division - single operation, minimal overhead
    # Using fused division operation for better performance
    output = x * (1.0 / divisor)
    
    # Store result with coalesced memory access
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_division(input_tensor):
    """
    Wrapper function for the optimized division kernel
    Uses data-type specific optimal configuration for better performance
    """
    # Get tensor properties
    n_elements = input_tensor.numel()
    
    # Choose optimal block size based on data type for better GPU occupancy
    if input_tensor.dtype == torch.float32:
        BLOCK_SIZE = 512  # Smaller block for float32 for better precision
    else:  # float16 or bfloat16
        BLOCK_SIZE = 1024  # Larger block for half precision types
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device
    output = torch.empty_like(input_tensor)
    
    # Launch the optimized division kernel
    optimized_division_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        divisor=DIVISOR,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized kernel wrapper function"""
    return optimized_division