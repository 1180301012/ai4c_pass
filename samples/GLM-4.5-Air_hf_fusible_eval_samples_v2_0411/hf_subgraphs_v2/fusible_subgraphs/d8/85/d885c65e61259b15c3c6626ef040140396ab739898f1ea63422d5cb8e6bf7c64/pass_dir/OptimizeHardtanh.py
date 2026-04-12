import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(hardtanh_input):
    """
    Match just the hardtanh operation that gets applied after sigmoid and multiplication
    """
    result = torch.nn.functional.hardtanh(hardtanh_input, 0.0, 6.0, False)
    return result

# Argument extraction function  
def replacement_args(hardtanh_input):
    """
    Extract arguments needed for the replacement kernel
    """
    return (hardtanh_input,)

# Optimized kernel for hardtanh operation
@triton.jit
def optimized_hardtanh_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance hardtanh kernel that clamps values between min_val and max_val
    """
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply clamping: min(max(input, min_val), max_val)
    result = tl.where(input_val < min_val, min_val,
                     tl.where(input_val > max_val, max_val, input_val))
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper for different data types
@torch.fx.wrap
def optimized_hardtanh(input_tensor):
    """
    Wrapper that handles different data types and launches the kernel
    """
    # Get input shape and determine data type
    n_elements = input_tensor.numel()
    dtype = input_tensor.dtype
    
    # Create output tensor with same properties as input
    out = torch.empty_like(input_tensor)
    
    # Block size configuration for GPU efficiency
    BLOCK_SIZE = 1024 if n_elements >= 1024 else 256
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with appropriate data type handling
    if dtype == torch.float16:
        # Specialized for float16
        optimized_hardtanh_kernel[(num_programs,)](
            input_tensor,
            out,
            n_elements,
            min_val=0.0,
            max_val=6.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif dtype == torch.bfloat16:
        # Specialized for bfloat16
        optimized_hardtanh_kernel[(num_programs,)](
            input_tensor,
            out,
            n_elements,
            min_val=0.0,
            max_val=6.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fallback for float32
        optimized_hardtanh_kernel[(num_programs,)](
            input_tensor,
            out,
            n_elements,
            min_val=0.0,
            max_val=6.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function
def replacement_func():
    """
    Return the optimized kernel function
    """
    return optimized_hardtanh