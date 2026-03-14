import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_3):
    """
    Match the HardTanh operation: tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    """
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return tmp_3

# Argument extraction function
def replacement_args(in_3):
    return (in_3,)

# Optimized HardTanh kernel using Triton
@triton.jit
def hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply HardTanh: y = min(max(x, min_val), max_val)
    y = tl.where(x < min_val, min_val, x)
    y = tl.where(y > max_val, max_val, y)
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_hardtanh(input, min_val=0.0, max_val=6.0):
    # Get total number of elements
    n_elements = input.numel()
    
    # Use optimal block size for GPU
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    hardtanh_kernel[(num_programs,)](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        min_val=min_val,
        max_val=max_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_hardtanh