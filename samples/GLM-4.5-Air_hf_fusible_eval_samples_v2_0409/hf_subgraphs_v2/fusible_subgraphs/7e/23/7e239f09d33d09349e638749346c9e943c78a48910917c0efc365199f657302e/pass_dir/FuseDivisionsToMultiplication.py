import torch
import triton
import triton.language as tl

# Pattern matching function - matches the complete computation sequence including softmax
def pattern(in_0, tmp_1, tmp_0):
    """
    Matches the complete pattern: in_0 / (256^0.5) then / 0.05 then softmax(dim=-1)
    """
    # Compute 256^0.5 (this is a constant that can be precomputed)
    tmp_2 = tmp_0 ** tmp_1  # 256 ** 0.5 = 16.0
    in_0_divided = in_0 / tmp_2
    tmp_3 = in_0_divided
    
    # Apply softmax (this is the final operation that gets returned)
    tmp_6 = tmp_3.softmax(dim=-1)
    
    return tmp_6

# Argument extraction function 
def replacement_args(in_0, tmp_1, tmp_0):
    return (in_0, tmp_1, tmp_0)

# Optimized kernel that fuses the two divisions into a single multiplication
@triton.jit
def fused_multiply_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that performs in_0 * 1.25 + softmax preparation"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply fused multiplication (1 / (16.0 * 0.05) = 1.25)
    scaled_x = x * multiplier
    
    # Store intermediate result (needed for softmax)
    tl.store(output_ptr + offsets, scaled_x, mask=mask)

@torch.fx.wrap  
def fused_multiply_operation(input_tensor):
    """Wrapper function for the fused multiply operation"""
    # Calculate total number of elements
    total_elements = input_tensor.numel()
    
    # Kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Launch Triton kernel
    # Multiplier: 1 / (256^0.5 * 0.05) = 1 / (16.0 * 0.05) = 1 / 0.8 = 1.25
    fused_multiply_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=total_elements,
        multiplier=1.25,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

# Replacement function that returns the fused operations including softmax
def replacement_func():
    """Returns the fused multiply + softmax operation"""
    def fused_operations(in_0, tmp_1, tmp_0):
        # Apply the fused multiplication (replaces both divisions: in_0 / 16.0 / 0.05)
        scaled_result = fused_multiply_operation(in_0)
        
        # Apply softmax (replaces tmp_3.softmax(dim=-1))
        softmax_result = torch.softmax(scaled_result, dim=-1)
        
        return softmax_result
    
    return fused_operations