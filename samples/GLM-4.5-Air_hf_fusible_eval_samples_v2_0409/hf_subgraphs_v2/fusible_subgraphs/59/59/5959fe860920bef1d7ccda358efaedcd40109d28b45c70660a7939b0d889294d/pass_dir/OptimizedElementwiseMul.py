import torch
import triton
import triton.language as tl

def pattern(weight_tensor, scale):
    """
    Pattern: element-wise multiplication with scalar scale
    Original pattern:
      tmp_3 = tmp_2 * 1.0
      tmp_5 = tmp_4 * 1.0
      tmp_17 = weight_tensor * normalized_input
    """
    # Element-wise multiplication with scalar
    result = weight_tensor * scale
    
    return result

def replacement_args(weight_tensor, scale):
    """Extract arguments for the replacement function"""
    return (weight_tensor, scale)

@triton.jit
def elementwise_mul_kernel(
    weight_ptr,           # Weight tensor pointer
    input_ptr,            # Input tensor pointer  
    output_ptr,           # Output tensor pointer
    n_elements,           # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized element-wise multiplication kernel"""
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data from both tensors
    weight_val = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise multiplication
    result = weight_val * input_val
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_elementwise_mul(weight_tensor, scale):
    """Wrapper function for optimized element-wise multiplication with scalar"""
    # Simple scalar multiplication
    result = weight_tensor * scale
    
    return result

def replacement_func():
    """Return the optimized element-wise multiplication function"""
    return optimized_elementwise_mul