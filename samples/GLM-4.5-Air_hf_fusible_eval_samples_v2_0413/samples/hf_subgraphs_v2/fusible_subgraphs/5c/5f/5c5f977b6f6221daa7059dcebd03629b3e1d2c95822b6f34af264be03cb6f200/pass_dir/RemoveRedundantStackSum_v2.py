import torch
import triton
import triton.language as tl

def pattern(input_tensor, weights, bias, concat_tensor):
    """
    Matches the redundant stack-then-sum pattern that occurs after conv2d
    """
    # Conv2D operation with exact same parameters as in original graph
    conv_result = torch.conv2d(input_tensor, weights, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # The redundant operations: stacking a single tensor and then summing
    stacked_tensor = torch.stack([conv_result], dim=0)
    reduced_tensor = stacked_tensor.sum(dim=0)
    
    # This intermediate result is what gets used in the concatenation
    intermediate_for_concat = reduced_tensor
    
    # Final concatenation operation - this is the observable output
    final_output = torch.cat([intermediate_for_concat, concat_tensor], 1)
    
    # Must return both the intermediate result (used in concat) and final result (returned by module)
    return intermediate_for_concat, final_output

def replacement_args(input_tensor, weights, bias, concat_tensor):
    """Extract arguments needed for the replacement"""
    return (input_tensor, weights, bias, concat_tensor)

@triton.jit
def identity_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Simple identity kernel that just copies input to output
    This eliminates the redundant stack-sum operations
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    # Store output (identity operation)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_function(input_tensor, weights, bias, concat_tensor):
    """
    Optimized function that eliminates the redundant stack-sum operations
    """
    # Perform conv2d directly (same as original)
    conv_result = torch.conv2d(input_tensor, weights, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # The redundant stack-sum operations are eliminated - we use conv_result directly
    # which is mathematically equivalent to stack([conv_result], dim=0).sum(dim=0)
    
    # Perform the concatenation using the conv result directly
    final_output = torch.cat([conv_result, concat_tensor], 1)
    
    # Return the results that the pattern expects
    return conv_result, final_output

def replacement_func():
    """Returns the optimized function (no arguments)"""
    return optimized_function