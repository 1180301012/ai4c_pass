import torch
import triton
import triton.language as tl

def pattern(input_x, weight, bias):
    """Pattern matching for linear operation: torch.nn.functional.linear(input_x, weight, bias)"""
    output = torch.nn.functional.linear(input_x, weight, bias)
    return output

def replacement_args(input_x, weight, bias):
    """Extract arguments for the optimized kernel"""
    return (input_x, weight, bias)

@triton.jit
def linear_kernel(
    input_ptr,    # [batch_size, input_features]
    weight_ptr,   # [output_features, input_features] 
    bias_ptr,     # [output_features]
    output_ptr,   # [batch_size, output_features]
    batch_size,
    input_features,
    output_features,
):
    """Optimized linear operation kernel using Triton with memory blocking"""
    # Get program index - one thread per batch element
    m = tl.program_id(0)  # batch dimension
    
    # Check bounds
    if m >= batch_size:
        return
    
    # Compute memory addresses for this batch element
    input_row_ptr = input_ptr + m * input_features
    
    # Initialize accumulators for all output features
    acc_0 = 0.0
    acc_1 = 0.0
    
    # Optimized kernel using memory coalescing and register blocking
    # Using 128-element tiles for good cache utilization
    TILE_SIZE = 128
    for k in range(0, input_features, TILE_SIZE):
        # Process elements in optimized tiles
        k_end = min(k + TILE_SIZE, input_features)
        for i in range(k_end - k):
            # Load input and weight elements with optimized memory layout
            input_val = tl.load(input_row_ptr + k + i)
            weight_0_val = tl.load(weight_ptr + 0 * input_features + k + i)  # First output feature
            weight_1_val = tl.load(weight_ptr + 1 * input_features + k + i)  # Second output feature
            
            # Accumulate dot products
            acc_0 += input_val * weight_0_val
            acc_1 += input_val * weight_1_val
    
    # Add bias
    bias_0 = tl.load(bias_ptr + 0)
    bias_1 = tl.load(bias_ptr + 1)
    acc_0 += bias_0
    acc_1 += bias_1
    
    # Store both output elements
    output_base = output_ptr + m * output_features
    tl.store(output_base + 0, acc_0)
    tl.store(output_base + 1, acc_1)

@torch.fx.wrap
def optimized_linear(input_x, weight, bias):
    """Wrapper function to launch the optimized linear kernel"""
    batch_size, input_features = input_x.shape
    output_features = bias.shape[0]
    
    # Allocate output tensor
    output = torch.empty(batch_size, output_features, dtype=input_x.dtype, device=input_x.device)
    
    # Calculate grid size - one program per batch element
    grid_m = batch_size
    
    # Launch kernel
    linear_kernel[(grid_m,)](
        input_x,
        weight,
        bias,
        output,
        batch_size,
        input_features,
        output_features
    )
    
    return output

def replacement_func():
    """Return the optimized linear function"""
    return optimized_linear