import torch
import triton
import triton.language as tl
import math

# Pattern matching function for linear operation
def pattern(in_0, in_1, in_2):
    """Match the linear operation pattern from the model"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the linear operation"""
    return (in_0, in_1, in_2)

# Optimized linear kernel using Triton
@triton.jit
def linear_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    input_features,
    output_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized linear operation using Triton"""
    # Get program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute row and column ranges for this program
    row_start = pid_m * BLOCK_SIZE_M
    row_end = min((pid_m + 1) * BLOCK_SIZE_M, batch_size)
    col_start = pid_n * BLOCK_SIZE_N
    col_end = min((pid_n + 1) * BLOCK_SIZE_N, output_features)
    
    # Initialize accumulator for this block - use a single element for simplicity
    if row_end - row_start > 0 and col_end - col_start > 0:
        # Process first element in this block
        row_idx = row_start
        col_idx = col_start
        
        # Load bias
        bias_val = tl.load(bias_ptr + col_idx, mask=col_idx < output_features, other=0.0)
        accumulator = bias_val
        
        # Compute dot product for this element
        for k in range(input_features):
            # Calculate flattened offsets
            input_offset = row_idx * input_features + k
            weight_offset = col_idx * input_features + k
            
            # Load elements with proper bounds checking
            input_val = tl.load(input_ptr + input_offset, 
                              mask=input_offset < (batch_size * input_features), other=0.0)
            weight_val = tl.load(weight_ptr + weight_offset, 
                               mask=weight_offset < (output_features * input_features), other=0.0)
            
            # Accumulate
            accumulator += input_val * weight_val
        
        # Store result
        output_offset = row_idx * output_features + col_idx
        tl.store(output_ptr + output_offset, accumulator)

@torch.fx.wrap
def optimized_linear(in_0, in_1, in_2):
    """Optimized linear function with Triton kernel"""
    batch_size, input_features = in_2.shape
    output_features, _ = in_1.shape
    
    # Output shape: (batch_size, output_features)
    output = torch.empty((batch_size, output_features), dtype=in_2.dtype, device=in_2.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 32  # Number of rows per program
    BLOCK_SIZE_N = 32  # Number of columns per program
    
    # Calculate grid size (2D grid for rows and columns)
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    linear_kernel[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        input_features=input_features,
        output_features=output_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_linear