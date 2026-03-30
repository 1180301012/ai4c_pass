import torch
import triton
import triton.language as tl
import math

# Pattern matching function for mean operation
def pattern(in_3):
    """Match the mean operation pattern from the model"""
    tmp_3 = in_3.mean(-2)
    return tmp_3

# Argument extraction function
def replacement_args(in_3):
    """Extract arguments for the mean operation"""
    return (in_3,)

# Optimized mean kernel using Triton
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
):
    """Optimized mean reduction along dimension -2 using Triton"""
    # Program ids for batch and features dimensions
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)
    
    # Compute ranges for this program
    b_start = pid_b * BLOCK_SIZE_B
    b_end = min((pid_b + 1) * BLOCK_SIZE_B, batch_size)
    f_start = pid_f * BLOCK_SIZE_F
    f_end = min((pid_f + 1) * BLOCK_SIZE_F, features)
    
    # Process a single element per program for simplicity
    if b_end - b_start > 0 and f_end - f_start > 0:
        # Process first element in this block
        b_idx = b_start
        f_idx = f_start
        
        # Compute mean for this batch and feature
        accumulator = 0.0
        
        # Loop over sequence dimension (dimension -2)
        for s in range(seq_len):
            # Calculate flattened offset: [batch, seq, feature] -> batch*seq_len*features + s*features + f
            input_offset = b_idx * seq_len * features + s * features + f_idx
            
            # Load element with bounds checking
            element = tl.load(input_ptr + input_offset, 
                            mask=input_offset < (batch_size * seq_len * features), other=0.0)
            
            # Accumulate
            accumulator += element
        
        # Compute mean
        mean_val = accumulator / seq_len
        
        # Store result
        output_offset = b_idx * features + f_idx
        tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_mean(in_3):
    """Optimized mean function with Triton kernel"""
    batch_size, seq_len, features = in_3.shape
    
    # Output shape: (batch_size, features) - mean over seq_len (dimension -2)
    output = torch.empty((batch_size, features), dtype=in_3.dtype, device=in_3.device)
    
    # Define block sizes for better performance
    BLOCK_SIZE_B = 64  # Block size for batch dimension
    BLOCK_SIZE_F = 64  # Block size for feature dimension
    
    # Calculate grid size
    grid_b = (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_f = (features + BLOCK_SIZE_F - 1) // BLOCK_SIZE_F
    grid = (grid_b, grid_f)
    
    # Launch kernel
    mean_kernel[grid](
        input_ptr=in_3,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_F=BLOCK_SIZE_F,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_mean