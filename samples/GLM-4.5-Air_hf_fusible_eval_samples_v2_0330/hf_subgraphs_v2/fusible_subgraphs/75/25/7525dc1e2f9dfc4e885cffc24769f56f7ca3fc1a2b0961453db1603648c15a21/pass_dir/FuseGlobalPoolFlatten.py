import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(tmp_4):
    """
    Match the computation pattern:
    adaptive_avg_pool2d with size 1, then flatten, then dropout with rate 0.0
    """
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7

# Argument extraction function
def replacement_args(tmp_4):
    return (tmp_4,)

# Triton kernel for fused global average pooling and flatten
@triton.jit
def fused_global_pool_flatten_kernel(
    input_ptr,           # input: [batch, channels, height, width]
    output_ptr,          # output: [batch, channels]
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program identifiers
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Total spatial elements for work distribution
    total_spatial = height * width
    spatial_elements_per_program = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = spatial_idx * spatial_elements_per_program
    
    # Clamp end index to avoid overflow
    end_idx = min(start_idx + spatial_elements_per_program, total_spatial)
    
    # Initialize accumulator for this batch-channel combination
    accumulator = 0.0
    
    # Process spatial elements in the work range
    for spatial_pos in range(start_idx, end_idx):
        # Calculate spatial coordinates
        h = spatial_pos // width
        w = spatial_pos % width
        
        # Calculate input pointer offset
        input_offset = (batch_idx * channels * height * width + 
                       spatial_idx * height * width + 
                       h * width + w)
        
        # Load input element and accumulate
        element = tl.load(input_ptr + input_offset, mask=(spatial_pos < total_spatial))
        accumulator += element
    
    # Calculate average by dividing by spatial size
    average = accumulator / total_spatial if total_spatial > 0 else 0.0
    
    # Store result (global average pooling effectively outputs the average)
    output_offset = batch_idx * channels + spatial_idx
    tl.store(output_ptr + output_offset, average)

# Faster kernel for global average pooling (more efficient)
@triton.jit
def simple_global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Simple global average pooling kernel"""
    # Block identifiers
    m = tl.program_id(0)  # batch dimension  
    n = tl.program_id(1)  # channel dimension
    
    # Each program handles one batch-channel pair
    spatial_elements = height * width
    base_offset = m * channels * height * width + n * height * width
    
    # Initialize sum accumulator
    sum_val = 0.0
    
    # Load all spatial elements using a fixed block size
    # Use smaller fixed block size for compatibility
    elem_per_program = min(BLOCK_SIZE_N, spatial_elements)
    
    # Load elements with proper masking
    for i in range(elem_per_program):
        offset = base_offset + i
        if i < spatial_elements:
            element = tl.load(input_ptr + offset)
            sum_val += element
    
    # Compute average
    avg_val = sum_val / spatial_elements if spatial_elements > 0 else 0.0
    
    # Store result  
    output_offset = m * channels + n
    tl.store(output_ptr + output_offset, avg_val)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_global_pool_flatten_dropout(input_tensor):
    # Get tensor shapes
    batch_size = input_tensor.shape[0]
    channels = input_tensor.shape[1]
    height = input_tensor.shape[2]
    width = input_tensor.shape[3]
    
    # Create output tensor (flattened pooling)
    output = torch.empty(batch_size, channels, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use efficient kernel
    grid_x = batch_size
    grid_y = channels
    
    # Launch kernel with appropriate block sizes
    simple_global_avg_pool_kernel[(
        grid_x, 
        grid_y,
    )](
        input_tensor,
        output,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE_M=1,      # Each program handles one batch
        BLOCK_SIZE_N=1024,   # Process multiple spatial elements per channel
    )
    
    # Dropout with rate 0.0 is effectively no-op, so we just return the result
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_global_pool_flatten_dropout