import torch
import triton
import triton.language as tl

def pattern(input_2):
    """
    Pattern matching: transpose(-2, -1) operation
    
    This optimizes transposing the last two dimensions of a tensor.
    Original: input_2.transpose(-2, -1) 
    Optimized: Custom Triton kernel with optimized memory access
    """
    result = input_2.transpose(-2, -1)
    return result

def replacement_args(input_2):
    """
    Extract arguments for the optimized kernel
    """
    return (input_2,)

@triton.jit
def transpose_kernel(
    input_ptr,      # Input tensor: [1, 16, 196, 48]
    output_ptr,     # Output tensor: [1, 16, 48, 196]
    
    # Dimensions
    N: tl.constexpr,    # 1 (batch size)
    C: tl.constexpr,    # 16 (channels)
    H_in: tl.constexpr, # 196 (input height)
    W_in: tl.constexpr, # 48 (input width)
    HW: tl.constexpr,   # H_in * W_in (flattened spatial dimensions)
    
    BLOCK_SIZE_HW: tl.constexpr
):
    """
    Optimized kernel for transpose(-2, -1) operation.
    Efficiently transposes the last two dimensions using flattened spatial approach.
    """
    
    # Determine which position to process: [batch, channel, spatial_position]
    b_id = tl.program_id(0)  # batch dimension
    c_id = tl.program_id(1)  # channel dimension
    hw_id = tl.program_id(2)  # flattened spatial position
    
    # Check bounds - avoid chained boolean operators
    if b_id >= N:
        return
    if c_id >= C:
        return
    if hw_id >= HW:
        return
        
    # Convert flattened hw_id back to h, w coordinates
    h_id = hw_id // W_in
    w_id = hw_id % W_in
    
    # Input/Output strides for memory layout
    input_stride = C * H_in * W_in
    output_stride = C * W_in * H_in
    
    # Load input data from [b, c, h_id, w_id]
    input_idx = (b_id * input_stride + 
                c_id * H_in * W_in + 
                h_id * W_in + w_id)
    
    # Load single element
    input_val = tl.load(input_ptr + input_idx)
    
    # Store output data to [b, c, w_id, h_id] (transposed)
    output_idx = (b_id * output_stride + 
                 c_id * W_in * H_in + 
                 w_id * H_in + h_id)
    
    tl.store(output_ptr + output_idx, input_val)

@torch.fx.wrap  
def optimized_transpose(input_tensor):
    """
    Wrapper function that launches the optimized transpose kernel
    """
    N, C, H_in, W_in = input_tensor.shape
    
    # Output is [N, C, W_in, H_in] (last two dimensions transposed)
    output_shape = [N, C, W_in, H_in]
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flattened spatial dimensions
    HW = H_in * W_in
    
    # Launch kernel with 3D grid: [batch, channels, spatial_positions]
    grid = (N, C, HW)
    
    transpose_kernel[grid](
        input_tensor,
        output,
        N, C, H_in, W_in, HW,
        BLOCK_SIZE_HW=1  # Process each spatial position individually
    )
    
    return output

def replacement_func():
    """
    Return the optimized function reference
    """
    return optimized_transpose