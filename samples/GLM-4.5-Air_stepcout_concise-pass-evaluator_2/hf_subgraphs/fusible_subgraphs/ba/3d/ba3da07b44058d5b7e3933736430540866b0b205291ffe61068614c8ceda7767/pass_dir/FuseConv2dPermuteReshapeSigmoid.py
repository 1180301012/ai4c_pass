import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matching: conv2d -> permute -> reshape -> sigmoid
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(in_2.shape[0], -1, in_0.shape[0])
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the fused kernel
    """
    output_channels = in_0.shape[0]
    return (in_0, in_1, in_2, output_channels)

@triton.jit
def fused_conv_reshape_sigmoid_kernel(
    bias_ptr,
    weight_ptr, 
    input_ptr,
    output_ptr,
    batch_size,
    input_channels,
    height,
    width,
    output_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel: conv2d (1x1) + reshape + sigmoid
    Processes data in a unified manner to avoid intermediate tensor allocations
    """
    # Calculate effective spatial size
    spatial_size = height * width
    
    # Each program handles one output channel across all spatial positions and batch elements
    c_o = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + c_o, mask=None)
    
    # Process spatial positions in chunks
    spatial_offset = tl.program_id(2) * BLOCK_SIZE_N
    spatial_mask = spatial_offset + tl.arange(0, BLOCK_SIZE_N) < spatial_size
    
    # Initialize output accumulator
    out_val = bias  # Start with bias
    
    # Process input channels (grouped by BLOCK_SIZE_K for efficiency)
    for k in range(0, input_channels, BLOCK_SIZE_K):
        # Calculate current k range
        k_start = k
        k_end = min(k + BLOCK_SIZE_K, input_channels)
        k_actual = k_end - k_start
        
        # Load weight slice: [output_channels, input_channels, 1, 1] -> [1, k_actual, 1, 1] -> scalar
        weight_offset = c_o * input_channels + k_start
        weight = tl.load(weight_ptr + weight_offset, mask=None)
        
        # For spatial positions: process in chunks
        spatial_idx_base = batch_id * spatial_size + spatial_offset
        
        for s in range(0, spatial_size, BLOCK_SIZE_N):
            s_end = min(s + BLOCK_SIZE_N, spatial_size)
            s_actual = s_end - s
            
            # Load input slice for this spatial position and k range
            input_offset = (batch_id * input_channels + k_start) * spatial_size + s
            input_val = tl.load(input_ptr + input_offset, mask=s + tl.arange(0, s_actual) < spatial_size)
            
            # Perform weighted sum (convolution operation)
            out_val += weight * input_val
    
    # Apply sigmoid activation
    out_val = 1.0 / (1.0 + tl.exp(-out_val))
    
    # Store output directly in the final shape [batch_size, spatial_size, output_channels]
    output_offset = (batch_id * spatial_size + spatial_offset) * output_channels + c_o
    tl.store(output_ptr + output_offset, out_val, mask=spatial_offset + tl.arange(0, BLOCK_SIZE_N) < spatial_size)

@torch.fx.wrap  
def fused_conv_reshape_sigmoid(bias, weight, input_tensor):
    """
    Wrapper function to launch the fused kernel
    """
    batch_size, input_channels, height, width = input_tensor.shape
    output_channels = bias.shape[0]
    spatial_size = height * width
    
    # Output shape: [batch_size, spatial_size, output_channels]
    output = torch.empty((batch_size, spatial_size, output_channels), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up grid dimensions
    # Program 0: output channels
    # Program 1: batch elements  
    # Program 2: spatial chunks
    BLOCK_SIZE_M = 1  # Output channels processed per program
    BLOCK_SIZE_N = 1024  # Spatial positions per program
    BLOCK_SIZE_K = 256  # Input channels per iteration
    
    grid_x = output_channels
    grid_y = batch_size
    grid_z = (spatial_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_conv_reshape_sigmoid_kernel[(grid_x, grid_y, grid_z)](
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        input_channels=input_channels,
        height=height,
        width=width,
        output_channels=output_channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    """
    Returns the fused kernel function
    """
    return fused_conv_reshape_sigmoid