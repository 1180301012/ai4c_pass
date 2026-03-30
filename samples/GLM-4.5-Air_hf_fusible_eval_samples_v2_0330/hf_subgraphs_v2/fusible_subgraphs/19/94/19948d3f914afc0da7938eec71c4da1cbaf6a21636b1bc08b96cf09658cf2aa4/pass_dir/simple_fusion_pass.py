import torch
import triton
import triton.language as tl

# Pattern matching function - exactly mirrors the computation in model.py
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_fusion_kernel(
    # Input pointers
    bias_ptr,
    weight_ptr,
    input_ptr,
    scale_ptr,
    # Output pointer  
    out_ptr,
    # Tensor shapes
    channels_out,
    height,
    width,
    # Strides
    out_stride_y,
    out_stride_x,
    scale_stride_y,
    scale_stride_x,
    # Constant parameters
    channels_in: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute program coordinates
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1) 
    pid_c = tl.program_id(2)
    
    # Check bounds - avoid chained boolean operators
    if pid_y >= height:
        return
    if pid_x >= width:
        return
    if pid_c >= channels_out:
        return
    
    # Process block of channels
    start_channel = pid_c * BLOCK_SIZE_C
    channel_offsets = start_channel + tl.arange(0, BLOCK_SIZE_C)
    channel_mask = channel_offsets < channels_out
    
    # Load bias for this channel block
    bias_vals = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0)
    
    # Compute weighted sum for each input channel
    sum_vals = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # Load each input channel individually and accumulate
    for c_in in range(channels_in):
        # Load input value for this channel (from [1, channels_in, 1, 1])
        input_val = tl.load(input_ptr + c_in)
        
        # Load weights for this input channel and channel block
        weight_offset = channel_offsets * channels_in + c_in
        weight_vals = tl.load(weight_ptr + weight_offset, mask=channel_mask, other=0.0)
        
        # Add contribution from this input channel
        sum_vals += input_val * weight_vals
    
    # Add bias and apply sigmoid
    conv_result = sum_vals + bias_vals
    sigmoid_result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Load scale value for this spatial location (broadcast across channels)
    scale_offset = pid_y * scale_stride_y + pid_x * scale_stride_x
    scale_vals = tl.load(scale_ptr + scale_offset)
    
    # Apply scaling to each channel in the block
    result = sigmoid_result * scale_vals
    
    # Store result
    out_offset = pid_y * out_stride_y + pid_x * out_stride_x
    result_pos = out_offset + channel_offsets * height * width
    tl.store(out_ptr + result_pos, result, mask=channel_mask)

@torch.fx.wrap
def simple_fusion(in_0, in_1, in_2, in_3):
    """
    Simple fusion of conv2d + sigmoid + scale
    """
    # Extract tensor shapes
    n_bias = in_0.shape[0]
    n_channels_out, n_channels_in, kh, kw = in_1.shape
    batch, scale_channels, out_h, out_w = in_2.shape
    
    assert n_channels_out == scale_channels, f"Channel mismatch: {n_channels_out} vs {scale_channels}"
    
    # Create output tensor
    output = torch.empty((1, n_channels_out, out_h, out_w), dtype=in_3.dtype, device=in_3.device)
    
    if n_channels_out == 0 or out_h == 0 or out_w == 0:
        return output
    
    # Adaptive block sizes based on tensor dimensions for better GPU occupancy
    if out_h * out_w >= 4096:  # Large spatial dimensions
        BLOCK_SIZE_Y = 8    # Smaller blocks for better parallelism
        BLOCK_SIZE_X = 8
        BLOCK_SIZE_C = 128   # More channels per block
    elif n_channels_out >= 128:  # Many channels
        BLOCK_SIZE_Y = 16   
        BLOCK_SIZE_X = 16  
        BLOCK_SIZE_C = 32   # Fewer channels per block
    else:  # Default configuration
        BLOCK_SIZE_Y = 16   
        BLOCK_SIZE_X = 16  
        BLOCK_SIZE_C = 64   
    
    # Calculate grid dimensions
    grid_y = (out_h + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (out_w + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_c = (n_channels_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    grid = (grid_y, grid_x, grid_c)
    
    simple_fusion_kernel[grid](
        in_0,                    # bias: [n_channels_out]
        in_1,                    # weight: [n_channels_out, n_channels_in, 1, 1]
        in_3,                    # input: [1, n_channels_in, 1, 1]
        in_2,                    # scale: [1, n_channels_out, out_h, out_w]
        output,                  # output: [1, n_channels_out, out_h, out_w]
        n_channels_out,
        out_h,
        out_w,
        output.stride(2),        # out_stride_y
        output.stride(3),        # out_stride_x
        in_2.stride(2),          # scale_stride_y
        in_2.stride(3),          # scale_stride_x
        n_channels_in,           # channels_in: compile-time constant
        BLOCK_SIZE_Y,
        BLOCK_SIZE_X, 
        BLOCK_SIZE_C,
    )
    
    return output

# Replacement function
def replacement_func():
    return simple_fusion