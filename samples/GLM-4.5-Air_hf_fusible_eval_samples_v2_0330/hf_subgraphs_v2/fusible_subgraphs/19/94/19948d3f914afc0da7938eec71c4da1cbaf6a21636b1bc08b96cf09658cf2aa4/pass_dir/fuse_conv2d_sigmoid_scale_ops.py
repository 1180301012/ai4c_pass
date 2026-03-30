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
def fused_conv_sigmoid_scale_kernel(
    # Conv2D inputs
    bias_ptr,
    weight_ptr,
    input_ptr,
    # Element-wise multiplier ptr
    scale_ptr,
    # Output
    out_ptr,
    # Tensor shapes
    n_channels_out,
    output_height,
    output_width,
    weight_channels_in,
    # Strides
    out_stride_y,
    out_stride_x,
    scale_stride_y,
    scale_stride_x,
    num_spatial_locations,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles a spatial location block and channel block
    pid_spatial_block = tl.program_id(0)
    pid_channel_block = tl.program_id(1)
    
    # Calculate spatial block bounds
    spatial_per_block = (num_spatial_locations + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    start_spatial = pid_spatial_block * BLOCK_SIZE_X
    end_spatial = min(start_spatial + BLOCK_SIZE_X, num_spatial_locations)
    spatial_mask = start_spatial + tl.arange(0, BLOCK_SIZE_X) < num_spatial_locations
    
    # Calculate channel block bounds  
    start_channel = pid_channel_block * BLOCK_SIZE_C
    end_channel = min(start_channel + BLOCK_SIZE_C, n_channels_out)
    channel_mask = start_channel + tl.arange(0, BLOCK_SIZE_C) < n_channels_out
    
    # Load bias for this channel block
    bias_vals = tl.load(bias_ptr + start_channel + tl.arange(0, BLOCK_SIZE_C), 
                        mask=channel_mask, other=0.0)
    
    # Pre-compute the weighted sum for each spatial location in the block
    spatial_sums = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Create mask for input channels (we only have valid input_channels_in channels)
    input_channel_mask = tl.arange(0, weight_channels_in) < weight_channels_in
    
    for c_in in range(weight_channels_in):
        # Load input value for this input channel (with proper masking)
        input_val = tl.load(input_ptr + c_in, mask=input_channel_mask, other=0.0)[0]
        
        # Load weights for this input channel and all output channels in the block
        weight_offset = (start_channel + tl.arange(0, BLOCK_SIZE_C)) * weight_channels_in + c_in
        weight_vals = tl.load(weight_ptr + weight_offset, mask=channel_mask, other=0.0)
        
        # Broadcast weight values across spatial dimension and accumulate
        spatial_sums += input_val * weight_vals[None, :]
    
    # Add bias and apply sigmoid
    conv_result = spatial_sums + bias_vals[None, :]
    sigmoid_result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Load scale values for this spatial block and channel block
    scale_vals = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_C), dtype=tl.float32)
    for i in range(BLOCK_SIZE_X):
        spatial_idx = start_spatial + i
        if spatial_idx < num_spatial_locations:
            y = spatial_idx // output_width
            x = spatial_idx % output_width
            
            scale_offset = y * scale_stride_y + x * scale_stride_x
            scale_ptr_idx = scale_offset + (start_channel + tl.arange(0, BLOCK_SIZE_C)) * output_height * output_width
            
            scale_vals[i, :] = tl.load(scale_ptr + scale_ptr_idx, mask=channel_mask, other=1.0)
    
    # Apply scaling
    result = sigmoid_result * scale_vals
    
    # Store results for this spatial block and channel block
    for i in range(BLOCK_SIZE_X):
        spatial_idx = start_spatial + i
        if spatial_idx < num_spatial_locations:
            y = spatial_idx // output_width
            x = spatial_idx % output_width
            
            out_offset = y * out_stride_y + x * out_stride_x
            output_base = out_offset + start_channel * output_height * output_width
            
            output_pos = output_base + tl.arange(0, BLOCK_SIZE_C) * output_height * output_width
            tl.store(out_ptr + output_pos, result[i, :], mask=channel_mask)

@torch.fx.wrap
def fused_conv_sigmoid_scale(in_0, in_1, in_2, in_3):
    """
    Fuse conv2d + sigmoid + view + element-wise multiplication + contiguous into single kernel
    """
    # Input shapes
    n_bias = in_0.shape[0]
    weight_shape = in_1.shape  # [n_channels_out, n_channels_in, kh, kw]
    scale_shape = in_2.shape   # [1, n_channels_scale, H, W] or similar
    input_shape = in_3.shape   # [1, n_channels_in, in_h, in_w]
    
    n_channels_out, n_channels_in, kh, kw = weight_shape
    
    # Based on the shapes we see in the weight_meta files:
    # The output spatial dimensions match those of in_2 (the element-wise multiplier)
    if in_2.dim() == 4:  # [1,96,H,W] expected
        batch, scale_channels, out_h, out_w = in_2.shape
        assert n_channels_out == scale_channels, f"Output channels {n_channels_out} don't match scale channels {scale_channels}"
    else:
        raise ValueError(f"Unexpected in_2 shape: {in_2.shape}")
    
    # Create output tensor with the same shape as in_2
    output = torch.empty((1, n_channels_out, out_h, out_w), dtype=in_3.dtype, device=in_3.device)
    
    if n_channels_out == 0:
        return output
    
    # Calculate number of spatial locations
    num_spatial_locations = out_h * out_w
    
    # Block sizes for spatial and channel dimensions
    BLOCK_SIZE_X = 16   # Process 16 spatial locations per block
    BLOCK_SIZE_C = 64   # Process 64 channels per block
    
    # Calculate number of blocks needed
    spatial_blocks = (num_spatial_locations + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    channel_blocks = (n_channels_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    grid = (spatial_blocks, channel_blocks)
    
    fused_conv_sigmoid_scale_kernel[grid](
        # Conv2D inputs
        in_0,                    # bias: [96]
        in_1,                    # weight: [96, weight_channels_in, 1, 1] 
        in_3,                    # input: [1, weight_channels_in, 1, 1]
        # Element-wise multiplier  
        in_2,                    # scale: [1,96,H,W] 
        # Output
        output,                  # output: [1,96,H,W]
        # Tensor shapes
        n_channels_out,
        out_h,
        out_w,
        n_channels_in,           # weight_channels_in
        # Strides
        output.stride(2),        # out_stride_y
        output.stride(3),        # out_stride_x
        in_2.stride(2),          # scale_stride_y
        in_2.stride(3),          # scale_stride_x
        num_spatial_locations,
        BLOCK_SIZE_X,
        BLOCK_SIZE_C,
    )
    
    return output

# Replacement function (returns the fused kernel function)
def replacement_func():
    return fused_conv_sigmoid_scale