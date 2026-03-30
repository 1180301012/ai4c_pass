import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern: Conv2D + Sigmoid + Multiplication + Hardtanh fusion"""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv_sigmoid_mul_hardtanh_kernel(
    x_se_ptr, weight_ptr, bias_ptr, x_29_ptr, out_ptr,
    batch_size, in_channels_c, out_channels_c, height, width,
    BLOCK_SIZE_C: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
):
    """
    Simplified fused kernel: 1x1 Conv2D + Sigmoid + Element-wise Multiplication + Hardtanh
    
    This kernel computes the channel attention mechanism efficiently:
    1. 1x1 conv: [batch, 19, 1, 1] x [228, 19, 1, 1] + [228] -> [batch, 228, 1, 1]  
    2. Sigmoid: sigmoid(conv_output) -> [batch, 228, 1, 1]
    3. Broadcast multiply: [batch, 228, 28, 28] * sigmoid_out -> [batch, 228, 28, 28]
    4. Hardtanh: hardtanh(x, 0, 6)
    """
    # Process each output channel independently 
    pid_b = tl.program_id(0)  # Batch dimension
    pid_c = tl.program_id(1)  # Output channel dimension
    
    # Check bounds
    batch_mask = pid_b < batch_size
    channel_mask = pid_c < out_channels_c
    
    if not (batch_mask and channel_mask):
        return
        
    # Load input x_se for this batch: [19, 1, 1]
    x_se_batch_ptr = x_se_ptr + pid_b * IN_CHANNELS * 1 * 1
    x_se = tl.load(x_se_batch_ptr + tl.arange(0, IN_CHANNELS), 
                  mask=True).to(tl.float32)  # [IN_CHANNELS]
    
    # Load weight for this output channel: [19, 1, 1]
    weight_channel_ptr = weight_ptr + pid_c * IN_CHANNELS * 1 * 1
    weight = tl.load(weight_channel_ptr + tl.arange(0, IN_CHANNELS), 
                     mask=True).to(tl.float32)  # [IN_CHANNELS]
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + pid_c, mask=True).to(tl.float32)
    
    # Compute 1x1 convolution output: sum(x_se * weight) + bias
    conv_output = tl.sum(x_se * weight) + bias
    
    # Apply sigmoid
    sigmoid_out = 1.0 / (1.0 + tl.exp(-conv_output))
    
    # Process all spatial locations for this batch and channel
    for spatial_offset in range(0, height * width, BLOCK_SIZE_C):
        spatial_block = spatial_offset + tl.arange(0, BLOCK_SIZE_C)
        spatial_mask = spatial_block < height * width
            
        # Load x_29 input: [batch, out_channels, height, width]
        x_29_ptr_batch = x_29_ptr + pid_b * out_channels_c * height * width + spatial_block * out_channels_c + pid_c
        x_29 = tl.load(x_29_ptr_batch, mask=spatial_mask).to(tl.float32)
        
        # Apply element-wise multiplication (sigmoid_out broadcasts to spatial dimensions)
        mul_out = x_29 * sigmoid_out
        
        # Apply hardtanh: max(0, min(6, x))
        hardtanh_out = tl.maximum(tl.minimum(mul_out, 6.0), 0.0)
        
        # Store result
        out_ptr_batch = out_ptr + pid_b * out_channels_c * height * width + spatial_block * out_channels_c + pid_c
        tl.store(out_ptr_batch, hardtanh_out, mask=spatial_mask)

@torch.fx.wrap
def fused_conv_sigmoid_mul_hardtanh(in_0, in_1, in_2, in_3):
    """Wrapper function for the fused kernel"""
    # Get tensor shapes
    batch_size = in_3.shape[0]
    out_channels = in_0.shape[0]  # bias shape gives output channels
    _, _, height, width = in_2.shape
    
    # Constants for the kernel
    IN_CHANNELS = 32  # Power of 2 constant for tl.arange (actual channels are 19)
    
    # Optimize block size based on spatial dimensions for better GPU occupancy
    spatial_size = height * width
    if spatial_size <= 128:
        BLOCK_SIZE_C = 128  # Small tensors use smaller blocks
    elif spatial_size <= 1024:
        BLOCK_SIZE_C = 256  # Medium tensors use moderate blocks  
    else:
        BLOCK_SIZE_C = 512  # Large tensors use larger blocks
    
    # Validate that we have the expected input channels (in_channels should be 19)
    actual_input_channels = in_3.shape[1]
    if actual_input_channels != 19:
        raise ValueError(f"Expected 19 input channels, got {actual_input_channels}")
    
    # Calculate output shape: [batch, out_channels, height, width]
    output_shape = (batch_size, out_channels, height, width)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel with grid: [batch_size, out_channels]
    fused_conv_sigmoid_mul_hardtanh_kernel[(batch_size, out_channels)](
        x_se_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        x_29_ptr=in_2,
        out_ptr=output,
        batch_size=batch_size,
        in_channels_c=actual_input_channels,
        out_channels_c=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        IN_CHANNELS=IN_CHANNELS,
    )
    
    return output

def replacement_func():
    """Return the fused function"""
    return fused_conv_sigmoid_mul_hardtanh