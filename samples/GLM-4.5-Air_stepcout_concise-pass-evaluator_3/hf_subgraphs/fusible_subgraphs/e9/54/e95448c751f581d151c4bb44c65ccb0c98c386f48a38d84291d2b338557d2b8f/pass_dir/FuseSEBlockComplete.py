import torch
import triton
import triton.language as tl

@triton.jit
def fused_se_kernel(
    # Input tensors
    x_ptr,          # in_3: [B, C, 1, 1] or [B, C, H, W]
    conv_weight_ptr, # in_1: [C_out, C_in, 1, 1] 
    conv_bias_ptr,   # in_0: [C_out]
    scale_ptr,       # in_2: [B, C_scale, H_scale, W_scale]
    # Output
    out_ptr,         # [B, C_out]
    # Tensor shapes
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_height: tl.constexpr,
    in_width: tl.constexpr,
    scale_channels: tl.constexpr,
    scale_spatial_height: tl.constexpr,
    scale_spatial_width: tl.constexpr,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch and one output channel
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate the starting index for this thread
    start_idx = pid_b * out_channels + pid_c
    
    if pid_b >= batch_size or pid_c >= out_channels:
        return
    
    # Load bias (shared by all elements in this channel)
    conv_bias = tl.load(conv_bias_ptr + pid_c)
    
    # Process each spatial location sequentially due to small spatial size
    total_sum = 0.0
    
    for h in range(in_height):
        for w in range(in_width):
            # Load input (conv input)
            if in_height == 1 and in_width == 1:
                # For bottleneck case: [B, C, 1, 1]
                idx_x = pid_b * in_channels + (h * in_width + w) * batch_size
                x_val = tl.load(x_ptr + idx_x)
            else:
                # For other cases: [B, C, H, W] 
                idx_x = ((pid_b * in_channels + (h * in_width + w)) * batch_size + h) * in_width + w
                x_val = tl.load(x_ptr + idx_x)
            
            # Load conv weight (for this output channel and input channel)
            # Since it's 1x1 conv, we sum over input channels
            conv_sum = conv_bias
            for c_in in range(in_channels):
                weight_idx = (pid_c * in_channels + c_in) * 1 * 1 + 0 * 1 + 0
                conv_sum += tl.load(conv_weight_ptr + weight_idx) * x_val
            
            # Apply sigmoid
            sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_sum))
            
            # Load scale tensor - correct indexing for [B, C_scale, H_scale, W_scale] layout
            # The scale tensor might have a different memory layout than expected
            # Access the scale value for the current batch and output channel at center spatial location
            scale_h = scale_spatial_height // 2
            scale_w = scale_spatial_width // 2
            
            # Try different indexing strategies to match the expected memory layout
            # Layout 1: [B, C, H, W] - standard layout
            scale_idx = (pid_b * scale_channels * scale_spatial_height * scale_spatial_width + 
                        pid_c * scale_spatial_height * scale_spatial_width + 
                        scale_h * scale_spatial_width + scale_w)
            
            scale_val = tl.load(scale_ptr + scale_idx)
            
            fused_out = sigmoid_val * scale_val
            
            # Apply GELU: simplified approximation x * sigmoid(sqrt(2/pi) * x)
            gelu_val = fused_out * 1.0 / (1.0 + tl.exp(-0.7978845608 * fused_out))
            
            # Sum for global average pooling
            total_sum += gelu_val
    
    # Compute global average pooling  
    avg_val = total_sum / (in_height * in_width)
    
    # Store final result - output should be [B, C_out]
    batch_idx = pid_b
    channel_idx = pid_c  
    out_idx = batch_idx * out_channels + channel_idx
    tl.store(out_ptr + out_idx, avg_val)

@torch.fx.wrap  
def fused_se_block(x, conv_weight, conv_bias, scale):
    """
    Fused implementation of:
    1. Conv2D + Sigmoid + Element-wise multiply + GELU
    2. Adaptive average pooling (effectfully global)  
    3. Flatten
    
    Args:
        x: Conv bias tensor [C_out]
        conv_weight: Conv weights [C_out, C_in, 1, 1]
        conv_bias: Element-wise multiplier [B, C, H, W] 
        scale: Input to conv [B, C_in, 1, 1]
    
    Returns:
        Output [B, C_out]
    """
    device = scale.device
    
    # Get conv input shapes (scale is actually the conv input)
    batch_size = scale.shape[0]
    in_channels = scale.shape[1]
    in_height = scale.shape[2]
    in_width = scale.shape[3]
    
    # Get output channels from weights
    out_channels = conv_weight.shape[0]
    
    # Get scale tensor dimensions
    scale_channels = conv_bias.shape[1] 
    scale_spatial_height = conv_bias.shape[2]
    scale_spatial_width = conv_bias.shape[3]
    
    # Calculate output shape
    out_size = batch_size * out_channels
    
    # Create output tensor
    output = torch.empty(out_size, dtype=torch.float32, device=device)
    
    # Launch Triton kernel
    grid = (batch_size, out_channels)
    fused_se_kernel[grid](
        x_ptr=scale,
        conv_weight_ptr=conv_weight,
        conv_bias_ptr=x,
        scale_ptr=conv_bias,
        out_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        scale_channels=scale_channels,
        scale_spatial_height=scale_spatial_height,
        scale_spatial_width=scale_spatial_width,
        BLOCK_SIZE=1024,
    )
    
    return output

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching the SE (Squeeze-and-Excitation) block computation:
    1. Conv2D with sigmoid activation
    2. Element-wise multiplication with scale
    3. GELU activation
    4. Global average pooling followed by flatten
    5. Dropout (with 0.0 probability - no-op)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    """Return the fused kernel function"""
    return fused_se_block