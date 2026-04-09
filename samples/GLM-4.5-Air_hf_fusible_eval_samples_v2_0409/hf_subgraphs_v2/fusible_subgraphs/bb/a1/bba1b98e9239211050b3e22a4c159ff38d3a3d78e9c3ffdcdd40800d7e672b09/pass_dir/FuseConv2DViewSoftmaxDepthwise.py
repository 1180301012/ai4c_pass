import torch
import triton
import triton.language as tl

# Pattern matching function - matches Conv2D → View → Softmax sequence
def pattern(in_2, in_1, in_0):
    """Match the Conv2D → View → Softmax pattern for depthwise normalization"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # The view operation varies by batch size, but we can match any [N, 1, -1] pattern
    tmp_3 = conv2d.view(conv2d.shape[0], 1, -1)
    softmax_result = tmp_3.softmax(dim=-1)
    return softmax_result

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Triton kernel implementation - fused Conv2D + Softmax with depthwise optimization
@triton.jit
def depthwise_conv2d_softmax_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Fused depthwise convolution + softmax kernel with optimized memory access"""
    
    # Program IDs for batch and spatial dimensions
    batch_id = tl.program_id(0)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(2)
    
    # Create pointers for each element in the flattened tensor (batch, 1, spatial)
    input_ptr_batch = input_ptr + batch_id * channels * height * width
    
    # Process spatial positions in blocks
    offsets_h = h_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_w = w_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create 2D offset for spatial positions
    spatial_offsets = offsets_h[:, None] * width + offsets_w[None, :]
    
    mask = (spatial_offsets < (height * width))
    
    # Load input data for this batch and spatial positions
    input_data = tl.load(
        input_ptr_batch + spatial_offsets,
        mask=mask,
        other=0.0
    )
    
    # Reshape to (channels, BLOCK_H, BLOCK_W) for depthwise convolution
    input_reshaped = input_data.reshape(BLOCK_H, BLOCK_W, channels).permute(2, 0, 1)
    
    # Initialize output accumulator for this block
    output_block = tl.zeros((channels, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Depthwise convolution: each channel processed independently
    for c in range(0, channels, BLOCK_C):
        # Process channels in blocks
        current_channels = min(BLOCK_C, channels - c)
        
        # Load weight and bias (they are [1, C, 1, 1], so we load first channel)
        weight_val = tl.load(weight_ptr + c, mask=(c < channels))
        bias_val = tl.load(bias_ptr + c, mask=(c < channels))
        
        # Apply depthwise convolution (element-wise multiplication + bias)
        # For depthwise conv with 1x1 kernel, it's just element-wise operation
        if current_channels > 0:
            # Extract input for current channel block
            channel_input = input_reshaped[c:c+current_channels]
            
            # Apply depthwise operation (element-wise multiplication with bias)
            # Since kernel is 1x1, it's just input * weight + bias
            output_c = channel_input * weight_val + bias_val
            
            # Store result
            if c == 0:  # First channel becomes our output since we have depthwise + view
                output_mask = mask & (tl.zeros_like(mask) == 0)  # Use first channel for softmax
                output_flat = output_c[:, :, 0].reshape(-1)  # Flatten spatial dimensions
                
                # Apply softmax along the flattened dimension
                max_val = tl.max(output_flat)
                exp_val = tl.exp(output_flat - max_val)
                sum_exp = tl.sum(exp_val)
                softmax_result = exp_val / sum_exp
                
                # Store softmax result
                tl.store(output_ptr + batch_id * height * width + spatial_offsets[0, 0] + tl.arange(0, BLOCK_H * BLOCK_W),
                        softmax_result, mask=spatial_offsets[0, 0] + tl.arange(0, BLOCK_H * BLOCK_W) < height * width)

# Kernel wrapper that handles the fused operation
@torch.fx.wrap
def fused_depthwise_conv2d_softmax(input, weight, bias):
    """
    Fused depthwise convolution + softmax with optimized memory access patterns
    """
    batch_size, channels, height, width = input.shape
    
    # Calculate grid size
    # We'll use blocks that work well with GPU architecture
    BLOCK_H, BLOCK_W = 8, 8
    BLOCK_C = min(32, channels)
    
    # Calculate grid dimensions
    h_blocks = (height + BLOCK_H - 1) // BLOCK_H
    w_blocks = (width + BLOCK_W - 1) // BLOCK_W
    
    # Create output tensor
    output = torch.empty((batch_size, 1, height * width), dtype=input.dtype, device=input.device)
    
    # Launch the kernel
    grid = (batch_size, h_blocks, w_blocks)
    
    fused_depthwise_conv2d_softmax_kernel[grid](
        input,
        weight,
        bias,
        output,
        batch_size,
        channels,
        height,
        width,
        BLOCK_C,
        BLOCK_H,
        BLOCK_W
    )
    
    return output

# Replacement function (returns the fused kernel function)
def replacement_func():
    return fused_depthwise_conv2d_softmax