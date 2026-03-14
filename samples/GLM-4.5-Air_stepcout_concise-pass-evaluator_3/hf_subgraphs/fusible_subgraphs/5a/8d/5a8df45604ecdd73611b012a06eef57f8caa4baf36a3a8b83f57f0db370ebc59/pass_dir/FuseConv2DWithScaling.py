import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, scale_factor, conv_output):
    """
    Pattern matches: Conv2D -> (no-op dropout) -> scale -> add_with_residual
    Eliminates no-op dropout and fuses Conv2D with scaling operations.
    """
    # No-op dropout with p=0.0 is eliminated
    scaled_conv = scale_factor.unsqueeze(-1).unsqueeze(-1) * conv_output
    result = conv_input + scaled_conv
    return result

def replacement_args(conv_input, conv_weight, conv_bias, scale_factor, conv_output):
    return (conv_input, conv_weight, conv_bias, scale_factor, conv_output)

@triton.jit
def fused_conv_scale_kernel(
    input_ptr, weight_ptr, bias_ptr, scale_ptr, output_ptr,
    batch_size, input_channels, output_channels, height, width,
    BLOCK_SIZE_m: tl.constexpr, BLOCK_SIZE_n: tl.constexpr, BLOCK_SIZE_k: tl.constexpr
):
    """
    Fused kernel that combines Conv2D with channel-wise scaling.
    Input shapes:
    - input: [batch_size, input_channels, height, width]
    - weight: [output_channels, input_channels, 1, 1] 
    - bias: [output_channels]
    - scale: [output_channels] (will be broadcasted to [output_channels, 1, 1])
    - output: [batch_size, output_channels, height, width]
    """
    # Program identifiers for 2D grid over batch and output channels
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    channel_offset = out_channel_idx * BLOCK_SIZE_n
    
    # Load bias and scale for this output channel
    bias = tl.load(bias_ptr + out_channel_idx, mask=out_channel_idx < output_channels, other=0.0)
    scale = tl.load(scale_ptr + out_channel_idx, mask=out_channel_idx < output_channels, other=1.0)
    
    # Load weight (1x1 conv, so just [input_channels])
    weight = tl.load(weight_ptr + out_channel_idx * input_channels + tl.arange(0, BLOCK_SIZE_k),
                     mask=tl.arange(0, BLOCK_SIZE_k) < input_channels)
    
    # Compute for spatial positions
    for h in range(0, height, BLOCK_SIZE_m):
        for w in range(0, width, BLOCK_SIZE_n):
            # Load input patch 
            input_offsets = batch_idx * input_channels * height * width + \
                           tl.arange(0, BLOCK_SIZE_k) * height * width + h * width + w + tl.arange(0, BLOCK_SIZE_n)
            inputs = tl.load(input_ptr + input_offsets,
                            mask=(tl.arange(0, BLOCK_SIZE_k)[:, None] < input_channels) &
                                  (h + tl.arange(0, BLOCK_SIZE_m)[:, None] < height) &
                                  (w + tl.arange(0, BLOCK_SIZE_n)[None, :] < width))
            
            # Conv2D operation (1x1 conv is essentially matrix multiplication)
            conv_val = tl.sum(inputs * weight[:, None], axis=0) + bias
            
            # Apply scaling
            scaled_val = conv_val * scale
            
            # Store output
            output_offset = batch_idx * output_channels * height * width + \
                           out_channel_idx * height * width + h * width + w + tl.arange(0, BLOCK_SIZE_n)
            tl.store(output_ptr + output_offset, scaled_val,
                    mask=(h + tl.arange(0, BLOCK_SIZE_m)[:, None] < height) &
                         (w + tl.arange(0, BLOCK_SIZE_n)[None, :] < width))

@torch.fx.wrap  
def fused_conv_scale(input_tensor, weight_tensor, bias_tensor, scale_factor):
    batch_size, input_channels, height, width = input_tensor.shape
    output_channels = weight_tensor.shape[0]
    
    # Triton kernel launch configuration
    BLOCK_SIZE_m = 16  # spatial height tile
    BLOCK_SIZE_n = 16  # spatial width tile  
    BLOCK_SIZE_k = 128  # input channels tile
    
    # Calculate grid dimensions
    grid_m = (height + BLOCK_SIZE_m - 1) // BLOCK_SIZE_m
    grid_n = (width + BLOCK_SIZE_n - 1) // BLOCK_SIZE_n
    grid_size = (batch_size, output_channels, grid_m, grid_n)
    
    # Create output tensor
    output = torch.empty((batch_size, output_channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_conv_scale_kernel[grid_size](
        input_tensor, weight_tensor, bias_tensor, scale_factor, output,
        batch_size, input_channels, output_channels, height, width,
        BLOCK_SIZE_m, BLOCK_SIZE_n, BLOCK_SIZE_k
    )
    
    return output

def replacement_func():
    return fused_conv_scale