import torch
import triton
import triton.language as tl

def pattern(conv_input, arg1, arg2, se_output, _unused1, _unused2, _unused3, _unused4):
    # Flexible SE module: conv2d -> sigmoid -> multiply
    conv_result = torch.conv2d(conv_input, arg1, arg2, (1, 1), (0, 0), (1, 1), 1)
    sigmoid_result = conv_result.sigmoid()
    multiply_result = se_output * sigmoid_result
    # Only return the final result that's actually used ( tmp_8 in original)
    return multiply_result

def replacement_args(conv_input, arg1, arg2, se_output, _unused1, _unused2, _unused3, _unused4):
    return (conv_input, arg1, arg2, se_output, _unused1, _unused2, _unused3, _unused4)

@triton.jit
def se_attention_kernel(
    conv_input_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    se_output_ptr,
    output_ptr,
    n_batch, n_channels_out, n_channels_in, height, width,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Grid mapping: batch, output_channel
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Compute memory addresses
    conv_input_base = pid_n * n_channels_in * height * width
    conv_weight_base = pid_c * n_channels_in
    se_output_base = pid_n * n_channels_out
    output_base = pid_n * n_channels_out * height * width
    
    # Load conv input: [N, C_in, H, W] -> for this SE block, H=W=1
    if height == 1 and width == 1:
        conv_input_val = tl.load(conv_input_ptr + conv_input_base + pid_c)
    else:
        # For general case, load from spatial position (0,0) since SE input is pooled
        conv_input_val = tl.load(conv_input_ptr + conv_input_base + pid_c * height * width)
    
    # Load conv weight and bias: [C_out, C_in, 1, 1]
    conv_weight_val = tl.load(conv_weight_ptr + conv_weight_base + pid_c)
    conv_bias_val = tl.load(conv_bias_ptr + pid_c)
    
    # Conv2D operation (equivalent to linear for 1x1 conv)
    conv_result = conv_input_val * conv_weight_val + conv_bias_val
    
    # Sigmoid activation
    sigmoid_result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Load SE output and multiply
    se_output_val = tl.load(se_output_ptr + se_output_base + pid_c)
    attention_result = se_output_val * sigmoid_result
    
    # Store result
    if height == 1 and width == 1:
        tl.store(output_ptr + output_base + pid_c, attention_result)
    else:
        # Broadcast to spatial dimensions
        for h in range(height):
            for w in range(width):
                output_idx = output_base + pid_c * height * width + h * width + w
                tl.store(output_ptr + output_idx, attention_result)

@torch.fx.wrap
def fused_se_attention(conv_input, conv_weight, conv_bias, se_output):
    # Get input shapes
    n_batch, n_channels_in, height, width = conv_input.shape
    n_out_channels = conv_weight.shape[0]  # [C_out, C_in, 1, 1]
    
    # Output shape should be [N, C_out, H, W] 
    # If se_output is [N, C_out, H, W], then output should match this shape
    if se_output.shape[0] == n_batch and se_output.shape[1] == n_out_channels:
        output_shape = se_output.shape
    else:
        # If shapes don't match, create output based on the output channels
        output_shape = (n_batch, n_out_channels, height, width)
    
    output = torch.empty(output_shape, dtype=se_output.dtype, device=se_output.device)
    
    # Block sizes for efficient GPU utilization
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_C = 64
    
    # Calculate grid dimensions
    grid_n = (n_batch + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (n_out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    se_attention_kernel[grid_n, grid_c](
        conv_input_ptr=conv_input,
        conv_weight_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        se_output_ptr=se_output,
        output_ptr=output,
        n_batch=n_batch,
        n_channels_out=n_out_channels,
        n_channels_in=n_channels_in,
        height=height,
        width=width,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return output

def replacement_func():
    return fused_se_attention