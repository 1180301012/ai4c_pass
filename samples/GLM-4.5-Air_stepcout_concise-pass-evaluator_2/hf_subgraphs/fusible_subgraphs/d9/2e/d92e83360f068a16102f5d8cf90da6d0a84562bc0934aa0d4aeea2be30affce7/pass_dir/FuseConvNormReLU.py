import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    """Pattern for fusing Conv2D + LayerNorm + ReLU operations"""
    # Conv2D: bias, weight, input, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    tmp_4 = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # LayerNorm: normalized_shape, weight, bias, eps
    tmp_5 = torch.nn.functional.layer_norm(tmp_4, (tmp_4.size(1), 1, 1), in_3, in_2, 1e-05)
    # ReLU with inplace=True
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def fused_conv_norm_relu_kernel(
    bias_ptr,
    weight_ptr,
    ln_bias_ptr,
    ln_weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + LayerNorm + ReLU kernel for 1x1 convolutions on tensors of shape [batch, channels, 1, 1]"""
    
    # Each program handles one batch-channel pair
    pid = tl.program_id(0)
    batch_idx = pid // out_channels
    channel_idx = pid % out_channels
    
    if batch_idx >= batch_size or channel_idx >= out_channels:
        return
    
    # Load bias (scalar per output channel)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Load convolution output (scalar per input channel)
    conv_out = 0.0
    for in_c in range(in_channels):
        weight = tl.load(weight_ptr + channel_idx * in_channels + in_c)
        input_val = tl.load(input_ptr + batch_idx * in_channels + in_c)
        conv_out += weight * input_val
    conv_out += bias
    
    # Load layer norm parameters
    ln_bias = tl.load(ln_bias_ptr + channel_idx)
    ln_weight = tl.load(ln_weight_ptr + channel_idx)
    
    # Apply LayerNorm and ReLU
    normalized = ln_weight * (conv_out + ln_bias)
    relu_out = tl.max(normalized, 0.0)
    
    # Store result
    tl.store(output_ptr + batch_idx * out_channels + channel_idx, relu_out)

@torch.fx.wrap
def fused_conv_norm_relu(in_0, in_1, in_2, in_3, in_4):
    """Execute the fused Conv2D + LayerNorm + ReLU operation"""
    
    # Get input shapes
    batch_size, in_channels = in_4.shape[0], in_4.shape[1]
    out_channels = in_0.shape[0]
    
    # Check for 1x1 spatial dimensions
    assert in_4.shape[2] == 1 and in_4.shape[3] == 1, f"Expected 1x1 spatial dimensions, got {in_4.shape}"
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels), dtype=in_4.dtype, device=in_4.device)
    
    # Set up grid and launch kernel
    grid_size = batch_size * out_channels
    fused_conv_norm_relu_kernel[grid_size](
        bias_ptr=in_0,
        weight_ptr=in_1,
        ln_bias_ptr=in_2,
        ln_weight_ptr=in_3,
        input_ptr=in_4,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        BLOCK_SIZE=32
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_norm_relu