import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, residual_input, skip_input, running_mean, running_var, bn_weight, bn_bias):
    # Conv2d operation (1x1 convolution)
    conv = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # First addition: skip_input + conv
    add1 = skip_input + conv
    # Second addition: residual_input + add1 (forming residual connection)
    result = residual_input + add1
    return result

def replacement_args(x, weight, bias, residual_input, skip_input, running_mean, running_var, bn_weight, bn_bias):
    return (x, weight, bias, residual_input, skip_input, running_mean, running_var, bn_weight, bn_bias)

@triton.jit
def fused_conv_add_residual_kernel(
    x_ptr, weight_ptr, bias_ptr, residual_ptr, skip_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute grid position
    pid = tl.program_id(0)
    num_blocks = (N * C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C * H * W)
    
    # Reshape offsets to 4D: [N, C, H, W]
    offset_w = offsets % W
    offset_h = (offsets // W) % H
    offset_c = (offsets // (W * H)) % C
    offset_n = offsets // (W * H * C)
    
    # Load input tensor [N, C, H, W]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load residual input [N, C, H, W]
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    
    # Load skip input [N, C, H, W]
    skip = tl.load(skip_ptr + offsets, mask=mask, other=0.0)
    
    # Convolution: for each channel, multiply by corresponding weight and add bias
    # Since it's 1x1 conv with groups=1, we process each channel independently
    conv_channels = C
    conv_weights = tl.load(weight_ptr + tl.arange(0, conv_channels), mask=tl.arange(0, conv_channels) < conv_channels)
    conv_biases = tl.load(bias_ptr + tl.arange(0, conv_channels), mask=tl.arange(0, conv_channels) < conv_channels)
    
    # Apply convolution channel-wise
    conv_out = x * conv_weights[offset_c % conv_channels] + conv_biases[offset_c % conv_channels]
    
    # Fused addition operations: result = residual + skip + conv_out
    result = residual + skip + conv_out
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_conv_add_residual(x, weight, bias, residual_input, skip_input):
    N, C, H, W = x.shape
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    fused_conv_add_residual_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight.squeeze(),  # Remove spatial dimensions for 1x1 conv
        bias_ptr=bias,
        residual_ptr=residual_input,
        skip_ptr=skip_input,
        out_ptr=output,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv_add_residual