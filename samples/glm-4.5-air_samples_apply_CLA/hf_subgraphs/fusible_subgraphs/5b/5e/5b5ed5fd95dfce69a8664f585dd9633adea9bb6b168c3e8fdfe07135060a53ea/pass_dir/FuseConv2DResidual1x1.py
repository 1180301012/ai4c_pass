import torch
import triton
import triton.language as tl

def pattern(conv_weight, conv_bias, x):
    # 1x1 conv followed by residual addition
    conv_out = torch.conv2d(x, conv_weight, conv_bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    return conv_out + x

def replacement_args(conv_weight, conv_bias, x):
    return (conv_weight, conv_bias, x)

@triton.jit
def fused_conv_residual_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, height, width, out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location across all channels
    pid = tl.program_id(0)
    num_programs = tl.cdiv(height * width, BLOCK_SIZE)
    
    # Block processing
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width
    
    # Reshape offsets to 2D spatial coordinates
    h = offsets // width
    w = offsets % width
    
    # Process each output channel
    for c in range(out_channels):
        # Load input feature for this spatial location and input channels
        x_base_ptr = x_ptr + batch_size * in_channels * height * width
        x_ptrs = x_base_ptr + (h * width + w) * in_channels + tl.arange(0, in_channels)
        x_vals = tl.load(x_ptrs, mask=tl.arange(in_channels) < in_channels, other=0.0)
        
        # Load conv weight (1x1, so spatial dimension is 1x1)
        weight_ptr_1x1 = weight_ptr + c * in_channels * 1 * 1
        weights = tl.load(weight_ptr_1x1 + tl.arange(0, in_channels), mask=tl.arange(in_channels) < in_channels, other=0.0)
        
        # Load bias
        bias_val = tl.load(bias_ptr + c, other=0.0)
        
        # Convolution operation (1x1, so just channel-wise multiply and accumulate)
        conv_val = tl.sum(x_vals * weights) + bias_val
        
        # Add residual (load original x value)
        residual_val = tl.load(x_ptr + batch_size * in_channels * height * width + (h * width + w) * in_channels + c, other=0.0)
        
        # Store result
        out_ptr_idx = batch_size * out_channels * height * width + (h * width + w) * out_channels + c
        tl.store(out_ptr + out_ptr_idx, conv_val + residual_val, mask=mask)

@torch.fx.wrap
def fused_conv_residual(conv_weight, conv_bias, x):
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.size(0)
    
    BLOCK_SIZE = 256  # Optimal block size for spatial dimension
    num_programs = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_conv_residual_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=conv_weight,
        bias_ptr=conv_bias,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        out_channels=out_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_conv_residual