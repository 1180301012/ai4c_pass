import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    conv2d = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def optimized_conv_reshape_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_size, stride, padding, dilation, groups,
    out_height, out_width,
    BLOCK_SIZE_M: tl.constexpr
):
    # Compute output dimensions
    total_elements = batch_size * out_channels * out_height * out_width
    block_start = tl.program_id(0) * BLOCK_SIZE_M
    offsets = block_start + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < total_elements
    
    # Convert linear offset to 4D coordinates
    w = offsets % out_width
    h = (offsets // out_width) % out_height
    c = (offsets // (out_width * out_height)) % out_channels
    b = offsets // (out_width * out_height * out_channels)
    
    # Load bias
    bias_val = tl.load(bias_ptr + c)
    
    # Initialize output with bias
    out = bias_val
    
    # For 1x1 convolution: output[b, c, h, w] = bias[c] + sum_{k=0}^{in_channels-1} weight[c, k, 0, 0] * input[b, k, h, w]
    for k in range(0, in_channels):
        # Load weight for output channel c and input channel k
        weight_idx = c * in_channels + k
        weight_val = tl.load(weight_ptr + weight_idx)
        
        # Load input at position [b, k, h, w]
        input_idx = (b * in_channels + k) * in_height * in_width + h * in_width + w
        input_val = tl.load(x_ptr + input_idx, mask=mask, other=0.0)
        
        # Accumulate
        out += weight_val * input_val
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_conv_reshape(x, weight, bias):
    # Input shapes
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Output dimensions for 1x1 conv with stride 1, padding 0
    out_height = in_height
    out_width = in_width
    
    # Output shape from conv: [batch_size, out_channels, out_height, out_width]
    # Then reshape to [-1, 17, 4096]
    # We need to ensure the reshape is mathematically valid
    conv_output_elements = batch_size * out_channels * out_height * out_width
    total_flattened_elements = 17 * 4096
    
    # Calculate the first dimension of the reshape
    first_dim = conv_output_elements // total_flattened_elements
    reshape_target = (first_dim, 17, 4096)
    
    # Create output tensor
    out = torch.empty(reshape_target, dtype=x.dtype, device=x.device)
    
    # Block size for optimal performance
    BLOCK_SIZE_M = 1024
    
    # Number of programs needed
    num_programs = (conv_output_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    optimized_conv_reshape_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out.view(-1),  # Flatten for kernel processing
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out

def replacement_func():
    return optimized_conv_reshape