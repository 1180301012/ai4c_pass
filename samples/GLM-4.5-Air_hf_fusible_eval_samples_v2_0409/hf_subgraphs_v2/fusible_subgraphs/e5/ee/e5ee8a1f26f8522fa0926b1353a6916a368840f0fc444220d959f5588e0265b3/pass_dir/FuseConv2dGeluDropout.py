import torch
import triton
import triton.language as tl

def pattern(conv_bias, conv_weight, input_tensor):
    """Match conv2d -> gelu -> dropout pattern"""
    # conv2d must use positional arguments like the original model
    conv2d_out = torch.conv2d(input_tensor, conv_weight, conv_bias, (1, 1), (1, 1), (1, 1), 128)
    gelu_out = torch.nn.functional.gelu(conv2d_out)
    dropout_out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return dropout_out

def replacement_args(conv_bias, conv_weight, input_tensor):
    """Extract arguments for the fused kernel"""
    return (conv_bias, conv_weight, input_tensor)

@triton.jit
def fused_conv2d_gelu_dropout_kernel(
    bias_ptr, weight_ptr, input_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_size, stride, padding,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: conv2d + gelu + dropout"""
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    
    # Each program handles one output pixel in all channels for one batch
    b = pid // (out_channels * out_height * out_width)
    c = (pid % (out_channels * out_height * out_width)) // (out_height * out_width)
    h = (pid % (out_height * out_width)) // out_width
    w = pid % out_width
    
    # Calculate input coordinates with padding
    input_h = h * stride - padding
    input_w = w * stride - padding
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + c * 1, mask=True)
    
    # Initialize accumulator
    acc = bias
    
    # Convolution with 1x1 kernel
    for kc in range(in_channels):
        # Calculate input pixel coordinate (since kernel is 1x1, we only need one position)
        if (0 <= input_h) and (input_h < in_height):
            if (0 <= input_w) and (input_w < in_width):
                if (0 <= kc) and (kc < in_channels):
                    input_idx = b * in_channels * in_height * in_width + kc * in_height * in_width + input_h * in_width + input_w
                    weight_idx = c * in_channels * kernel_size * kernel_size + kc * kernel_size * kernel_size + 0 * kernel_size + 0
                    
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    acc += input_val * weight_val
    
    # Apply fused GELU + Dropout (with p=0.0, dropout is no-op)
    # GELU approximation: x * 0.5 * (1.0 + erf(x * 0.70710678118))
    x = acc
    x_erf_arg = x * 0.70710678118  # x / sqrt(2)
    erf_val = tl.erf(x_erf_arg)
    gelu_out = x * 0.5 * (1.0 + erf_val)
    
    # Dropout with p=0.0 is identity, so just store the GELU output
    tl.store(output_ptr + pid, gelu_out, mask=True)

@torch.fx.wrap
def fused_conv2d_gelu_dropout(conv_bias, conv_weight, input_tensor):
    """Wrapper for the fused kernel"""
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = conv_bias.shape[0]
    kernel_size = conv_weight.shape[2]  # Assuming [out_channels, in_channels, kH, kW]
    
    # Output dimensions for 1x1 conv with stride 1, padding 1
    out_height = in_height
    out_width = in_width
    
    n_elements = batch_size * out_channels * out_height * out_width
    
    # Output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), 
                       dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv2d_gelu_dropout_kernel[(grid_size,)](
        bias_ptr=conv_bias,
        weight_ptr=conv_weight, 
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=1,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv2d_gelu_dropout