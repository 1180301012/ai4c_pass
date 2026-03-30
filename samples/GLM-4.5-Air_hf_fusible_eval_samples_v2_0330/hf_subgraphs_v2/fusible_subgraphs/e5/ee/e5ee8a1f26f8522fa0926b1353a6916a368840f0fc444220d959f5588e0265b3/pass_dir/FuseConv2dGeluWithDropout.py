import torch
import triton
import triton.language as tl

def pattern(conv_bias, conv_weight, conv_input):
    """Pattern matching for Conv2D + GELU + Dropout (no-op)"""
    # Use dynamic groups parameter to match any group size
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (1, 1), (1, 1), conv_bias.shape[0])
    gelu_out = torch.nn.functional.gelu(conv_out)
    dropout_out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return dropout_out

def replacement_args(conv_bias, conv_weight, conv_input):
    """Extract arguments for the fused kernel"""
    return (conv_bias, conv_weight, conv_input)

@triton.jit
def fused_conv2d_gelu_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simplified fused Conv2D + GELU kernel using Triton"""
    
    pid = tl.program_id(0)
    
    # Compute current work item
    batch_idx = pid // (out_channels * height * width)
    out_ch_idx = (pid % (out_channels * height * width)) // (height * width)
    h_idx = (pid % (height * width)) // width
    w_idx = pid % width
    
    # Output pointer for this element
    output_idx = batch_idx * out_channels * height * width + out_ch_idx * height * width + h_idx * width + w_idx
    
    # Initialize accumulator
    acc = 0.0
    
    # Load weight 
    weight_ptrs = weight_ptr + (n_offsets[:, None] * in_channels * 3 * 3 + 
                               (tl.arange(0, 3)[None, :, None] * 3 + 
                                tl.arange(0, 3)[:, None, None] * in_channels +
                                tl.arange(0, in_channels)[None, None, :]))
    weight = tl.load(weight_ptrs, mask=(n_offsets[:, None] < out_channels) & 
                                      (tl.arange(0, 3)[:, None] < 3) & 
                                      (tl.arange(0, 3)[None, :] < 3) & 
                                      (tl.arange(0, in_channels)[None, None, :] < in_channels), other=0.0)
    
    # Load input data for the current block
    input_ptrs = input_ptr + (m_offsets[:, None, None] * in_channels * height * width +
                             (tl.arange(0, 3)[None, :, None] * height + height//2 - 1) * width + 
                             (tl.arange(0, 3)[:, None] * width + width//2 - 1) * in_channels +
                             tl.arange(0, in_channels)[None, None, :])
    input_data = tl.load(input_ptrs, 
                        mask=(m_offsets[:, None, None] < batch_size) &
                             (tl.arange(0, 3)[None, :, None] < height) &
                             (tl.arange(0, 3)[:, None] < width) &
                             (tl.arange(0, in_channels)[None, None, :] < in_channels), other=0.0)
    
    # Perform 3x3 convolution with padding=1
    
    
    
    # Add bias
    acc += tl.load(bias_ptr + out_ch_idx, other=0.0)
    
    # Apply GELU activation using approximation
    # GELU(x) ≈ x * 0.5 * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x = acc
    gelu_approx = x * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    
    # Store result
    tl.store(output_ptr + output_idx, gelu_approx)

@torch.fx.wrap
def fused_conv2d_gelu(conv_bias, conv_weight, conv_input):
    """Wrapper function for the fused Conv2D + GELU kernel"""
    
    # Get input dimensions
    batch_size = conv_input.shape[0]
    in_channels = conv_input.shape[1]
    height = conv_input.shape[2] 
    width = conv_input.shape[3]
    out_channels = conv_bias.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), 
                       dtype=conv_input.dtype, device=conv_input.device)
    
    # Total number of output elements
    total_elements = batch_size * out_channels * height * width
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Grid computation
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv2d_gelu_kernel[grid](
        bias_ptr=conv_bias,
        weight_ptr=conv_weight,
        input_ptr=conv_input,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv2d_gelu