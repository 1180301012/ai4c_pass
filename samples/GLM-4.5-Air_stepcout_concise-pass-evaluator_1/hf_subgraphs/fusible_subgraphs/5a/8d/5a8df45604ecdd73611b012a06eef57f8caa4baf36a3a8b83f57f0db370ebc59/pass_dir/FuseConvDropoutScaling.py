import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, scaling_factor, dropout_input, conv_input_2):
    """
    Pattern that matches:
    1. Conv2D operation with 1x1 kernel
    2. No-op dropout with 0.0 probability
    3. Scaling with unsqueeze operations
    4. Element-wise addition with another tensor
    """
    # Conv2D operation
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Dropout with 0.0 probability is essentially a no-op
    dropout_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    
    # Scaling operations with inefficient unsqueeze
    scaling_expanded = scaling_factor.unsqueeze(-1).unsqueeze(-1)
    scaled_out = scaling_expanded * dropout_out
    
    # Element-wise addition
    final_out = conv_input_2 + scaled_out
    
    return conv_out, final_out

def replacement_args(conv_input, conv_weight, conv_bias, scaling_factor, dropout_input, conv_input_2):
    return (conv_input, conv_weight, conv_bias, scaling_factor, conv_input_2)

@triton.jit
def fused_conv_scaling_kernel(
    conv_input_ptr, conv_weight_ptr, conv_bias_ptr,
    scaling_factor_ptr, conv_input_2_ptr,
    output_ptr,
    batch_size, channels, height, width,
    in_channels, out_channels,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate program ID
    pid = tl.program_id(0)
    
    # Conv2D parameters (1x1 kernel with stride 1, padding 0)
    kernel_size = 1
    
    # Each program handles one element in output
    output_offset = pid * BLOCK_SIZE
    output_idx = output_offset + tl.arange(0, BLOCK_SIZE)
    
    # Convert to multi-dimensional indices
    b = output_idx // (channels * height * width)
    remainder = output_idx % (channels * height * width)
    c = remainder // (height * width)
    h = (remainder % (height * width)) // width
    w = remainder % width
    
    # Create masks
    b_mask = b < batch_size
    c_mask = c < out_channels
    h_mask = h < height
    w_mask = w < width
    mask = b_mask & c_mask & h_mask & w_mask
    
    # Load input with broadcasting for 1x1 conv
    in_ptr_base = conv_input_ptr + b * in_channels * height * width + h * width + w
    conv_vals = 0.0
    for ic in range(in_channels):
        in_ptr = in_ptr_base + ic * (height * width)
        conv_vals += tl.load(in_ptr, mask=mask, other=0.0)
    
    # Load weight (1x1 conv, so we take the sum over input channels)
    weight_ptr = conv_weight_ptr + c * in_channels
    weights = tl.load(weight_ptr, mask=mask, other=0.0)
    conv_vals *= weights
    
    # Load bias
    bias_ptr = conv_bias_ptr + c
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    conv_vals += bias
    
    # Load scaling factor and expand to spatial dimensions
    scaling_ptr = scaling_factor_ptr + c
    scaling_factor = tl.load(scaling_ptr, mask=mask, other=0.0)
    conv_vals *= scaling_factor
    
    # Load second input and add
    conv_input_2_ptr_base = conv_input_2_ptr + b * channels * height * width + c * height * width + h * width + w
    conv_input_2 = tl.load(conv_input_2_ptr_base, mask=mask, other=0.0)
    final_val = conv_vals + conv_input_2
    
    # Store result
    output_ptr_base = output_ptr + b * channels * height * width + c * height * width + h * width + w
    tl.store(output_ptr_base, final_val, mask=mask)

@torch.fx.wrap
def fused_conv_scaling_function(conv_input, conv_weight, conv_bias, scaling_factor, conv_input_2):
    # Get tensor shapes
    batch_size, channels, height, width = conv_input.shape
    
    # Determine input and output channels from conv weight
    out_channels, in_channels, _, _ = conv_weight.shape
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height, width, dtype=conv_input.dtype, device=conv_input.device)
    
    # Calculate block size and grid size
    total_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate strides
    conv_input_stride = conv_input.stride()
    conv_weight_stride = conv_weight.stride()
    conv_bias_stride = conv_bias.stride()
    scaling_factor_stride = scaling_factor.stride()
    conv_input_2_stride = conv_input_2.stride()
    output_stride = output.stride()
    
    # Launch kernel
    fused_conv_scaling_kernel[(num_programs,)](
        conv_input, conv_weight, conv_bias,
        scaling_factor, conv_input_2,
        output,
        batch_size, out_channels, height, width,
        in_channels, out_channels,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv_scaling_function