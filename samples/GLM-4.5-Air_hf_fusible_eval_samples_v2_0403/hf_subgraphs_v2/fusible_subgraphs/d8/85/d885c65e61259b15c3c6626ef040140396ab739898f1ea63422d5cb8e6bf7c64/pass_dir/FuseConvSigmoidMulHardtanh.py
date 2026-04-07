import torch
import triton
import triton.language as tl
from torch.fx import wrap

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matches Conv2D + Sigmoid + Multiplication + Hardtanh fusion"""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv_sigmoid_mul_hardtanh_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    scale_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized fused kernel: Conv2D + Sigmoid + Element-wise Mul + Hardtanh"""
    
    # Calculate program ID and total elements
    pid = tl.program_id(0)
    n_elements = batch_size * out_channels * height * width
    
    # Compute offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Reshape offsets to 4D: [batch, out_channels, height, width]
    batch = offsets // (out_channels * height * width)
    remainder = offsets % (out_channels * height * width)
    out_c = remainder // (height * width)
    remainder = remainder % (height * width)
    h = remainder // width
    w = remainder % width
    
    # Load bias vector
    bias = tl.load(bias_ptr + out_c, mask=out_c < out_channels, other=0.0)
    
    # Load weight tensor for current output channel and spatial position (1x1 conv)
    weight = tl.load(weight_ptr + out_c * in_channels, mask=out_c < out_channels, other=0.0)
    
    # For 1x1 convolution with stride 1, each output element depends on input at same spatial position
    input_idx = batch * in_channels * height * width + h * width + w
    input_val = tl.load(input_ptr + input_idx, mask=input_idx < (batch_size * in_channels * height * width), other=0.0)
    
    # Compute convolution output for 1x1 case
    # For 1x1 conv: output = sum(input_over_channels * weight) + bias
    conv_out = input_val * weight + bias
    
    # Apply Sigmoid
    sigmoid_out = 1.0 / (1.0 + tl.exp(-conv_out))
    
    # Load scale value for element-wise multiplication
    scale_val = tl.load(scale_ptr + batch, mask=batch < batch_size, other=0.0)
    
    # Element-wise multiplication
    mul_out = scale_val * sigmoid_out
    
    # Apply Hardtanh: clamp between 0.0 and 6.0
    hardtanh_out = tl.where(mul_out < 0.0, 0.0, tl.where(mul_out > 6.0, 6.0, mul_out))
    
    # Store result
    out_idx = batch * out_channels * height * width + out_c * height * width + h * width + w
    tl.store(output_ptr + out_idx, hardtanh_out, mask=mask)

@wrap
def fused_conv_sigmoid_mul_hardtanh(in_0, in_1, in_2, in_3):
    """Wrapper function for the fused kernel"""
    
    # Get input shapes and properties
    bias_shape = in_0.shape  # [out_channels]
    weight_shape = in_1.shape  # [out_channels, in_channels, 1, 1]
    input_shape = in_2.shape  # [batch_size, in_channels, height, width]
    scale_shape = in_3.shape  # [batch_size, out_channels, 1, 1]
    
    batch_size, in_channels, height, width = input_shape
    out_channels = bias_shape[0]
    
    # Create output tensor
    output_shape = [batch_size, out_channels, height, width]
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Determine optimal block size
    n_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 1024  # Default block size
    if n_elements < 1024:
        BLOCK_SIZE = 128
    elif n_elements < 8192:
        BLOCK_SIZE = 256
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Handle the case where this is actually a 1x1 convolution
    # For 1x1 conv with stride 1, we can optimize significantly
    if weight_shape[2] == 1 and weight_shape[3] == 1:
        fused_conv_sigmoid_mul_hardtanh_kernel[(num_programs,)](
            bias_ptr=in_0,
            weight_ptr=in_1.reshape(out_channels, in_channels),  # Reshape 2D for easier access
            input_ptr=in_2,
            scale_ptr=in_3.reshape(batch_size, out_channels),  # Reshape for broadcast
            output_ptr=output,
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fall back to separate operations for non-1x1 convolutions
        conv_out = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
        sigmoid_out = conv_out.sigmoid()
        mul_out = in_2 * sigmoid_out
        output = torch.nn.functional.hardtanh(mul_out, 0.0, 6.0, False)
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_sigmoid_mul_hardtanh