import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match Conv2D + HardTanh + Multiply pattern"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for fused kernel"""
    return in_0, in_1, in_2, in_3

@triton.jit
def fused_conv2d_hardtanh_multiply_kernel(
    output_ptr,
    bias_ptr,
    weight_ptr,
    input_ptr,
    hardtanh_input_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    """Fused kernel for Conv2D + HardTanh + Multiply operations"""
    # Simple 1D grid: one program per output element
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * height * width
    
    # Early return if out of bounds
    if pid >= total_elements:
        return
    
    # Decode indices from flat position
    batch_idx = pid // (out_channels * height * width)
    channel_idx = (pid // (height * width)) % out_channels
    spatial_idx = pid % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + channel_idx).to(tl.float32)
    
    # Compute 1x1 convolution: sum over input channels
    conv_sum = 0.0
    for c in range(in_channels):
        # Load input value at [batch_idx, c, h_idx, w_idx]
        input_offset = batch_idx * in_channels * height * width + c * height * width + h_idx * width + w_idx
        input_val = tl.load(input_ptr + input_offset).to(tl.float32)
        
        # Load weight value at [channel_idx, c, 0, 0]
        weight_offset = channel_idx * in_channels + c
        weight_val = tl.load(weight_ptr + weight_offset).to(tl.float32)
        
        # Accumulate dot product
        conv_sum += input_val * weight_val
    
    # Add bias to get final convolution result
    conv_result = conv_sum + bias_val
    
    # Load HardTanh input and apply HardTanh activation
    hardtanh_offset = batch_idx * out_channels * height * width + channel_idx * height * width + h_idx * width + w_idx
    hardtanh_val = tl.load(hardtanh_input_ptr + hardtanh_offset).to(tl.float32)
    hardtanh_output = tl.maximum(0.0, tl.minimum(6.0, hardtanh_val))
    
    # Final element-wise multiplication
    final_output = hardtanh_output * conv_result
    
    # Store result
    tl.store(output_ptr + pid, final_output)

@torch.fx.wrap
def fused_conv2d_hardtanh_multiply(bias, weight, input_tensor, hardtanh_input):
    """Fused Conv2D + HardTanh + Multiply kernel wrapper"""
    
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = bias.shape[0]
    
    # Create output tensor
    output_shape = (batch_size, out_channels, height, width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Verify we have 1x1 weights
    if weight.shape[2:4] != (1, 1):
        raise ValueError("Only 1x1 convolution supported in this fused pass")
    
    # Total number of output elements
    total_elements = batch_size * out_channels * height * width
    
    # Launch the kernel with 1D grid
    fused_conv2d_hardtanh_multiply_kernel[(total_elements,)](
        output_ptr=output,
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        hardtanh_input_ptr=hardtanh_input,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv2d_hardtanh_multiply