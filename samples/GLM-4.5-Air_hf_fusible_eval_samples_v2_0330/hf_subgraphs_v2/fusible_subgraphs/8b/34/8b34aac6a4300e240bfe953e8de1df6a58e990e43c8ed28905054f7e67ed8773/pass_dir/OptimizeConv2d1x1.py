import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_0):
    """
    Pattern: 1x1 convolution with bias
    Optimization: Implement as depth-wise operation using Triton
    """
    tmp_2 = torch.conv2d(in_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_stride, channel_height_width_stride,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    
    # Each program processes a block of channels
    batch_idx = pid // (height * width)
    spatial_pid = pid % (height * width)
    h = spatial_pid // width
    w = spatial_pid % width
    channel_start = (pid // ((height * width) * (channels + BLOCK_SIZE - 1) // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Check if this program should process any channels
    if channel_start >= channels:
        return
    
    # Calculate base offsets
    input_base = batch_idx * batch_stride + h * width + w
    output_base = input_base
    bias_base = channel_start
    
    # Load bias, input, and weight data for current channel block (always load BLOCK_SIZE elements)
    bias_idx = bias_base + tl.arange(0, BLOCK_SIZE)
    input_idx = input_base + tl.arange(0, BLOCK_SIZE) * channel_height_width_stride
    weight_idx = bias_base + tl.arange(0, BLOCK_SIZE)  # 1x1 weights are flattened
    
    # Load data with masking
    bias_data = tl.load(bias_ptr + bias_idx, mask=bias_idx < channels, other=0.0)
    input_data = tl.load(input_ptr + input_idx, mask=input_idx < batch_stride, other=0.0)
    weight_data = tl.load(weight_ptr + weight_idx, mask=weight_idx < channels, other=0.0)
    
    # Apply 1x1 convolution (element-wise multiply + bias add)
    conv_result = input_data * weight_data + bias_data
    
    # Store result with proper masking
    output_idx = output_base + tl.arange(0, BLOCK_SIZE) * channel_height_width_stride
    tl.store(output_ptr + output_idx, conv_result, mask=(output_idx < batch_stride) & (bias_idx < channels))

@torch.fx.wrap
def optimized_conv2d_1x1(in_5, in_1, in_0):
    batch_size, channels, height, width = in_5.shape
    
    # Output has same shape as input for 1x1 conv with stride 1, padding 0, dilation 1
    out = torch.empty_like(in_5)
    
    # Ensure weight tensor is contiguous and flattened
    weight_flat = in_1.squeeze().contiguous().view(-1)  # Flattened 1x1 weights
    
    # Calculate strides for kernel
    batch_stride = channels * height * width
    channel_height_width_stride = height * width
    
    # Determine optimal grid size - now processing channel blocks
    BLOCK_SIZE = 128  # Number of channels processed per program
    num_channel_blocks = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_programs = batch_size * height * width * num_channel_blocks
    grid_size = total_programs
    
    # Launch kernel
    conv2d_1x1_kernel[(grid_size,)](
        in_5,
        weight_flat,
        in_0,
        out,
        batch_stride, channel_height_width_stride,
        channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_conv2d_1x1