import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def optimized_1x1_conv_kernel(
    bias_ptr,
    weight_ptr, 
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    out_channels: tl.constexpr,
    in_channels: tl.constexpr,
    spatial_size: tl.constexpr,
):
    # Each program handles one spatial location across all channels and batches
    spatial_idx = tl.program_id(0)
    out_c = tl.program_id(1) 
    batch_idx = tl.program_id(2)
    
    # Calculate linearized offsets
    input_offset = batch_idx * in_channels * spatial_size + spatial_idx
    weight_offset = out_c * in_channels
    bias_offset = out_c
    output_offset = batch_idx * out_channels * spatial_size + spatial_idx + out_c * spatial_size
    
    # Load bias
    bias_val = tl.load(bias_ptr + bias_offset)
    
    # Perform pointwise convolution (essentially a linear combination)
    acc = bias_val
    for c in range(0, in_channels, 4):
        # Load weight and input with vectorization
        weight_vals = tl.load(weight_ptr + weight_offset + c)
        input_vals = tl.load(input_ptr + input_offset + c * spatial_size)
        acc = acc + weight_vals * input_vals
    
    # Store result
    tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def optimized_1x1_conv(bias, weight, input_tensor):
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = bias.shape[0]
    spatial_size = height * width
    
    # Create output tensor
    output = torch.zeros((batch_size, out_channels, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure kernel launch
    grid_z = batch_size
    grid_y = out_channels  
    grid_x = spatial_size
    
    # Launch kernel
    optimized_1x1_conv_kernel[(grid_x, grid_y, grid_z)](
        bias_ptr=bias,
        weight_ptr=weight, 
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        out_channels=out_channels,
        in_channels=in_channels,
        spatial_size=spatial_size,
    )
    
    return output

def replacement_func():
    return optimized_1x1_conv