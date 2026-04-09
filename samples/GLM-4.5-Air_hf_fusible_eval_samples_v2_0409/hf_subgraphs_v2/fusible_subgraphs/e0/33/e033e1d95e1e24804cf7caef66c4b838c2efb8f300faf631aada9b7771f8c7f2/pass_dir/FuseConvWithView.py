import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(-1, 256, -1)
    return tmp_3

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def fused_conv_view_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    out_channels: tl.constexpr,
    in_channels: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program handles one entire batch of output channels
    program_id = tl.program_id(0)
    batch_idx = program_id // out_channels
    out_c = program_id % out_channels
    
    # Calculate base offset for this program
    output_base = batch_idx * out_channels * spatial_size + out_c * spatial_size
    weight_base = out_c * in_channels
    bias_val = tl.load(bias_ptr + out_c)
    
    # Process one channel at a time for this batch
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < spatial_size
    
    # Initialize accumulator
    acc = bias_val
    
    # Vectorized computation across spatial dimensions
    for c in range(0, in_channels):
        # Load weight and input
        weight_val = tl.load(weight_ptr + weight_base + c)
        input_offset = batch_idx * in_channels * spatial_size + c * spatial_size + offset
        input_vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Accumulate
        acc = acc + weight_val * tl.sum(input_vals)
    
    # Store result
    output_offset = output_base + offset
    tl.store(output_ptr + output_offset, acc, mask=mask)

@triton.jit
def fused_conv_view_kernel_optimized(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    out_channels: tl.constexpr,
    in_channels: tl.constexpr,
    spatial_size: tl.constexpr,
):
    """Optimized kernel with better memory access patterns"""
    # Each program handles one spatial position for all batches and channels
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    out_c = tl.program_id(2)
    
    # Calculate offsets
    input_base = batch_idx * in_channels * spatial_size + spatial_idx
    weight_base = out_c * in_channels
    bias_base = out_c
    output_base = batch_idx * out_channels * spatial_size + out_c * spatial_size + spatial_idx
    
    # Load bias
    bias_val = tl.load(bias_ptr + bias_base)
    
    # Compute convolution
    acc = bias_val
    for c in range(0, in_channels, 4):
        # Vectorized load
        weight_vals = tl.load(weight_ptr + weight_base + c)
        input_vals = tl.load(input_ptr + input_base + c * spatial_size)
        acc = acc + weight_vals * input_vals
    
    # Store result
    tl.store(output_ptr + output_base, acc)

@torch.fx.wrap
def fused_conv_view(bias, weight, input_tensor):
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = bias.shape[0]
    spatial_size = height * width
    
    # Create output tensor
    output = torch.zeros((batch_size, out_channels, spatial_size), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose kernel based on tensor size
    if spatial_size >= 4096:  # Use optimized kernel for larger spatial sizes
        grid_x = batch_size
        grid_y = spatial_size
        grid_z = out_channels
        
        fused_conv_view_kernel_optimized[(grid_x, grid_y, grid_z)](
            bias_ptr=bias,
            weight_ptr=weight,
            input_ptr=input_tensor,
            output_ptr=output,
            batch_size=batch_size,
            out_channels=out_channels,
            in_channels=in_channels,
            spatial_size=spatial_size,
        )
    else:  # Use simpler kernel for smaller spatial sizes
        BLOCK_SIZE = 256
        programs_per_batch = out_channels
        total_programs = batch_size * programs_per_batch
        num_programs = (total_programs + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        fused_conv_view_kernel[(num_programs,)](
            bias_ptr=bias,
            weight_ptr=weight,
            input_ptr=input_tensor,
            output_ptr=output,
            batch_size=batch_size,
            out_channels=out_channels,
            in_channels=in_channels,
            spatial_size=spatial_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    return fused_conv_view