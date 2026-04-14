import torch
import triton
import triton.language as tl

# Pattern matching function for simple 1x1 conv2d
def pattern(in_3, in_1, in_0):
    # 1x1 convolution with specific parameters
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

# Argument extraction function
def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Triton kernel for efficient 1x1 convolution
@triton.jit
def fast_conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, out_channels,
    input_height, input_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a batch.channel pair
    pid = tl.program_id(0)
    total_bc = batch_size * out_channels
    block_start = pid * BLOCK_SIZE
    bc_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = bc_offsets < total_bc
    
    # Reshape offsets to batch and channel
    batch_offsets = bc_offsets // out_channels
    out_channel_offsets = bc_offsets % out_channels
    
    # Process for each spatial location (1x1 conv independent of spatial dims)
    for h in range(input_height):
        for w in range(input_width):
            # Load weight and bias for each output channel
            weight_val = tl.load(weight_ptr + out_channel_offsets, mask=mask)
            bias_val = tl.load(bias_ptr + out_channel_offsets, mask=mask)
            
            # Load input data - in this case, since it's 1x1 conv with in_channels=out_channels,
            # we assume the input and output have same number of channels
            input_val = tl.load(input_ptr + batch_offsets * input_height * input_width + out_channel_offsets, 
                               mask=mask)
            
            # Compute 1x1 convolution
            output_val = input_val * weight_val + bias_val
            
            # Store result
            output_offset = batch_offsets * input_height * input_width + out_channel_offsets
            tl.store(output_ptr + output_offset, output_val, mask=mask)

@torch.fx.wrap
def fast_conv1x1(in_3, in_1, in_0):
    # Get tensor shapes
    batch_size, in_channels, input_height, input_width = in_3.shape
    out_channels = in_1.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, input_height, input_width), 
                        dtype=in_3.dtype, device=in_3.device)
    
    # Calculate grid size - each program handles one batch + output channel pair
    total_bc_elements = batch_size * out_channels
    BLOCK_SIZE = 512  # Optimized block size for better occupancy
    num_programs = (total_bc_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fast_conv1x1_kernel[(num_programs,)](
        input_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fast_conv1x1