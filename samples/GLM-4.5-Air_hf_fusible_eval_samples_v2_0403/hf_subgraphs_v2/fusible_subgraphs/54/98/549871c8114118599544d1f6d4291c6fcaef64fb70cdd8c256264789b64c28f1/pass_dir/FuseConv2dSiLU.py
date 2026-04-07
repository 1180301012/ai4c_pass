import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    conv2d = torch.conv2d(x, y, z, (1, 1), (0, 0), (1, 1), 1)
    return (conv2d,)

def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID
    pid = tl.program_id(0)
    # Each program handles one output spatial location
    h = pid // width
    w = pid % width
    
    if h >= height or w >= width:
        return
    
    # Process all output channels for this spatial location
    for c_out in range(0, out_channels, BLOCK_SIZE):
        # Calculate how many channels we can actually process in this block
        max_channels_in_block = min(BLOCK_SIZE, out_channels - c_out)
        
        # Handle each channel in this block
        for c_out_idx in range(max_channels_in_block):
            c_out_actual = c_out + c_out_idx
            
            # Load bias for this output channel
            bias = tl.load(bias_ptr + c_out_actual)
            
            # Sum over input channels
            conv_result = 0.0
            for c_in in range(in_channels):
                # Load weight for this input->output channel pair
                weight_offset = (c_out_actual, c_in, 0, 0)
                weight = tl.load(weight_ptr + weight_offset)
                
                # Load input for this channel
                input_offset = (0, c_in, h, w)
                input_val = tl.load(input_ptr + input_offset)
                
                # Accumulate result
                conv_result += input_val * weight
            
            # Add bias
            conv_val = conv_result + bias
            
            # Store result for this single channel
            output_offset = (0, c_out_actual, h, w)
            tl.store(output_ptr + output_offset, conv_val)

@torch.fx.wrap
def conv2d_fused(in_0, in_1, in_2):
    # Get input dimensions
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_1.shape[0]
    
    # Determine block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid_size = height * width
    num_programs = (grid_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    conv2d_kernel[(num_programs,)](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return conv2d_fused