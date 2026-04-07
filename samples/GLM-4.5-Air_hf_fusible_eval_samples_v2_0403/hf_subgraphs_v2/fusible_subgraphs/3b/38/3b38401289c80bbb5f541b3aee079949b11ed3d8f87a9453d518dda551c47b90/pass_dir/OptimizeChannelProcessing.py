import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    # Branch 1: in_1 processing (multiply + add)
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    
    # Branch 2: in_0 channel 1 processing (extract + unsqueeze + multiply + add)
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    
    # Branch 3: in_0 channel 2 processing (extract + unsqueeze + multiply + add)
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    
    # Final concatenation
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    
    return tmp_11

def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Optimized Triton kernel with better performance characteristics
@triton.jit
def optimized_channel_processing_kernel(
    # Input tensors
    in_1, in_0,
    # Output tensor
    out,
    # Tensor shapes
    channels, height, width,
    # All scale and bias parameters
    scale1, bias1, scale2, bias2, scale3, bias3,
    BLOCK_SIZE: tl.constexpr,
):
    # Program id for spatial processing
    pid = tl.program_id(0)
    
    # Calculate spatial block range
    spatial_size = height * width
    block_start = pid * BLOCK_SIZE
    
    # Create linear offsets for spatial processing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size
    
    # Convert linear offsets to 2D coordinates when needed
    h_coords = offsets // width
    w_coords = offsets % width
    
    # Process each branch with fused arithmetic
    # Branch 1: Process in_1 directly (multiply + add)
    in_1_vals = tl.load(in_1 + offsets, mask=mask, other=0.0)
    out_0_vals = in_1_vals * scale1 + bias1
    
    # Branch 2: Process channel 1 from in_0 (multiply + add)
    # Calculate offset: channel 1 + spatial position
    channel1_offset = 1 * spatial_size + offsets
    in_0_1_vals = tl.load(in_0 + channel1_offset, mask=mask, other=0.0)
    out_1_vals = in_0_1_vals * scale2 + bias2
    
    # Branch 3: Process channel 2 from in_0 (multiply + add)
    # Calculate offset: channel 2 + spatial position
    channel2_offset = 2 * spatial_size + offsets
    in_0_2_vals = tl.load(in_0 + channel2_offset, mask=mask, other=0.0)
    out_2_vals = in_0_2_vals * scale3 + bias3
    
    # Store results to output in proper concatenation order
    # Output layout: [channel0, channel1, channel2] each of size [height, width]
    channel0_output_offset = 0 * spatial_size + offsets
    channel1_output_offset = 1 * spatial_size + offsets
    channel2_output_offset = 2 * spatial_size + offsets
    
    tl.store(out + channel0_output_offset, out_0_vals, mask=mask)
    tl.store(out + channel1_output_offset, out_1_vals, mask=mask)
    tl.store(out + channel2_output_offset, out_2_vals, mask=mask)

@torch.fx.wrap  
def optimized_channel_processing(in_1, in_0):
    # Get tensor shapes
    batch_size, _, height, width = in_1.shape
    _, channels, _, _ = in_0.shape
    
    # Scale and bias parameters
    scale1, bias1 = 0.458, -0.030000000000000027
    scale2, bias2 = 0.448, -0.08799999999999997
    scale3, bias3 = 0.45, -0.18799999999999994
    
    # Create output tensor: [batch, 3, height, width]
    output_shape = (batch_size, 3, height, width)
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Optimized block size for better performance
    BLOCK_SIZE = 1024
    
    # Process each batch separately
    for b in range(batch_size):
        # Extract data for current batch
        in_1_batch = in_1[b, 0].contiguous()
        in_0_batch = in_0[b].contiguous()
        out_batch = out[b].contiguous()
        
        # Calculate grid size for 1D spatial processing
        spatial_elements = height * width
        num_programs = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch 1D grid kernel
        optimized_channel_processing_kernel[(num_programs,)](
            in_1=in_1_batch,
            in_0=in_0_batch,
            out=out_batch,
            channels=channels,
            height=height,
            width=width,
            scale1=scale1, bias1=bias1,
            scale2=scale2, bias2=bias2,
            scale3=scale3, bias3=bias3,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return optimized_channel_processing