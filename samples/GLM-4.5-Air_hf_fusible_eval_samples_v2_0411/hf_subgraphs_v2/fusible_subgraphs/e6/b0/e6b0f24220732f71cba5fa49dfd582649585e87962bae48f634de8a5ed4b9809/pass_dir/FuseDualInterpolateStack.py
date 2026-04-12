import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching: two interpolate operations + concatenation + stack"""
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode='nearest')
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return (tmp_3,)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the optimized kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_interpolate_kernel(
    input0_ptr, input1_ptr, input2_ptr, input3_ptr,
    output_ptr,
    batch_size, channels0, channels1, channels23,
    h0, w0, h1, w1,
    out_channels, out_height, out_width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Fused kernel that performs dual interpolation and concatenation"""
    pid = tl.program_id(0)
    batch_idx = pid // out_channels
    channel_idx = pid % out_channels
    
    # Initialize output tensor
    offset = batch_idx * (out_channels * out_height * out_width) + channel_idx * out_height * out_width
    out_ptr = output_ptr + offset
    
    # Determine which operation to perform based on channel index
    if channel_idx < channels0:
        # Process in_0 interpolation
        src_channel = channel_idx
        # Compute source coordinates for nearest neighbor interpolation
        scale_h = h0 / out_height if h0 != out_height else 1.0
        scale_w = w0 / out_width if w0 != out_width else 1.0
        
        for i in range(out_height):
            for j in range(out_width):
                # Nearest neighbor: find closest source coordinate
                src_i = int(i * scale_h)
                src_j = int(j * scale_w)
                src_i = max(0, min(h0 - 1, src_i))
                src_j = max(0, min(w0 - 1, src_j))
                
                # Load input value
                src_offset = batch_idx * (channels0 * h0 * w0) + src_channel * (h0 * w0) + src_i * w0 + src_j
                val = tl.load(input0_ptr + src_offset)
                
                # Store output value
                dst_offset = i * out_width + j
                tl.store(out_ptr + dst_offset, val)
                
    elif channel_idx < channels0 + channels1:
        # Process in_1 interpolation  
        src_channel = channel_idx - channels0
        scale_h = h1 / out_height if h1 != out_height else 1.0
        scale_w = w1 / out_width if w1 != out_width else 1.0
        
        for i in range(out_height):
            for j in range(out_width):
                src_i = int(i * scale_h)
                src_j = int(j * scale_w)
                src_i = max(0, min(h1 - 1, src_i))
                src_j = max(0, min(w1 - 1, src_j))
                
                src_offset = batch_idx * (channels1 * h1 * w1) + src_channel * (h1 * w1) + src_i * w1 + src_j
                val = tl.load(input1_ptr + src_offset)
                
                dst_offset = i * out_width + j
                tl.store(out_ptr + dst_offset, val)
                
    else:
        # Process concatenated in_2 + in_3
        concat_channel_idx = channel_idx - channels0 - channels1
        
        if concat_channel_idx < channels23:
            src_channel = concat_channel_idx
            # Direct copy since in_2 and in_3 are already 40x40
            if src_channel < channels23 // 2:
                # From in_2
                src_in_idx = src_channel
                src_offset = batch_idx * (channels23 // 2 * out_height * out_width) + src_in_idx * (out_height * out_width)
                for i in range(out_height):
                    for j in range(out_width):
                        src_pos = src_offset + i * out_width + j
                        dst_pos = i * out_width + j
                        val = tl.load(input2_ptr + src_pos)
                        tl.store(out_ptr + dst_pos, val)
            else:
                # From in_3
                src_in_idx = src_channel - channels23 // 2
                src_offset = batch_idx * (channels23 // 2 * out_height * out_width) + src_in_idx * (out_height * out_width)
                for i in range(out_height):
                    for j in range(out_width):
                        src_pos = src_offset + i * out_width + j
                        dst_pos = i * out_width + j
                        val = tl.load(input3_ptr + src_pos)
                        tl.store(out_ptr + dst_pos, val)

@torch.fx.wrap
def fused_interpolate_func(in_0, in_1, in_2, in_3):
    """Main function that computes all outputs and stacks them"""
    batch_size, channels0, h0, w0 = in_0.shape
    _, channels1, h1, w1 = in_1.shape  
    _, channels23_half, _, _ = in_2.shape
    channels23 = channels23_half * 2  # from concatenation
    
    # Final concatenated channels: channels0 + channels1 + channels23
    out_channels = channels0 + channels1 + channels23
    out_height, out_width = 40, 40
    
    # Allocate output tensor [batch, out_channels, out_height, out_width]
    output = torch.empty((batch_size, out_channels, out_height, out_width), 
                        dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    total_elements = batch_size * out_channels
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_interpolate_kernel[(num_programs,)](
        in_0, in_1, in_2, in_3,
        output,
        batch_size, channels0, channels1, channels23,
        h0, w0, h1, w1,
        out_channels, out_height, out_width,
        BLOCK_SIZE
    )
    
    # Stack into final [3, batch, out_channels//3, 40, 40] format
    # Split output into three equal parts
    out_part0 = output[:, :channels0, :, :]      # interpolated in_0
    out_part1 = output[:, channels0:channels0+channels1, :, :]  # interpolated in_1  
    out_part2 = output[:, channels0+channels1:, :, :]  # concatenated in_2+in_3
    
    # Stack along new dimension at front
    final_output = torch.stack([out_part0, out_part1, out_part2], dim=0)
    
    return final_output

def replacement_func():
    return fused_interpolate_func