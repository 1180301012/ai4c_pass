import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    """Pattern to match the concatenation operation specifically"""
    return torch.cat((in_2, in_3), 1)

def replacement_args(in_2, in_3):
    return (in_2, in_3)

@triton.jit
def high_perf_concat_kernel(
    in2_ptr, in3_ptr, out_ptr,
    total_elements,
    batch_size, channels2, channels3, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance concatenation kernel with vectorized loads"""
    # Each program handles a contiguous block of data like the reference
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Handle blocks of elements efficiently  
    if tl.all(mask):  # All offsets in range
        for offset in offsets:
            # Calculate coordinates from linear index
            batch_idx = offset // ((channels2 + channels3) * height * width)
            remainder = offset % ((channels2 + channels3) * height * width)
            channel_idx = remainder // (height * width)
            h_idx = (remainder % (height * width)) // width
            w_idx = remainder % width
            
            # Determine source tensor and channel
            if channel_idx < channels2:
                src_channel = channel_idx
                src_offset = batch_idx * channels2 * height * width + src_channel * height * width + h_idx * width + w_idx
                val = tl.load(in2_ptr + src_offset)
            else:
                src_channel = channel_idx - channels2
                src_offset = batch_idx * channels3 * height * width + src_channel * height * width + h_idx * width + w_idx
                val = tl.load(in3_ptr + src_offset)
            
            # Store in output
            out_offset = batch_idx * (channels2 + channels3) * height * width + channel_idx * height * width + h_idx * width + w_idx
            tl.store(out_ptr + out_offset, val)
    else:
        # Handle boundary cases
        for i, offset in enumerate(offsets):
            if mask[i]:
                # Same logic as above for individual elements
                batch_idx = offset // ((channels2 + channels3) * height * width)
                remainder = offset % ((channels2 + channels3) * height * width)
                channel_idx = remainder // (height * width)
                h_idx = (remainder % (height * width)) // width
                w_idx = remainder % width
                
                if channel_idx < channels2:
                    src_channel = channel_idx
                    src_offset = batch_idx * channels2 * height * width + src_channel * height * width + h_idx * width + w_idx
                    val = tl.load(in2_ptr + src_offset)
                else:
                    src_channel = channel_idx - channels2
                    src_offset = batch_idx * channels3 * height * width + src_channel * height * width + h_idx * width + w_idx
                    val = tl.load(in3_ptr + src_offset)
                
                out_offset = batch_idx * (channels2 + channels3) * height * width + channel_idx * height * width + h_idx * width + w_idx
                tl.store(out_ptr + out_offset, val)

@torch.fx.wrap
def high_perf_concat(in_2, in_3):
    """High-performance concatenation function"""
    batch_size, channels2, height, width = in_2.shape
    channels3 = in_3.shape[1]
    
    out = torch.empty((batch_size, channels2 + channels3, height, width), 
                     dtype=in_2.dtype, device=in_2.device)
    
    # Use efficient grid configuration
    total_elements = batch_size * (channels2 + channels3) * height * width
    BLOCK_SIZE = 1024  # Process 1024 elements per program
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    high_perf_concat_kernel[(num_programs,)](
        in_2, in_3, out,
        total_elements,
        batch_size, channels2, channels3, height, width,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return high_perf_concat