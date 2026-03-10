import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern 2: in_0 + in_2 with slice from in_1
    tmp_0 = in_0 + in_2
    tmp_1 = in_1[slice(None, None, None), slice(0, None, None)]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    # For pattern: in_0 + in_2 with slice from in_1  
    # The slice start is always the number of channels in the first tensor being added
    slice_start = in_0.shape[1]  # in_0 and in_2 have same shape, in_1 has more channels
    return (in_0, in_1, in_2, slice_start)

@triton.jit
def optimized_channel_slice_add_concat_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    n_batch,
    channels1,
    channels2,
    height,
    width,
    slice_start: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    # Each program handles a block of spatial and channel data
    pid_spatial = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    # Get spatial dimensions
    total_spatial = n_batch * height * width
    spatial_block_start = pid_spatial * BLOCK_SIZE_N
    spatial_block_end = min(spatial_block_start + BLOCK_SIZE_N, total_spatial)
    
    # Get channel dimensions  
    total_channels = channels1 + channels2
    channel_block_start = pid_channel * BLOCK_SIZE_C
    channel_block_end = min(channel_block_start + BLOCK_SIZE_C, total_channels)
    
    # Process spatial block
    for spatial_idx in range(spatial_block_start, spatial_block_end):
        # Convert spatial index to batch, height, width coordinates
        batch_idx = spatial_idx // (height * width)
        spatial_remainder = spatial_idx % (height * width)
        h_idx = spatial_remainder // width
        w_idx = spatial_remainder % width
        
        # Calculate base offset in output tensor
        out_base = out_ptr + (batch_idx * total_channels * height * width + 
                             h_idx * width + w_idx) * 2  # *2 for float32
        
        # Process channel block
        for channel_idx in range(channel_block_start, channel_block_end):
            output_offset = out_base + channel_idx * 2
            
            if channel_idx < channels1:
                # Part 1: in_0 + in_2 (regular addition channels)
                offset0 = batch_idx * channels1 * height * width + channel_idx * height * width + h_idx * width + w_idx
                offset2 = offset0  # in_0 and in_2 have same shape for addition
                
                val0 = tl.load(in0_ptr + offset0, mask=(batch_idx < n_batch) and (channel_idx < channels1), other=0.0)
                val2 = tl.load(in2_ptr + offset2, mask=(batch_idx < n_batch) and (channel_idx < channels1), other=0.0)
                result = val0 + val2
            else:
                # Part 2: Slice from in_1 (additional channels)
                slice_channel_idx = channel_idx - channels1
                offset1 = batch_idx * channels1 * height * width + (slice_start + slice_channel_idx) * height * width + h_idx * width + w_idx
                
                val1 = tl.load(in1_ptr + offset1, mask=(batch_idx < n_batch) and 
                              (slice_start + slice_channel_idx < in1.shape[1]), other=0.0)
                result = val1
            
            tl.store(output_offset, result, mask=(batch_idx < n_batch) and (h_idx < height) and (w_idx < width))

@torch.fx.wrap
def optimized_channel_slice_add_concat(in_0, in_1, in_2, slice_start):
    n_batch, channels1, height, width = in_0.shape
    channels2 = in_1.shape[1] - slice_start  # Number of channels to slice from in_1
    
    # Calculate output shape
    out_channels = channels1 + channels2
    out_shape = (n_batch, out_channels, height, width)
    out = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    # Optimized block sizes for better GPU utilization
    BLOCK_SIZE_N = 128  # Spatial block size per program
    BLOCK_SIZE_C = 32   # Channel block size per program
    
    # Calculate grid sizes
    total_spatial = n_batch * height * width
    total_channels = channels1 + channels2
    
    num_spatial_programs = (total_spatial + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_channel_programs = (total_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch 2D grid kernel
    optimized_channel_slice_add_concat_kernel[(num_spatial_programs, num_channel_programs)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out_ptr=out,
        n_batch=n_batch,
        channels1=channels1,
        channels2=channels2,
        height=height,
        width=width,
        slice_start=slice_start,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return out

def replacement_func():
    return optimized_channel_slice_add_concat