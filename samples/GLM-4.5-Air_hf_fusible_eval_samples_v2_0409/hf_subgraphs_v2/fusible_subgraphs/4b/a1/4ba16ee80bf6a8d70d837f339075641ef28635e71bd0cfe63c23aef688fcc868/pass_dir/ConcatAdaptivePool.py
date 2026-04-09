import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Pattern matches concatenation followed by adaptive_avg_pool2d to (1,1)
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def concat_adaptive_pool_kernel(input_ptrs, output_ptr, c0_end, c1_end, c2_end, total_channels, height, width, BLOCK_SIZE: tl.constexpr):
    """Kernel that concatenates tensors and performs adaptive pooling in one pass"""
    pid = tl.program_id(0)
    
    # Each thread block processes one channel
    channel_idx = pid
    
    if channel_idx >= total_channels:
        return
    
    # Find which input tensor contains this channel
    tensor_idx = 0
    local_channel_idx = 0
    if channel_idx < c0_end:
        tensor_idx = 0
        local_channel_idx = channel_idx
    elif channel_idx < c1_end:
        tensor_idx = 1
        local_channel_idx = channel_idx - c0_end
    elif channel_idx < c2_end:
        tensor_idx = 2
        local_channel_idx = channel_idx - c1_end
    else:
        tensor_idx = 3
        local_channel_idx = channel_idx - c2_end
    
    # Load values from the appropriate tensor
    input_ptr = input_ptrs[tensor_idx]
    
    # Process spatial positions in blocks
    spatial_size = height * width
    spatial_blocks = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Accumulate sums for all spatial positions
    channel_sum = 0.0
    
    for block_idx in range(spatial_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load values for this spatial block
        spatial_ptr = input_ptr + local_channel_idx * height * width + offsets
        values = tl.load(spatial_ptr, mask=mask, other=0.0)
        channel_sum += tl.sum(values)
    
    # Average over spatial dimensions and store result
    channel_avg = channel_sum / spatial_size
    output_offset = channel_idx
    tl.store(output_ptr + output_offset, channel_avg)

@torch.fx.wrap
def concat_adaptive_pool(in_0, in_1, in_2, in_3):
    # Get input shapes
    batch_size = in_0.shape[0]
    h, w = in_0.shape[2], in_0.shape[3]
    
    # Calculate channel boundaries for each input tensor
    c0, c1, c2, c3 = in_0.shape[1], in_1.shape[1], in_2.shape[1], in_3.shape[1]
    total_channels = c0 + c1 + c2 + c3
    
    # Channel boundaries (cumulative sums)
    c0_end = c0
    c1_end = c0 + c1
    c2_end = c0 + c1 + c2
    # c3_end = total_channels (implicit)
    
    # Create output tensor [batch_size, total_channels, 1, 1]
    output = torch.empty((batch_size, total_channels, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Process each batch item
    for b in range(batch_size):
        # Get pointers for this batch
        input_ptrs = []
        input_ptrs.append(in_0[b].data_ptr())
        input_ptrs.append(in_1[b].data_ptr())
        input_ptrs.append(in_2[b].data_ptr())
        input_ptrs.append(in_3[b].data_ptr())
        
        # Calculate output pointer for this batch
        output_ptr = output[b].data_ptr()  # Flatten [total_channels, 1, 1]
        
        # Launch kernel
        n_programs = total_channels
        block_size = 256  # Process 256 elements per spatial position
        
        concat_adaptive_pool_kernel[(n_programs,)](
            input_ptrs=input_ptrs,
            output_ptr=output_ptr,
            c0_end=c0_end,
            c1_end=c1_end,
            c2_end=c2_end,
            total_channels=total_channels,
            height=h,
            width=w,
            BLOCK_SIZE=block_size
        )
    
    return output

def replacement_func():
    return concat_adaptive_pool