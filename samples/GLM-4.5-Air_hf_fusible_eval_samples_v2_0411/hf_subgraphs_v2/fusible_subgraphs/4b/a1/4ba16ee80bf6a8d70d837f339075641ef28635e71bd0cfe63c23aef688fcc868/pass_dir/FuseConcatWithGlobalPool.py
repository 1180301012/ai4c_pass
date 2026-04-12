import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match concatenation followed by global average pooling pattern
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def global_avg_pool_per_input_kernel(
    input_ptr_0,
    input_ptr_1,
    input_ptr_2,
    input_ptr_3,
    out_ptr,
    batch_size_0,
    channels_0,
    height_0,
    width_0,
    batch_size_1,
    channels_1,
    height_1,
    width_1,
    batch_size_2,
    channels_2,
    height_2,
    width_2,
    batch_size_3,
    channels_3,
    height_3,
    width_3,
    total_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel across all inputs
    channel_idx = tl.program_id(0)
    
    if channel_idx >= total_channels:
        return
    
    # Find which input and which channel within that input this corresponds to
    running_total = 0
    input_idx = 0
    channel_in_input = 0
    
    # Check input 0
    if running_total + channels_0 > channel_idx:
        input_idx = 0
        channel_in_input = channel_idx - running_total
    else:
        running_total += channels_0
        
        # Check input 1
        if running_total + channels_1 > channel_idx:
            input_idx = 1
            channel_in_input = channel_idx - running_total
        else:
            running_total += channels_1
            
            # Check input 2
            if running_total + channels_2 > channel_idx:
                input_idx = 2
                channel_in_input = channel_idx - running_total
            else:
                input_idx = 3
                channel_in_input = channel_idx - running_total
    
    # Calculate total elements for averaging and get the right shape
    if input_idx == 0:
        elements_per_channel = height_0 * width_0
        batch_size = batch_size_0
        height = height_0
        width = width_0
    elif input_idx == 1:
        elements_per_channel = height_1 * width_1
        batch_size = batch_size_1
        height = height_1
        width = width_1
    elif input_idx == 2:
        elements_per_channel = height_2 * width_2
        batch_size = batch_size_2
        height = height_2
        width = width_2
    else:  # input_idx == 3
        elements_per_channel = height_3 * width_3
        batch_size = batch_size_3
        height = height_3
        width = width_3
    
    # Calculate offset to load the channel data
    offset = (
        batch_size * channels_0 * height_0 * width_0 * input_idx +
        channel_in_input * height * width +
        tl.arange(0, BLOCK_SIZE)
    )
    
    mask = offset < batch_size * channels_0 * height_0 * width_0
    
    # Load the channel data
    if input_idx == 0:
        channel_data = tl.load(input_ptr_0 + offset, mask=mask, other=0.0)
    elif input_idx == 1:
        channel_data = tl.load(input_ptr_1 + offset, mask=mask, other=0.0)
    elif input_idx == 2:
        channel_data = tl.load(input_ptr_2 + offset, mask=mask, other=0.0)
    else:  # input_idx == 3
        channel_data = tl.load(input_ptr_3 + offset, mask=mask, other=0.0)
    
    # Compute average for this channel
    sum_val = tl.sum(channel_data)
    avg_val = sum_val / elements_per_channel
    
    # Store the result
    out_offset = channel_idx
    tl.store(out_ptr + out_offset, avg_val)

@torch.fx.wrap
def fused_concat_with_global_pool(in_0, in_1, in_2, in_3):
    # Get input shapes
    shape0 = list(in_0.shape)
    shape1 = list(in_1.shape)
    shape2 = list(in_2.shape)
    shape3 = list(in_3.shape)
    
    # Calculate total channels after concatenation
    total_channels = (shape0[1] + shape1[1] + shape2[1] + shape3[1])
    
    # Output shape: [batch_size, total_channels]
    batch_size = shape0[0]
    out = torch.empty((batch_size, total_channels), dtype=in_0.dtype, device=in_0.device)
    
    # Process each batch element
    for b in range(batch_size):
        num_channels = total_channels
        block_size = 1024
        
        # Launch kernel for this batch element
        global_avg_pool_per_input_kernel[(num_channels,)](
            input_ptr_0=in_0.data_ptr(),
            input_ptr_1=in_1.data_ptr(),
            input_ptr_2=in_2.data_ptr(),
            input_ptr_3=in_3.data_ptr(),
            out_ptr=out[b].data_ptr(),
            batch_size_0=shape0[0],
            channels_0=shape0[1],
            height_0=shape0[2],
            width_0=shape0[3],
            batch_size_1=shape1[0],
            channels_1=shape1[1],
            height_1=shape1[2],
            width_1=shape1[3],
            batch_size_2=shape2[0],
            channels_2=shape2[1],
            height_2=shape2[2],
            width_2=shape2[3],
            batch_size_3=shape3[0],
            channels_3=shape3[1],
            height_3=shape3[2],
            width_3=shape3[3],
            total_channels=total_channels,
            BLOCK_SIZE=block_size,
        )
    
    return out

def replacement_func():
    return fused_concat_with_global_pool