import torch
import triton
import triton.language as tl

def pattern(x):
    split = torch.functional.split(x, [38, 57, 57], dim=1)
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    return (tmp_6, tmp_7, tmp_8)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_split_kernel_part1(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    split_size,
    input_stride, output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate linear index for this thread
    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < (batch_size * split_size * height * width)
    
    # Calculate input indices (this handles the first split part)
    batch_idx = indices // (split_size * height * width)
    remainder = indices % (split_size * height * width)
    channel_idx = remainder // (height * width)
    h_idx = (remainder // width) % height
    w_idx = remainder % width
    
    # Input offset in the original tensor
    input_offset = (batch_idx * input_stride[0] + 
                   (channel_idx) * input_stride[1] + 
                   h_idx * input_stride[2] + w_idx * input_stride[3])
    
    # Output offset for the split part
    output_offset = (batch_idx * output_stride[0] + 
                    channel_idx * output_stride[1] + 
                    h_idx * output_stride[2] + w_idx * output_stride[3])
    
    # Copy data
    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, data, mask=mask)

@triton.jit
def optimized_split_kernel_part2(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    start_channel, split_size,
    input_stride, output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate linear index for this thread
    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < (batch_size * split_size * height * width)
    
    # Calculate input indices (for subsequent part splits)
    batch_idx = indices // (split_size * height * width)
    remainder = indices % (split_size * height * width)
    local_channel_idx = remainder // (height * width)
    h_idx = (remainder // width) % height
    w_idx = remainder % width
    
    # Map local channel to original channel index
    original_channel_idx = start_channel + local_channel_idx
    
    # Input offset in the original tensor
    input_offset = (batch_idx * input_stride[0] + 
                   original_channel_idx * input_stride[1] + 
                   h_idx * input_stride[2] + w_idx * input_stride[3])
    
    # Output offset for the split part (local channel indexing)
    output_offset = (batch_idx * output_stride[0] + 
                    local_channel_idx * output_stride[1] + 
                    h_idx * output_stride[2] + w_idx * output_stride[3])
    
    # Copy data
    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, data, mask=mask)

@torch.fx.wrap
def optimized_split_operation(x, split_sizes=(38, 57, 57)):
    batch_size, total_channels, height, width = x.shape
    split1_size, split2_size, split3_size = split_sizes
    
    # Verify that split sizes match total channels
    assert total_channels == sum(split_sizes), f"Total channels {total_channels} doesn't match split sizes {split_sizes}"
    
    # Create output tensors
    out1 = torch.empty((batch_size, split1_size, height, width), dtype=x.dtype, device=x.device)
    out2 = torch.empty((batch_size, split2_size, height, width), dtype=x.dtype, device=x.device)
    out3 = torch.empty((batch_size, split3_size, height, width), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    
    # First split: channels [0:38]
    total_elements1 = batch_size * split1_size * height * width
    num_threads1 = (total_elements1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_threads1 > 0:
        optimized_split_kernel_part1[(num_threads1,)](
            input_ptr=x,
            output_ptr=out1,
            batch_size=batch_size,
            channels=total_channels,
            height=height,
            width=width,
            split_size=split1_size,
            input_stride=x.stride(),
            output_stride=out1.stride(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Second split: channels [38:95]  
    total_elements2 = batch_size * split2_size * height * width
    num_threads2 = (total_elements2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_threads2 > 0:
        optimized_split_kernel_part2[(num_threads2,)](
            input_ptr=x,
            output_ptr=out2,
            batch_size=batch_size,
            channels=total_channels,
            height=height,
            width=width,
            start_channel=split1_size,
            split_size=split2_size,
            input_stride=x.stride(),
            output_stride=out2.stride(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Third split: channels [95:152]
    total_elements3 = batch_size * split3_size * height * width
    num_threads3 = (total_elements3 + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_threads3 > 0:
        optimized_split_kernel_part2[(num_threads3,)](
            input_ptr=x,
            output_ptr=out3,
            batch_size=batch_size,
            channels=total_channels,
            height=height,
            width=width,
            start_channel=split1_size + split2_size,
            split_size=split3_size,
            input_stride=x.stride(),
            output_stride=out3.stride(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out1, out2, out3

def replacement_func():
    return optimized_split_operation