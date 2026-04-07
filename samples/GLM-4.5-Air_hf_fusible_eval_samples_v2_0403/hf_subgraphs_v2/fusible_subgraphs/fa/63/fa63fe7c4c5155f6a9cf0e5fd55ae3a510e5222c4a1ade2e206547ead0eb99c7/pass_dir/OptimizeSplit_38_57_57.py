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
def split_kernel_part1(
    in_ptr,
    out_ptr,
    batch_size,
    total_channels,
    height,
    width,
    split_size,
    stride_in_b,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    stride_out_b,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate which elements this thread handles
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * split_size * height * width
    
    # Map to input coordinates
    # input offset: b * total_channels * H * W + c * H * W + h * W + w
    # where c ranges from [0, split_size)
    
    # For the first split, c ranges from [0, split_size)
    input_idx = offsets.reshape(-1, 1, 1, 1).expand(
        -1, split_size, height, width
    )
    
    b_idx = input_idx // (split_size * height * width)
    remainder = input_idx % (split_size * height * width)
    c_idx = remainder // (height * width)
    h_idx = (remainder // width) % height
    w_idx = remainder % width
    
    # Total input offset
    in_offset = b_idx * stride_in_b + c_idx * stride_in_c + h_idx * stride_in_h + w_idx * stride_in_w
    
    # Output offset (since split doesn't change spatial dimensions)
    out_c_idx = c_idx
    out_offset = b_idx * stride_out_b + out_c_idx * stride_out_c + h_idx * stride_out_h + w_idx * stride_out_w
    
    # Load and store
    val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + out_offset, val, mask=mask)

@triton.jit
def split_kernel_part2(
    in_ptr,
    out_ptr,
    batch_size,
    start_channel,
    split_size,
    total_channels,
    height,
    width,
    stride_in_b,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    stride_out_b,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate which elements this thread handles
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * split_size * height * width
    
    # Map to input coordinates where c ranges from [start_channel, start_channel + split_size)
    input_idx = offsets.reshape(-1, 1, 1, 1).expand(
        -1, split_size, height, width
    )
    
    b_idx = input_idx // (split_size * height * width)
    remainder = input_idx % (split_size * height * width)
    c_idx = remainder // (height * width) + start_channel
    h_idx = (remainder // width) % height
    w_idx = remainder % width
    
    # Total input offset
    in_offset = b_idx * stride_in_b + c_idx * stride_in_c + h_idx * stride_in_h + w_idx * stride_in_w
    
    # Output offset (c starts from 0 for each split)
    out_c_idx = remainder // (height * width)
    out_offset = b_idx * stride_out_b + out_c_idx * stride_out_c + h_idx * stride_out_h + w_idx * stride_out_w
    
    # Load and store
    val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + out_offset, val, mask=mask)

@torch.fx.wrap
def optimized_split_38_57_57(x, splits=(38, 57, 57)):
    batch_size, total_channels, height, width = x.shape
    split1_size, split2_size, split3_size = splits
    
    assert total_channels == split1_size + split2_size + split3_size
    
    # Create output tensors
    out1 = torch.empty((batch_size, split1_size, height, width), dtype=x.dtype, device=x.device)
    out2 = torch.empty((batch_size, split2_size, height, width), dtype=x.dtype, device=x.device)
    out3 = torch.empty((batch_size, split3_size, height, width), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    total_elements1 = batch_size * split1_size * height * width
    total_elements2 = batch_size * split2_size * height * width  
    total_elements3 = batch_size * split3_size * height * width
    
    # Launch kernel for first split (channels 0-38)
    num_threads1 = (total_elements1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_threads1 > 0:
        split_kernel_part1[(num_threads1,)](
            in_ptr=x,
            out_ptr=out1,
            batch_size=batch_size,
            total_channels=total_channels,
            height=height,
            width=width,
            split_size=split1_size,
            stride_in_b=x.stride(0), stride_in_c=x.stride(1), stride_in_h=x.stride(2), stride_in_w=x.stride(3),
            stride_out_b=out1.stride(0), stride_out_c=out1.stride(1), stride_out_h=out1.stride(2), stride_out_w=out1.stride(3),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Launch kernel for second split (channels 38-95)  
    num_threads2 = (total_elements2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_threads2 > 0:
        split_kernel_part2[(num_threads2,)](
            in_ptr=x,
            out_ptr=out2,
            batch_size=batch_size,
            start_channel=split1_size,
            split_size=split2_size,
            total_channels=total_channels,
            height=height,
            width=width,
            stride_in_b=x.stride(0), stride_in_c=x.stride(1), stride_in_h=x.stride(2), stride_in_w=x.stride(3),
            stride_out_b=out2.stride(0), stride_out_c=out2.stride(1), stride_out_h=out2.stride(2), stride_out_w=out2.stride(3),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Launch kernel for third split (channels 95-152)
    num_threads3 = (total_elements3 + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_threads3 > 0:
        split_kernel_part2[(num_threads3,)](
            in_ptr=x,
            out_ptr=out3,
            batch_size=batch_size,
            start_channel=split1_size + split2_size,
            split_size=split3_size,
            total_channels=total_channels,
            height=height,
            width=width,
            stride_in_b=x.stride(0), stride_in_c=x.stride(1), stride_in_h=x.stride(2), stride_in_w=x.stride(3),
            stride_out_b=out3.stride(0), stride_out_c=out3.stride(1), stride_out_h=out3.stride(2), stride_out_w=out3.stride(3),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out1, out2, out3

def replacement_func():
    return optimized_split_38_57_57