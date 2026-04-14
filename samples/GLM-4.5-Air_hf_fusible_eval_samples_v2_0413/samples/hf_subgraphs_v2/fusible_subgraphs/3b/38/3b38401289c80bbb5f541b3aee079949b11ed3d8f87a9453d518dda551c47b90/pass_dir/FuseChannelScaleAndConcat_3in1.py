import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def stream1_kernel(
    out_ptr, in1_ptr,
    batch_size, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (batch_size * height * width)
    
    x = tl.load(in1_ptr + idx, mask=mask, other=0.0)
    result = x * 0.458 + (-0.030000000000000027)
    tl.store(out_ptr + idx, result, mask=mask)

@triton.jit  
def stream2_kernel(
    out_ptr, in0_ptr,
    batch_size, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (batch_size * height * width)
    
    # Load from channel 1 of in_0: offset = batch * channels * H * W + 1 * H * W + local_offset
    offset = 1 * batch_size * height * width
    x = tl.load(in0_ptr + offset + idx, mask=mask, other=0.0)
    result = x * 0.448 + (-0.08799999999999997)
    tl.store(out_ptr + idx, result, mask=mask)

@triton.jit
def stream3_kernel(
    out_ptr, in0_ptr,
    batch_size, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (batch_size * height * width)
    
    # Load from channel 2 of in_0: offset = batch * channels * H * W + 2 * H * W + local_offset
    offset = 2 * batch_size * height * width
    x = tl.load(in0_ptr + offset + idx, mask=mask, other=0.0)
    result = x * 0.45 + (-0.18799999999999994)
    tl.store(out_ptr + idx, result, mask=mask)

@torch.fx.wrap
def fused_channel_operation(in_0, in_1):
    batch_size, channels0, height, width = in_0.shape
    
    # Total elements: batch * height * width for each stream
    elements_per_stream = batch_size * height * width
    
    BLOCK_SIZE = 1024
    num_programs = (elements_per_stream + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((batch_size, 3, height, width), dtype=in_0.dtype, device=in_0.device)
    
    # Stream 1: process in_1 -> output channel 0
    stream1_kernel[(num_programs,)](
        out_ptr=out,
        in1_ptr=in_1,
        batch_size=batch_size,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Stream 2: process in_0[:, 1] -> output channel 1
    stream2_kernel[(num_programs,)](
        out_ptr=out + (batch_size * height * width),  # Second channel offset
        in0_ptr=in_0,
        batch_size=batch_size,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Stream 3: process in_0[:, 2] -> output channel 2
    stream3_kernel[(num_programs,)](
        out_ptr=out + (2 * batch_size * height * width),  # Third channel offset
        in0_ptr=in_0,
        batch_size=batch_size,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_channel_operation