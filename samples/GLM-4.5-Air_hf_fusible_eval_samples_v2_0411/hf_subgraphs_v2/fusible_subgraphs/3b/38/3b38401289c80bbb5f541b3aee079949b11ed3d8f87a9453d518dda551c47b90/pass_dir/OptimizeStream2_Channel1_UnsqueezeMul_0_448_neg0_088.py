import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Stream 2: Channel 1 processing
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    return tmp_6

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def stream2_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    scale: tl.constexpr,
    bias: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    in_data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out_data = in_data * scale + bias
    tl.store(out_ptr + offsets, out_data, mask=mask)

@torch.fx.wrap
def optimized_stream2(in_0):
    # Extract channel 1 and unsqueeze it (equivalent to the original pattern)
    # Shape: [batch, 1, height, width] -> [batch, 1, height, width] (unsqueeze at dim=1)
    channel1 = in_0[:, 1:2, :, :]  # This handles both indexing and unsqueezing efficiently
    
    n_elements = channel1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(channel1)
    
    stream2_kernel[(num_programs,)](
        in_ptr=channel1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        scale=0.448,
        bias=-0.08799999999999997,
    )
    
    return out

def replacement_func():
    return optimized_stream2