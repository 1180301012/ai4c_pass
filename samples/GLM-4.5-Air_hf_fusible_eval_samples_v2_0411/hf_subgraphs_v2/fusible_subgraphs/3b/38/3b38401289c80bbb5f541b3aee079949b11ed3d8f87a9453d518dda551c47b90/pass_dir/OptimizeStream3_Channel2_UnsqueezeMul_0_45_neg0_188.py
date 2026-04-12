import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Stream 3: Channel 2 processing
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    return tmp_10

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def stream3_kernel(
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
def optimized_stream3(in_0):
    # Extract channel 2 and unsqueeze it (equivalent to the original pattern)
    # Shape: [batch, 1, height, width] -> [batch, 1, height, width] (unsqueeze at dim=1)
    channel2 = in_0[:, 2:3, :, :]  # This handles both indexing and unsqueezing efficiently
    
    n_elements = channel2.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(channel2)
    
    stream3_kernel[(num_programs,)](
        in_ptr=channel2,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        scale=0.45,
        bias=-0.18799999999999994,
    )
    
    return out

def replacement_func():
    return optimized_stream3