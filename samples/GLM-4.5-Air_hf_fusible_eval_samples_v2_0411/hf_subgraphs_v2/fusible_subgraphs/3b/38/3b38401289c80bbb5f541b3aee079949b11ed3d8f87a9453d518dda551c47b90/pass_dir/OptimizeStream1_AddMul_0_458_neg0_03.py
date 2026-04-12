import torch
import triton
import triton.language as tl

def pattern(in_1):
    # Stream 1: in_1 * 0.458 + (-0.030000000000000027)
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    return tmp_2

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def stream1_kernel(
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
def optimized_stream1(in_1):
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_1)
    
    stream1_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        scale=0.458,
        bias=-0.030000000000000027,
    )
    
    return out

def replacement_func():
    return optimized_stream1