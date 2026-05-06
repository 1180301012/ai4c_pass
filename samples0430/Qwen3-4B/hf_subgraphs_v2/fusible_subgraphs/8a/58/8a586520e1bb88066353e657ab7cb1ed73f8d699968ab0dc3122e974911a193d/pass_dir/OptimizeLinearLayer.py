import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    tmp_4 = in_2.transpose(-2, -1)
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_linear_kernel(
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < N
    
    in_3 = tl.load(in_3_ptr + offset, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offset, mask=mask, other=0.0)
    in_0 = tl.load(in_0_ptr + offset, mask=mask, other=0.0)
    
    out = in_3 + in_1 + in_0
    tl.store(out_ptr + offset, out, mask=mask)

@torch.fx.wrap
def optimized_linear(in_0, in_1, in_2, in_3):
    N = in_3.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_3)
    optimized_linear_kernel[(num_blocks,)](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.permute(0, 3, 1, 2), in_2.transpose(-2, -1)

def replacement_func():
    return optimized_linear