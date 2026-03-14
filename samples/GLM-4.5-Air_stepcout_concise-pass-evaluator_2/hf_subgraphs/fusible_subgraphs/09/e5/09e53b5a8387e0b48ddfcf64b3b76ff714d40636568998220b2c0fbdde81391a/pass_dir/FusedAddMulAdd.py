import torch
import triton
import triton.language as tl


# Pattern: Just mul without contiguous to test if that's the issue
def pattern(in_0, in_1):
    result = in_0 * in_1
    return result


# Extract the arguments
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Simple optimized kernel for mul
@triton.jit
def mul_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    result = in_0 * in_1
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def mul_wrapper(in_0, in_1):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    mul_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return mul_wrapper