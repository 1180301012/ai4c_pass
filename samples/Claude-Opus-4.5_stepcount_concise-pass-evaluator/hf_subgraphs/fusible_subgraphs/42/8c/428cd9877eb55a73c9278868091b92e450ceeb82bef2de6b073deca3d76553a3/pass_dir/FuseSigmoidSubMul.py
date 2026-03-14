import torch
import triton
import triton.language as tl


# Match sub -> mul pattern (after sigmoid)
def pattern(s):
    sub = s - 0.25
    mul = sub * 3.141592653589793
    return mul


def replacement_args(s):
    return (s,)


@triton.jit
def fused_sub_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    result = (x - 0.25) * 3.141592653589793
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_sub_mul(s):
    n_elements = s.numel()
    out = torch.empty_like(s)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_sub_mul_kernel[grid](
        s, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return out


def replacement_func():
    return fused_sub_mul