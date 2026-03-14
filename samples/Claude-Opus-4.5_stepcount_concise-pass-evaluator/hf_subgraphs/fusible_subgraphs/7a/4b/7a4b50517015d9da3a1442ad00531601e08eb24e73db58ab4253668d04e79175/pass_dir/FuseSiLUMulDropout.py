import torch
import triton
import triton.language as tl

# Pattern matching - just multiply
def pattern(in_0, in_1):
    tmp_1 = in_0 * in_1
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel for Multiply - fixed config for 263168 elements
@triton.jit
def mul_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    result = x * y
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def triton_mul(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul_kernel[grid](
        in_0,
        in_1,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_mul